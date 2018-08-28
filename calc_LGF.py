import time, scipy.io, argparse, logging
import numpy as np
import scipy.sparse.linalg as sla
import h5py
import elastic
import setup
import IO_xyz


def setBC(i,grid,size_in,size_all,GEn,phi_R_grid,N,a0,t_mag,f):

    """
    Use the large R LGF (= EGF) to displace atoms in the far-field boundary
    due to a point force on atom i in region 2.
    
    Parameters
    ----------
    i          : my atom index of the atom on which the point force is applied
    grid       : list of namedtuples containing atom info:
                 index, region, m-coord, n-coord, t-coord, basis 
    size_in    : number of atoms in regions 1+2+3+buffer
    size_all   : total number of atoms in the system, including the far-field boundary region
    GEn        : list of complex ndarrays of fourier coefficients evaluated for each component of EGF
    phi_R_grid : list of angular terms in the real space large R LGF, computed for 
                 N equally-spaced angular (phi) values. Same as GEn, each entry in 
                 the list corresponds to the values for the different components of LGF
    N          : number of angular values for which the angular term in the real space large R LGF 
                 has been explicitly computed
    a0         : lattice constant in angstroms
    t_mag      : magnitude of the periodic vector along the dislocation threading direction
    f          : array of shape (size_in,3) for forces
          
    Returns
    -------
    u : array of shape (size_all,3) for displacements of atoms
        in the far-field boundary region
    
    """ 
    
    u = np.zeros((size_all,3))  
    for atom_ff,u_ff in zip(grid[size_in:],u[size_in:]):
        ## rvec is the vector between the 2 atoms in terms of mnt
        rvec = np.array([grid[i].m,grid[i].n,grid[i].t]) - np.array([atom_ff.m,atom_ff.n,atom_ff.t])
        ## R is the R_perp in the mn plane
	## CHANGED BY DRT: no longer scaled by a0.
        # R = a0*np.sqrt(rvec[0]**2 + rvec[1]**2)
        R = np.sqrt(rvec[0]**2 + rvec[1]**2)
        ## phi is the angle wrt +m axis
        phi = np.arctan2(rvec[1],rvec[0])
        if (abs(phi) > 1E-8):
            phi = phi%(2*np.pi)
        else:
            phi = 0.                
        ## given R and phi, calculate G(large R limit) 
        ## and use it to get displacements u = - G.f
        u_ff -= np.dot(elastic.G_largeR(GEn,phi_R_grid,R,phi,N,a0,t_mag),f[i]) 
        
    return u


if __name__ == '__main__':

                     
    parser = argparse.ArgumentParser(description='Computes the dislocation lattice Green function.')
    parser.add_argument('inputfile',
                        help='input file that contains the crystal and dislocation setup info')
    parser.add_argument('atomxyzfile',
                        help='xyz file that contains the atom positions')
    parser.add_argument('-atomlabel', action='append', required=True,
                        help='name label for each basis atom type as used in xyz file; '
                        'may be passed multiple times as required. '
                        'Place the flag -atomlabel before each entry. '
                        'Despite the flag, this is a REQUIRED (not optional) argument!')
    parser.add_argument('Dfile',
                        help='HDF5 file to read the FC matrix D from')
    parser.add_argument('Gfile',
                        help='HDF5 file to save the computed G to')
    parser.add_argument('-logfile',
                        help='logfile to save to')
    parser.add_argument('-LGF_jmin',type=int,
                        help='(int) first atom index to compute LGF for. '
                        'Default is the first atom in region 2.')
    parser.add_argument('-LGF_jmax',type=int,
                        help='(int) last atom index to compute LGF for. '
                        'Default is the last atom in region 2. '
                        'Note! Atom indices are based on 0-based indexing.')
    parser.add_argument('-tol',type=float,
                        help='(float) tolerance for relaxation.',
                        default=1e-5)
    
       
    ## read in the above arguments from command line
    args = parser.parse_args()   
    
    ## set up logging
    if args.logfile:
        logging.basicConfig(filename=args.logfile,filemode='w',format='%(levelname)s:%(message)s', level=logging.DEBUG)    
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(levelname)s:%(message)s'))
        logging.getLogger('').addHandler(console)
    else:
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        
    ## read in setup details
    """
    crystalclass: crystal class (4=cubic; 9=hexagonal; 10=isotropic)    
    a0          : lattice constant (angstroms)
    Cijs        : list of Cijs (GPa)
    M           : 3x3 matrix for rotating from mnt basis to cartesian basis
                  (columns are normalized m,n,t vectors)
    t_mag       : magnitude of the periodic vector along the dislocation threading direction
                  
    """
    with open(args.inputfile,'r') as f1:
        crystalclass,a0,Cijs,M,t_mag = setup.readinputs(f1)
    
    ## read in grid of atoms
    """
    grid : list of namedtuples containing atom info
           [index, region, m-coord, n-coord, t-coord, basis]
    size_1,size_12,size_123,size_in,size_all: cumulative # atoms in each of the regions
    
    """
    with open(args.atomxyzfile,'r') as f2:
        grid,[size_1,size_12,size_123,size_in] = IO_xyz.grid_from_xyz_reg(f2.read(),args.atomlabel,a0)
    size_all = len(grid)
    logging.info('System setup: size_1 = %d, size_12 = %d, size_123 = %d, size_in = %d, size_all = %d'
                  %(size_1,size_12,size_123,size_in,size_all))

    t0 = time.time()

    ## load the big D matrix from file
    logging.info('loading D...')
    # D = scipy.io.mmread(args.Dfile).tocsr()
    with h5py.File(args.Dfile, 'r') as f:
        # need to be explicit with *shape* in case we have disconnected
        # atoms in the far-field:
        D = scipy.sparse.csr_matrix((f['data'], f['indices'], f['indptr']),
                shape=(size_in*3, size_all*3))

    ## construct the 3x3x3x3 elastic stiffness tensor  
    ## convert elastic constants from GPa to eV/A^3
    C = elastic.convert_from_GPa(elastic.expand_C(elastic.construct_C(crystalclass,Cijs)))
    # print(C)
    
    ## assemble the pieces necessary to evaluate the large R LGF, i.e. EGF
    ## based on the expression found in D.R. Trinkle, Phys. Rev. B 78, 014110 (2008)
    N = 256
    GEn = elastic.EGF_Fcoeffs(N,C,M)
    phi_R_grid = elastic.G_largeR_ang(GEn,N,N_max=int(N/2))

    ## compute the LGF matrix
    if args.LGF_jmin: 
        LGF_jmin = args.LGF_jmin
    else: 
        LGF_jmin = size_1  ## if LGF_jmin not specified, default = size_1
    if args.LGF_jmax: 
        LGF_jmax = args.LGF_jmax
    else: 
        LGF_jmax = size_12-1  ## if LGF_jmax not specified, default = size_12-1

    Din = D[0:3*size_in, 0:3*size_in]
    # Din = 0.5*(D[0:3*size_in, 0:3*size_in] + D[0:3*size_in, 0:3*size_in].T)
    
    logging.info('Looping through atoms...')
    direction = 'mnt'
    ## loop through every atom and every direction in reg 2
    for j in range(LGF_jmin,LGF_jmax+1):
        for d in range(0,3):
            f = np.zeros((3*size_in,))
            
            ## apply a force in an atom in reg 2
            f[j*3+d] = 1

            ## displace atoms in far-field boundary according to u_bc = -EGF.f(II)
            logging.debug('  EGF displacement of {} along {}'.format(j, direction[d]))
            u_bc = setBC(j,grid,size_in,size_all,GEn,phi_R_grid,N,a0,t_mag,np.reshape(f,(size_in,3)))
            
            ## add the "correction forces" out in the buffer region
            ## f_eff = f(II) - (-D.(-EGF.f(II)) = f(II) + D.u_bc
            f += D.dot(np.reshape(u_bc,3*size_all))

            ## solve Dii.u = f_eff for u
            logging.debug('  entering solver for {} along {}'.format(j, direction[d]))
            t1 = time.time()  
            # [uf,conv] = sla.cg(D[0:3*size_in,0:3*size_in],f,tol=1e-08)
            [uf,conv] = sla.cg(Din,f,tol=args.tol)
            logging.debug('%d, solve time: %f'%(conv,time.time()-t1))

            ## since I put in initial forces of unit magnitude,,
            ## the column vector uf = column of LGF matrix
            if ((j == LGF_jmin) and (d == 0)):
                G = uf[0:3*size_123].copy()
            else:
                G = np.column_stack((G,uf[0:3*size_123]))

            logging.info('Atom {} direction {}'.format(j,direction[d]))

        # np.save(args.Gfile, G) ## save updated G after every atom (3 columns)
    with h5py.File(args.Gfile, 'w') as f:
        f.attrs['size_1'] = size_1
        f.attrs['size_12'] = size_12
        f.attrs['size_123'] = size_123
        f.attrs['size_in'] = size_in        
        f['GF'] = G
        
    logging.info('COMPLETED !! Total time taken: %.5f'%(time.time()-t0))
