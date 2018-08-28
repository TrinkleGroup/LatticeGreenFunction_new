import numpy as np
import h5py
from lammps import lammps
from collections import namedtuple
import argparse, logging
import setup
import IO_xyz
import IO_lammps

PAIR_STYLE = "pair_style	eam/fs"
PAIR_COEFF = "pair_coeff	* * ./w_eam4.fs W W W"

def init_lammps(lmp, datafilename):
    """
    Initialize our LAMMPS object in a sane way
    """
    lmp.command('units		metal')
    lmp.command('atom_style	atomic')
    lmp.command('atom_modify map array sort 0 0')  # forces LAMMPS to output in sorted order
    lmp.command('boundary	f f p')
    lmp.command('thermo 1')
    lmp.command('read_data	{}'.format(datafilename))
    lmp.command(PAIR_STYLE)
    lmp.command(PAIR_COEFF)


def lammps_minimize_getforces(datafilename,maxcgiter,ftol, antiplane=False):

    """
    Call lammps to relax region 1 and compute the forces afterwards 

    Parameters
    ----------
    datafilename : filename for lammps to read the atomic position data from
    maxcgiter    : maximum iterations stopping criteria for cg
    ftol         : force tolerance stopping criteria for cg
    
    Returns
    -------
    grid : (size_123,5) ndarray of the index,type(region),m,n,t coords of each atom
           * the coords have not been scaled out by a0 here!
    forces : (size_123,3) ndarray of the forces on each atom
       
    """ 
    
    lmp = lammps()
    init_lammps(lmp, datafilename)
    
    ## relax reg 1, keeping reg 2+3 fixed  
    lmp.command("group 	reg1 type 1")
    lmp.command("group 	reg12 type 1 2")
    lmp.command("group 	reg23 type 2 3")
    lmp.command("group 	reg3 type 3")
    if antiplane:
        lmp.command("fix		1 reg1 setforce 0.0 0.0 NULL")
    lmp.command("fix		2 reg23 setforce 0.0 0.0 0.0")
    # lmp.command("min_style	cg")
    lmp.command("min_style	hftn")
    lmp.command("minimize	0.0 %0.16f %d 10000"%(ftol,maxcgiter))
    
    ## compute reg 1+2 forces 
    if antiplane:
        lmp.command("unfix	1")
        lmp.command("fix		3 reg12 setforce 0.0 0.0 NULL")
    lmp.command("unfix       2")
    lmp.command("fix		4 reg3 setforce 0.0 0.0 0.0")
    lmp.command("compute 	output all property/atom id type x y z fx fy fz")
    lmp.command("run 		0")
    
    output = lmp.extract_compute("output",1,2)
    
    ## extract the atom positions after region 1 LAMMPS relaxation        
    grid_temp, forces = np.zeros((size_123,3)), np.zeros((size_123,3))
    for i in range (size_123):
        grid_temp[i], forces[i] = output[i][2:5], output[i][5:8]
        
    return grid_temp,forces


def lammps_getforces(datafilename, antiplane=False):
    
    """
    Call lammps to compute the forces in regions 1 & 2 

    Parameters
    ----------
    datafilename : filename for lammps to read the atomic position data from
    
    Returns
    -------
    forces : (size_123,3) nd array of the forces on each atom.
    
    """
    
    lmp = lammps()
    init_lammps(lmp, datafilename)
    
    ## compute reg 1+2 forces 
    lmp.command("group 	reg12 type 1 2")
    lmp.command("group       reg3 type 3")
    lmp.command("fix		1 reg3 setforce 0.0 0.0 0.0")
    if antiplane:
        lmp.command("fix		2 reg12 setforce 0.0 0.0 NULL")
    lmp.command("compute 	output all property/atom id type fx fy fz")
    lmp.command("run 		0")
    
    output = lmp.extract_compute("output",1,2)
    
    ## extract the forces   
    forces = np.zeros((size_123,3))
    for i in range (size_123):
        forces[i] = output[i][2:5]
        
    return forces
    

def relaxation_cycle(datafilename,G,size_1,size_12,size_123,method,maxcgiter, maxdisp=1e2,
                     antiplane=False):
    
    """
    carries out 1 relaxation cycle = 1 core relax + 1 LGF update
    
    Parameters
    ----------
    datafilename : filename for lammps to read the atomic position data from
    G            : LGF matrix to use in the LGF update step
    size_1       : number of atoms in region 1
    size_12      : number of atoms in regions 1+2
    size_123     : number of atoms in regions 1+2+3
    method       : (string) method to use for the LGF update step
    maxcgiter    : maximum iterations stopping criteria for cg
    maxdisp      : upper limit on displacement to allow from LGF

    Returns
    -------
    grid       : (size_123,3) ndarray of the m,n,t coords of each atom
                 at the end of the loop
    
    """

    ## call LAMMPS to relax region 1 
    ## relax for a fixed number of iterations (maxcgiter) each time
    ## the grid output by this function has atom mnt coords in Angstroms
    grid,forces = lammps_minimize_getforces(datafilename,maxcgiter,ftol=1E-6,antiplane=antiplane)
    forces_2 = np.reshape(forces[size_1:size_12],(3*(size_12-size_1),1))
    
    ## LGF update
    if method == 'dislLGF123' or method == 'perfbulkLGF123':
        ## displace region 1+2+3 according to LGF and update atom positions in grid
        disp = np.reshape(np.dot(G,forces_2),(size_123,3))
        dispmax = np.sqrt(max(sum(u**2) for u in disp))
        logging.debug('### Displacement max: {}'.format(dispmax))
        scale = 1 if dispmax < maxdisp else maxdisp/dispmax
        logging.debug('### scale: {}'.format(scale))
        grid += scale*disp
            
    elif method == 'dislLGF23' or method == 'perfbulkLGF23':
        ## displace region 2+3 according to LGF and update atom positions in grid
        grid[size_1:] -= np.reshape(-np.dot(G[3*size_1:,:],forces_2),
                                    (size_123-size_1,3))
        
    else:
        raise ValueError('invalid method!')

    return grid
    

if __name__ == '__main__':
    

    parser = argparse.ArgumentParser(description='Code for coupling core relaxation using LAMMPS and LGF updates.')
    parser.add_argument('inputfile',
                        help='input file that contains the crystal and dislocation setup info')
    parser.add_argument('atomxyzfile',
                        help='xyz file that contains the atom positions')
    parser.add_argument('-atomlabel', action='append', required=True,
                        help='name label for each basis atom type as used in xyz file; '
                        'may be passed multiple times as required. '
                        'Place the flag -atomlabel before each entry. '
                        'Despite the flag, this is a REQUIRED (not optional) argument!')
    parser.add_argument('Gfile',
                        help='.npy file with the LGF')
    parser.add_argument('-method',
                        help='method of LGF update to use'
                             'valid options: dislLGF123 dislLGF23 perfbulkLGF123 perfbulkLGF23',
                             default='dislLGF123')   
    parser.add_argument('-maxcgiter', type=int,
                        help='maximum number of steps of CG to run for every core relaxation',
                        default=5)
    parser.add_argument('-maxiter', type=int,
                        help='maximum number of iterations of core relax + LGF update to run',
                        default=51)
    parser.add_argument('-forcetol', type=float,
                        help='force tolerance convergence criteria',
                        default=1E-6)
    parser.add_argument('-logfile',
                        help='logfile to save to')
    parser.add_argument('-mappingfile',
                        help='.npy file with the mapping from edge to perfect bulk geometry')
    parser.add_argument('-antiplane', type=bool, default=False,
                        help='only perform displacements in the threading direction')
                          
    ## read in the above arguments from command line
    args = parser.parse_args()
    method = args.method

    ## set up logging
    if args.logfile:
        logging.basicConfig(filename=args.logfile,filemode='w',format='%(levelname)s: %(message)s', level=logging.DEBUG)    
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        logging.getLogger('').addHandler(console)
    else:
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)


    """""
    SETUP 
    
    """""
             
    ## read in setup details
    """
    crystalclass: crystal class (4=cubic; 9=hexagonal; 10=isotropic)  
    a0          : lattice constant (angstroms)
    Cijs        : list of Cijs (GPa)
    M           : 3x3 matrix for rotating from mnt basis to cubic cartesian basis
                  (columns are normalized m,n,t vectors)
    t_mag       : magnitude of the periodic vector along the dislocation threading direction
                  
    """
    with open(args.inputfile,'r') as f:
        crystalclass,a0,Cijs,M,t_mag = setup.readinputs(f)
    
    ## read in grid of atoms
    """
    grid : list of namedtuples containing atom info
           [index, region, m-coord, n-coord, t-coord, basis]
    size_1,size_12,size_123,size_in,size_all: cumulative # atoms in each of the regions
    
    """
    with open(args.atomxyzfile,'r') as f:
        grid, (size_1,size_12,size_123,size_in) = \
               IO_xyz.grid_from_xyz_reg(f.read(),args.atomlabel,1.)
    size_2 = size_12 - size_1

    ## write lammps .data file of dislocation geometry
    datafilename = 'dislgeom.data'
    with open(datafilename, 'w') as f:
        ## I just modified the function to label the atoms by region instead of basis atom type
        ## but I haven't checked it, so be careful!
        f.write(IO_lammps.lammps_writedatafile_reg(grid[:size_123],1.,t_mag*a0))
                
    ## Load G matrix computed by calc_LGF.py
    # G = np.load(args.Gfile)
    with h5py.File(args.Gfile, 'r') as f:
        if f.attrs['size_1'] != size_1:
            raise ValueError('LGF not consistent with setup? Has size_1 = {} != {}'.format(f.attrs['size_1'], size_1))
        if f.attrs['size_12'] != size_12:
            raise ValueError('LGF not consistent with setup? Has size_12 = {} != {}'.format(f.attrs['size_12'], size_12))
        if f.attrs['size_123'] != size_123:
            raise ValueError('LGF not consistent with setup? Has size_123 = {} != {}'.format(f.attrs['size_123'], size_123))
        G = f['GF'].value

    ## rotate G from xyz to mnt basis
    G_mnt = np.zeros((size_123*3,size_2*3))
    if method == 'dislLGF123' or method == 'dislLGF23':
        if G.shape[0] != size_123*3 or G.shape[1] != size_2*3:
            raise ValueError('G has the wrong shape!')
        for i in range(size_123):
            for j in range(size_2):
                G_mnt[i*3:(i+1)*3,j*3:(j+1)*3] = np.dot(M.T,np.dot(G[i*3:(i+1)*3,j*3:(j+1)*3],M))
                 
    elif method == 'perfbulkLGF123' or method == 'perfbulkLGF23':
        ## If you want to use the perfect bulk LGF, you need to provide
        ## the index mapping between the dislocation and perfect bulk!
        mapping = np.load(args.mappingfile)  
        for i in range(size_123):
            for j in range(size_2):
                i_b,j_b = mapping[i][j+size_1]
                j_b = j_b-size_1
                G_mnt[i*3:(i+1)*3,j*3:(j+1)*3] = np.dot(M.T,np.dot(G[i_b*3:(i_b+1)*3,j_b*3:(j_b+1)*3],M))   
                
    else:
        raise ValueError('invalid method!')


    """""""""
    RELAXATION 
    
    """""""""
  
    atominfo = namedtuple('atom',['ind','reg','m','n','t','basis'])
    
    ## compute initial forces in regions 1,2,3 using LAMMPS
    forces = lammps_getforces(datafilename, args.antiplane)
    forces_12 = np.reshape(forces[:size_12],(3*size_12,1))

    ## carry out relaxation until force tolerance level or max. # iterations is reached
    ## you should double check the force criteria, whether comparing force 2-norm or max. force...
    with open('initial-dislocation.data', 'w') as f:
        f.write(IO_lammps.lammps_writedatafile_reg(grid,1.,t_mag*a0))
    force_evolution = []
    for i in range(args.maxiter):
        force_2norm = np.linalg.norm(forces_12)
        force_max = abs(forces_12.flat[abs(forces_12).argmax()])
        logging.info('Iteration {}'.format(i+1))
        logging.info('Forces region 1:\n{}'.format(forces_12.reshape((size_12, 3))[:size_1]))
        logging.info('Forces region 2:\n{}'.format(forces_12.reshape((size_12, 3))[size_1:]))
        logging.info('Force norm: {}'.format(force_2norm))
        logging.info('Force max: {}'.format(force_max))
        force_evolution.append(force_2norm)
        # if force_2norm < args.forcetol*size_12:
        if force_max < args.forcetol:
            break
        elif force_2norm > 1E2: ## if forces blow up, something has gone very wrong!
            break
        else:
            ## perform 1 core relaxation in LAMMPS followed by 1 LGF update step
            grid_mat = relaxation_cycle(datafilename,G_mnt,size_1,size_12,size_123,method,args.maxcgiter, 0.05, antiplane=args.antiplane)
            ## convert grid from ndarray to namedtuple
            ## (basis atom type is not important here so I set it as a dummy)
            grid_new = [atominfo(atom.ind,atom.reg,mnt[0],mnt[1],mnt[2],0)
                        for atom, mnt in zip(grid, grid_mat)]
            ## write out the new atom positions into LAMMPS atom data input file
            ## since the atom coords in the new grid are in Angstroms (not scaled out by a0)
            ## pass a dummy a0=1.0 to the lammps_writedatafile_reg function
            with open(datafilename, 'w') as f:
                f.write(IO_lammps.lammps_writedatafile_reg(grid_new,1.,t_mag*a0))
            ## call LAMMPS again to compute forces in reg 1+2 after LGF update
            forces_new = lammps_getforces(datafilename, antiplane=args.antiplane)
            forces_12 = np.reshape(forces_new[:size_12],(3*size_12,1))

    ## write out the force 2-norms / max. force at every cycle
    np.save('forces.npy',force_evolution)
    print(force_evolution)   
    
