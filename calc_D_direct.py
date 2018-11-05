import argparse
import logging
import h5py
import scipy.io
import numpy as np
from lammps import lammps
import setup
import IO_xyz
import IO_lammps


class lammps_settings:
    
    """
    Initialize a lammps_settings object with the following attributes:
    datafilename : filename for lammps to read the atomic position data from
    pair_style   : lammps pair_style command
    pair_coeff   : lammps pair_coeff command
         
    """
    
    def __init__(self, datafilename, lammpspairfile):
        
        self.datafilename = datafilename
        with open(lammpspairfile, 'r') as f:
            lines = f.readlines()
            self.pair_style = lines[0].rstrip('\n')  ## e.g. "pair_style eam/fs"   
            self.pair_coeff = lines[1].rstrip('\n')  ## e.g. "pair_coeff * * ./w_eam4.fs W W W" 


def init_lammps(lmp, ls):
    
    """
    Initialize our LAMMPS object in a sane way.
    
    Parameters
    ----------
    lmp : LAMMPS object
    ls : lammps_settings object containing the attributes:
         datafilename, pair_style, pair_coeff
    
    """
    lmp.command('units        metal')
    lmp.command('atom_style   atomic')
    lmp.command('atom_modify  map array sort 0 0')  ## forces LAMMPS to output in sorted order!
    lmp.command('boundary     f f p')
    lmp.command('read_data	{}'.format(ls.datafilename))
    lmp.command('{}'.format(ls.pair_style))
    lmp.command('{}'.format(ls.pair_coeff))
    

def lammps_calcforces(ls, size_all):
    
    """
    Call LAMMPS to compute the forces in a given system

    Parameters
    ----------
    ls : lammps_settings object containing the attributes:
         datafilename, pair_style, pair_coeff
    size_all : number of atoms in the system
    
    Returns
    -------
    f0 : (3,size_all) np array of the forces on each atom in the given system
         without additional displacements
    
    """
    
    lmp = lammps()
    init_lammps(lmp, ls)
    ## compute the forces in the given system
    lmp.command("compute  output all property/atom id type fx fy fz")
    lmp.command("run  0")
    ## extract the forces   
    output = lmp.extract_compute("output",1,2)    
    f0 = np.array([output[j][2:5] for j in range(size_all)]).T   
    
    return f0


def lammps_calcforces_findiff(ls, i, delta, size_all, centraldiff, f0):
    
    """
    Call LAMMPS to compute the forces in response to a displaced atom i
    and apply finite differences to get force-constants

    Parameters
    ----------
    ls : lammps_settings object containing the attributes:
         datafilename, pair_style, pair_coeff
    i : atom index of the atom to be displaced (0-based index)
    delta : displacement amount (in Angstroms)
    size_all : number of atoms in the system
    centraldiff: (boolean) to use central diff or fwd diff
    f0: (3,size_all) np array of the forces on each atom in the given system
        without additional displacements (pass None if centraldiff=True)
    
    Returns
    -------
    fc: (3,3,size_all) np array of the forces on each atom
        in response to unit displacements of atom i
    
    """
    
    ## now applying +/(-)m, +/(-)n, +/(-)t displacements internally to LAMMPS
    lmp = lammps()
    init_lammps(lmp, ls)
    lmp.command("group    testatom id %d"%(i+1))
    lmp.command("compute  output all property/atom id type fx fy fz")
    
    fc = np.zeros((3,3,size_all))
    for d, disp in enumerate(delta*np.eye(3)):
        
        ## apply positive displacements
        lmp.command("displace_atoms testatom move %0.12f %0.12f %0.12f"%(disp[0],disp[1],disp[2]))
        lmp.command("run  0")    
        ## extract the forces   
        output = lmp.extract_compute("output",1,2)
        fpos = np.array([output[j][2:5] for j in range(size_all)]).T
        
        if centraldiff:
            ## apply negative displacements 
            lmp.command("displace_atoms testatom move %0.12f %0.12f %0.12f"%(-2*disp[0],-2*disp[1],-2*disp[2]))
            lmp.command("run  0")    
            ## extract the forces   
            output = lmp.extract_compute("output",1,2)
            fneg = np.array([output[j][2:5] for j in range(size_all)]).T            
            ## apply central differences
            fc[d] = 0.5*(fpos-fneg)/delta            
        else:
            ## apply forward differences
            fc[d] = (fpos-f0)/delta

    return fc


if __name__ == '__main__':

    
    parser = argparse.ArgumentParser(description='Directly evaluates dislocation force-constants using empirical potential.')
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
                        help='HDF5 file to save the FC matrix D to')
    parser.add_argument('lammpspairfile',
                        help='file listing the LAMMPS pair_style and pair_coeff')
    parser.add_argument('-logfile',
                        help='logfile to save to')
    parser.add_argument('-finitediff',
                        help='finite difference method to use (forward/central). '
                        'Default is central difference.')
    parser.add_argument('-disp',type=float,
                        help='(float) magnitude of displacements to apply. '
                        'Default is 1E-05 Angstroms.',default=1E-05)
    parser.add_argument('-istart',type=int,
                        help='(int) first atom index to displace. '
                        'Default is the first atom in region 1.',default=0)
    parser.add_argument('-iend',type=int,
                        help='(int) last atom index to displace. '
                        'Default is the last atom in the buffer. ' 
                        'Note! Atom indices are based on 0-based indexing.')
     
    ## read in arguments from the command line
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
    with open(args.inputfile,'r') as f:
        crystalclass,a0,Cijs,M,t_mag = setup.readinputs(f)
    
    ## read in grid of atoms
    """
    grid : list of namedtuples containing atom info
           [index, region, m-coord, n-coord, t-coord, basis]
           ** we'll leave the atom coords in Angstroms, i.e. not scaled out by a0
    size_1,size_12,size_123,size_in,size_all: cumulative # atoms in each of the regions
    
    """
    with open(args.atomxyzfile,'r') as f:
        grid, (size_1,size_12,size_123,size_in) = IO_xyz.grid_from_xyz_reg(f.read(),args.atomlabel,1.0)
    size_all = len(grid)
    logging.info('System setup: size_1 = %d, size_12 = %d, size_123 = %d, size_in = %d, size_all = %d'
                  %(size_1,size_12,size_123,size_in,size_all))

    ## initialize lammps_settings object
    datafilename = 'disl.data'      
    ls = lammps_settings(datafilename,args.lammpspairfile,args.maxcgiter,args.ftol)

    ## write lammps .data file of dislocation geometry
    with open(ls.datafilename, 'w') as f:
        f.write(IO_lammps.lammps_writedatafile(grid,1.0,t_mag*a0))

    centraldiff=True
    if args.finitediff:
        if args.finitediff[0] == 'f':
            centraldiff = False
            logging.info('using forward differences')
        elif args.finitediff[0] == 'c':
            logging.info('using central differences')
    else:
        logging.info('using central differences by default')

    istart = args.istart
    if args.iend: 
        iend = args.iend
    else:
        ## if iend not specified, default = size_in-1 (all atoms up to buffer)
        iend = size_in-1      

    ## finally, we get to the calculations...
    if not centraldiff:
        ## if fwd diff, evaluate forces in undisplaced system
        f0 = lammps_calcforces(ls,size_all)
    else:
        f0 = None

    D = scipy.sparse.lil_matrix((size_in*3,size_all*3))        
    for i in range(istart,iend+1):
        
        logging.info('displacing atom %d'%i)      
        ## displace atom +/(-)m, +/(-)n, +/(-)t, evaluate forces,
        ## and evaluate force-constant using finite differences
        fc = lammps_calcforces_findiff(ls,i,args.disp,size_all,centraldiff,f0)
        
        for j in range(size_all):
            ## only store significant force-constants
            if np.linalg.norm(fc[:,:,j]) >= 1e-6:
                D_mnt = -fc[:,:,j]
                ## rotate from mnt to cubic cart coords
                D[i*3:(i+1)*3,j*3:(j+1)*3] = np.dot(M,np.dot(D_mnt,M.T))
            
    ## Dump as hdf file (faster, and less space)
    D_csr = D.tocsr()
    with h5py.File(args.Dfile, 'w') as f:
        f['data'] = D_csr.data
        f['indices'] = D_csr.indices
        f['indptr'] = D_csr.indptr
        
        