import scipy.io, argparse, logging
import numpy as np
import h5py
from lammps import lammps
import setup
import IO_xyz
import IO_lammps

PAIR_STYLE = "pair_style	eam/fs"
PAIR_COEFF = "pair_coeff	* * ./w_eam4.fs W"
# PAIR_STYLE = "pair_style     quip"
# PAIR_COEFF = "pair_coeff     * * gp33b.xml 'Potential xml_label=GAP_2016_10_3_60_19_29_10_891' 26"

def init_lammps(lmp, datafilename):
    """
    Initialize our LAMMPS object in a sane way
    """
    lmp.command('units		metal')
    lmp.command('atom_style	atomic')
    lmp.command('atom_modify map array sort 0 0')  # forces LAMMPS to output in sorted order
    lmp.command('boundary	f f p')
    lmp.command('read_data	{}'.format(datafilename))
    lmp.command(PAIR_STYLE)
    lmp.command(PAIR_COEFF)

def calcforces_lammps(datafilename,i,delta,size_all):
    
    """
    Call lammps to compute the forces in response to a displaced atom i

    Parameters
    ----------
    datafilename : data file from which lammps reads the atomic position data from (dislocation geometry)
    i : atom index of the atom to be displaced (0-based index)
    delta : displacement amount
    size_all : number of atoms in the system
    
    Returns
    -------
    forces : (size_all,3) np array of the forces on each atom
    
    """
    
    # now using centered difference on the cheap (internal to LAMMPS)
    lmp = lammps()
    init_lammps(lmp, datafilename)
    lmp.command("group          testatom id %d"%(i+1))
    lmp.command("compute        output all property/atom id type fx fy fz")
    f_d = np.zeros((3, 3, size_all))
    for d, disp in enumerate(delta*np.eye(3)):
        lmp.command("displace_atoms testatom move %0.12f %0.12f %0.12f"%(disp[0],disp[1],disp[2]))
        lmp.command("run            0")
    
        ## extract the forces   
        output = lmp.extract_compute("output",1,2)
        fpos = np.array([output[j][2:5] for j in range(size_all)]).T

        lmp.command("displace_atoms testatom move %0.12f %0.12f %0.12f"%(-2*disp[0],-2*disp[1],-2*disp[2]))
        # lmp.command("compute        output all property/atom id type fx fy fz")
        lmp.command("run            0")
        lmp.command("displace_atoms testatom move %0.12f %0.12f %0.12f"%(disp[0],disp[1],disp[2]))
    
        ## extract the forces   
        output = lmp.extract_compute("output",1,2)
        fneg = np.array([output[j][2:5] for j in range(size_all)]).T
        f_d[d] = 0.5*(fpos-fneg)/delta

    return f_d


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
    parser.add_argument('cutoff',type=float,
                        help='(float) cutoff distance for forces and force-constants (Angstroms)')
    parser.add_argument('-logfile',
                        help='logfile to save to')
    parser.add_argument('-disp',type=float,
                        help='(float) magnitude of displacements to apply. '
                        'Default is 1E-05 Angstroms.')
    parser.add_argument('-istart',type=int,
                        help='(int) first atom index to displace. '
                        'Default is the first atom in region 1.')
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
    size_1,size_12,size_123,size_in,size_all: cumulative # atoms in each of the regions
    
    """
    with open(args.atomxyzfile,'r') as f:
        grid, (size_1,size_12,size_123,size_in) = IO_xyz.grid_from_xyz_reg(f.read(),args.atomlabel,1.)
    size_all = len(grid)
    logging.info('System setup: size_1 = %d, size_12 = %d, size_123 = %d, size_in = %d, size_all = %d'
                  %(size_1,size_12,size_123,size_in,size_all))

    ## write lammps .data file of dislocation geometry
    datafilename = 'disl.data'
    with open(datafilename, 'w') as f:
        f.write(IO_lammps.lammps_writedatafile(grid,1.,t_mag*a0))

    if args.istart: 
        istart = args.istart
    else: 
        istart = 0  ## if istart not specified, default = 0
    if args.iend: 
        iend = args.iend
    else: 
        iend = size_in-1  ## if iend not specified, default = size_in-1 (all atoms up to buffer)        
 
    if args.disp:
        disp = args.disp  ## the magnitude of displacement to apply on each atom
    else:
        disp = 1E-05

    D = scipy.sparse.lil_matrix((size_in*3,size_all*3)) 
    for i in range(istart,iend+1):
        logging.info('displacing atom %d'%i)
      
        forces = calcforces_lammps(datafilename,i,disp,size_all)  # now does +-m, +-n, +-t
        for j in range(size_all):
            # changed to only use distance in-plane, and to not scale cutoff by a0:
            if np.linalg.norm(forces[:,:, j]) >= 1e-6:
                D_mnt = -forces[:, :, j]
        
                D[i*3:(i+1)*3,j*3:(j+1)*3] = np.dot(M,np.dot(D_mnt,M.T)) ## rotate from mnt to xyz cart coords
#                D[i*3:(i+1)*3,j*3:(j+1)*3] = np.dot(M,np.dot(D_mnt.T,M.T)) ## rotate from mnt to xyz cart coords
            
    # Dump as hdf file instead (faster, and less space):
    D_csr = D.tocsr()
    with h5py.File(args.Dfile, 'w') as f:
        f['data'] = D_csr.data
        f['indices'] = D_csr.indices
        f['indptr'] = D_csr.indptr
