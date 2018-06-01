import numpy as np
import numpy.linalg as la
from collections import namedtuple
        
        
def readinputs(f):
    
    """
    Reads in information from input file:
    parameters related to the material and crystal structure,
    followed by information related to setting up the dislocation slab geometry.

    Parameters
    ----------
    f : filename to read data from e.g. input_bccscrew  

    Returns
    -------    
    crystalclass: crystal class (0=isotropic; 1=cubic; 2=hexagonal)    
    A           : 3x3 matrix for rotating from primitive cell basis to cartesian basis
                  (columns are unitcell vectors a1,a2,a3)
    unitcell_pos: positions of the basis atoms within the primitive unit cell
    a0          : lattice constant in angstroms
    Cijs        : list of Cijs
    
    M           : 3x3 matrix for rotating from mnt basis to cartesian basis
                  (columns are normalized m,n,t vectors)
    t_mag       : magnitude of the periodic vector along the dislocation threading direction
   
    """  
    
    ## read in lines from input file and ignore blank lines and comment lines
    lines = [line.rstrip() for line in f if line.rstrip() if line[0] != '#']
             
    # first line is the crystal class
    crystalclass = int(lines[0].split()[0])

    # a1,a2,a3
    A = np.array([[float(lines[1].split()[0]),float(lines[1].split()[1]),float(lines[1].split()[2])],
                  [float(lines[2].split()[0]),float(lines[2].split()[1]),float(lines[2].split()[2])],
                  [float(lines[3].split()[0]),float(lines[3].split()[1]),float(lines[3].split()[2])]]).T
                  
    # number of basis atoms
    num_basis = int(lines[4].split()[0]) 

    # basis atom positions in unit cell
    unitcell_pos = []
    for i in range(num_basis): 
        unitcell_pos.append([float(lines[5+i].split()[0]),float(lines[5+i].split()[1]),float(lines[5+i].split()[2])])                 
            
    # lattice constant (Angstroms)
    a0 = float(lines[-8].split()[0])
    
    # elastic constants (GPa)
    Cijs = [float(lines[-7].split()[0]),float(lines[-7].split()[1]),float(lines[-7].split()[2])]
                  
    # m,n,t
    m = np.array([float(lines[-6].split()[0]),float(lines[-6].split()[1]),float(lines[-6].split()[2])])
    n = np.array([float(lines[-5].split()[0]),float(lines[-5].split()[1]),float(lines[-5].split()[2])])
    t = np.array([float(lines[-4].split()[0]),float(lines[-4].split()[1]),float(lines[-4].split()[2])])
    M = np.array([m/la.norm(m),n/la.norm(n),t/la.norm(t)]).T
    
    # t_mag
    t_mag = np.sqrt(float(lines[-3].split()[0]))
    
    # m,n coordinates of centre of dislocation geometry 
#    m_c = float(lines[-2].split()[0])
#    n_c = float(lines[-2].split()[1])
    
    # region 1,2,3 size info from DFT setup
#    dftsize_1 = int(lines[-1].split()[0])
#    dftsize_12 = int(lines[-1].split()[1])
#    dftsize_123 = int(lines[-1].split()[2])
    
    return (crystalclass,A,unitcell_pos,a0,Cijs,M,t_mag)

                                    