import numpy as np
from collections import namedtuple


def grid_from_POSCAR(s):

    """
    Read atom cooredinates from a VASP POSCAR/CONTCAR file.
    ** This function assumes the coordinates are in Direct coordinates
    
    Parameters
    ---------- 
    s    : string containing the data from a VASP POSCAR file

    Returns
    -------
    grid : list of [atom index,region,m-coord,n-coord,t-coord,basis]
           for each atom in the geometry
           ** coordinates are scaled out by a factor of a0 !!
    
    """

    atominfo = namedtuple('atom',['ind','reg','m','n','t','basis'])

    ## read in lines from input file and ignore blank lines and comment lines
    lines = [line.rstrip() for line in s.splitlines() if line.rstrip() if line[0] != '#']
             
    ## lattice constant
#    a0 = float(lines[1].split()[0])
    
    ## supercell vectors
    m = lines[2].split()
    m_mag = np.linalg.norm([float(m[0]),float(m[1]),float(m[2])])
    n = lines[3].split()
    n_mag = np.linalg.norm([float(n[0]),float(n[1]),float(n[2])])
    t = lines[4].split()
    t_mag = np.linalg.norm([float(t[0]),float(t[1]),float(t[2])])
    
    ## number of atoms 
    ## the line position in the POSCAR file may vary
    ## so I'm just finding the next line with a number instead of a string
    ## I can't guarantee that it's 100% foolproof though...
    for i,line in enumerate(lines[5:]):
        if line.split()[0].isdigit():
            num_atoms = [int(entry) for entry in line.split()]
            tot_atoms = sum(num_atoms)
            continue
        if line.split()[0] == 'Direct':
            break

    ## atom positions, assumed to be in Direct coordinates
    grid = []
    for line in lines[5+i+1:5+i+1+tot_atoms]:
        entries = line.split()
        basis = 0
        ## assign basis atom type based on atom type defined in POSCAR?
#        for cumsum in np.cumsum(num_atoms):
#            if len(grid) >= cumsum: basis += 1
        grid.append(atominfo(int(len(grid)),0,float(entries[0])*m_mag,float(entries[1])*n_mag,float(entries[2])*t_mag,basis))
            
    return grid
    
    
#def grid_from_OUTCAR(s):
#
#    """
#    Read from a string containing the data from a VASP OUTCAR file. 
#    ** This old version only reads the data from the first iteration it finds
#    
#    Parameters
#    ---------- 
#    s    : string containing the data from a VASP OUTCAR file
#
#    Returns
#    -------
#    grid : list of [atom index,region,x-coord,y-coord,z-coord,basis]
#           for each atom in the geometry
#    forces : list of [force_x,force_y,force_z]
#    
#    """
#
#    atominfo = namedtuple('atom',['ind','reg','m','n','t','basis'])
#    forceinfo = namedtuple('force',['m','n','t'])
#
#    for i,line in enumerate(s.splitlines()):
#        if line.rstrip():
#            if line.split()[0] == 'POSITION':
#                break
#
#    grid = []
#    forces = []
#    for line in s.splitlines()[i+2:]:
#        if line.split()[0][0] == '-':
#            break
#        else:
#            entries = line.split()
#            grid.append(atominfo(int(len(grid)),0,float(entries[0]),float(entries[1]),float(entries[2]),0))
#            forces.append(forceinfo(float(entries[3]),float(entries[4]),float(entries[5])))
#            
#    return grid,forces
    
    
def grid_from_OUTCAR(s):

    """
    Read from a string containing the data from a VASP POSCAR file. 
    
    Parameters
    ---------- 
    s  : string containing the data from a VASP POSCAR file

    Returns
    -------
    grid : list of [atom index,region,x-coord,y-coord,z-coord,basis]
           for each atom in the geometry
           for each iteration in the OUTCAR
    forces : list of [force_x,force_y,force_z]
             for each iteration in the OUTCAR
    
    """

    atominfo = namedtuple('atom',['ind','reg','m','n','t','basis'])
    forceinfo = namedtuple('force',['m','n','t'])

    grid = []
    forces = []
    for i,line in enumerate(s.splitlines()[:]):
        if line.rstrip():
            ## find the start of "POSITION" data for each iteration
            if line.split()[0] == 'POSITION':
                grid.append([])
                forces.append([])
                ## start reading data, which starts 2 lines after
                for line in s.splitlines()[i+2:]:
                    if line.split()[0][0] == '-':
                        ## dashed line indicates end of this set of data
                        break
                    else:
                        entries = line.split()
                        grid[-1].append(atominfo(int(len(grid[-1])),0,float(entries[0]),float(entries[1]),float(entries[2]),0))
                        forces[-1].append(forceinfo(float(entries[3]),float(entries[4]),float(entries[5])))
            
    return grid,forces


def write_LGFCAR(G,mapping,size_1,size_12,size_123,header):
    
    """
    Write a string containing the data for a VASP LGFCAR file. 
    
    Parameters
    ---------- 
    G : LGF matrix (numpy array of shape (size_123,size_2))
    mapping : list, in which mapping[LGF_index] = DFT_index
    size_1 : number of atoms in reg 1
    size_12 : number of atoms in reg 1+2
    size_123 : number of atoms in reg 1+2+3
    header : comment string

    Returns
    -------
    s : string containing the data for a VASP LGFCAR file
    
    """
    
    size_2 = size_12-size_1 

    ## create string which will be written to LGFCAR file
    s = header + '\n'
    ## this next line must be:
    ## <min DFT index reg 2> <max DFT index reg 2> <min DFT index reg 1> <max DFT index reg 3> <total # entries>
    s += '%d %d %d %d %d\n'%(int(min(mapping[size_1:size_12])),int(max(mapping[size_1:size_12])),
                                1,size_123,size_2*size_123)
    
    ## determine the order to output LGF entries
    indexlist = []
    for j in range(size_2):
        for i in range(size_123):
            ## [DFT atom index j, DFT atom index i, LGF row index i, LGF col index j]
            indexlist.append([mapping[j+size_1],mapping[i],i,j])
            
    ## sort indexlist based on DFT atom index j first, then DFT atom index i
    indexlist = sorted(indexlist)
    
    for DFTj,DFTi,row,col in indexlist:
        ## <DFT atom index j> <DFT atom index i> <Gxx> <Gxy> <Gxz> <Gyx> <Gyy> <Gyz> <Gzx> <Gzy> <Gzz>
        Gij = G[row*3:(row+1)*3,col*3:(col+1)*3]
        s += '%d %d %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f %.12f\n'%(int(DFTj),int(DFTi),
                Gij[0,0],Gij[0,1],Gij[0,2],Gij[1,0],Gij[1,1],Gij[1,2],Gij[2,0],Gij[2,1],Gij[2,2])
    
    return s


def map_indices(grid_123,elements):
    
    """
    Map atom indices used in calculating LGF to atom indices in VASP
    
    Parameters
    ---------- 
    grid_123 : list of [atom index,region,m-coord,n-coord,t-coord,basis]
               for each atom in regions 1-3 of geometry
    elements : list of element indices corresponding to each basis atom

    Returns
    -------
    mapping_LGFtoDFT : list, in which mapping[LGF_index] = DFT_index
    
    """
    
    temp = [[atom.ind,elements[atom.basis]] for atom in grid_123]
    
    ## resort temp list by element type       
    ## mapping[DFT_index-1] = lGF_index
    mapping_DFTtoLGF = np.array(sorted(temp,key=lambda temp:temp[1]))[:,0]
    
    mapping_LGFtoDFT = mapping_DFTtoLGF.copy()
    ## mapping[LGF_index] = DFT_index
    ## LGF indexing starts from 0; DFT indexing starts from 1
    for DFTind,LGFind in enumerate(mapping_DFTtoLGF):
        mapping_LGFtoDFT[LGFind] = int(DFTind+1)
   
    return mapping_LGFtoDFT

