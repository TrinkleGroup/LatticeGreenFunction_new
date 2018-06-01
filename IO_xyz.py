import numpy as np
from collections import namedtuple


def grid_from_xyz(s,atomtypes,a0):

    """
    Read from a string containing the data from an xyz file. 
    
    Parameters
    ---------- 
    s : string containing the data from an xyz file.
    atomtypes: list of name labels for each basis atom type
    a0 : lattice constant in Angstroms

    Returns
    -------
    grid : list of [atom index,region,m-coord,n-coord,t-coord,basis]
           for each atom in the geometry
           ** coordinates are scaled out by a factor of a0 !!
    
    """

    atominfo = namedtuple('atom',['ind','reg','m','n','t','basis'])

    grid = []    
    for line in s.splitlines()[2:]:
        if line != '':
            entries = line.split()
            i = int(len(grid))
            reg = 0
            m,n,t = float(entries[1])/a0,float(entries[2])/a0,float(entries[3])/a0
            basis = atomtypes.index(entries[0])
            grid.append(atominfo(i,reg,m,n,t,basis))
            
    return grid


def grid_from_xyz_reg(s,atomtypes,a0):

    """
    Read from a string containing the data from an xyz file and label atoms by regions.
    The atoms in the xyz file must already be listed in order by regions and 
    the second line of the file contains the size_1,size_12,size_123,size_in info.
    
    Parameters
    ---------- 
    s : string containing the data from the anisotropic dislocation geometry setup file
    atomtypes: list of name labels for each basis atom type
    a0 : lattice constant in Angstroms

    Returns
    -------
    grid : list of [atom index,region,m-coord,n-coord,t-coord,basis]
           for each atom in the geometry
           ** coordinates are scaled out by a factor of a0 !!
    sizes : numbers of atoms in reg1, 1+2, 1+2+3, 1+2+3+buffer
  
    """

    atominfo = namedtuple('atom',['ind','reg','m','n','t','basis'])

    size_1,size_12,size_123,size_in = [int(i) for i in (s.splitlines()[1]).split()[:4]]

    grid = []    
    for line in s.splitlines()[2:]:
        if line != '':
            entries = line.split()
            i = int(len(grid))
            if i < size_1: reg = 1
            elif i < size_12: reg = 2
            elif i < size_123: reg = 3
            elif i < size_in: reg = 4
            else: reg = 5  
            m,n,t = float(entries[1])/a0,float(entries[2])/a0,float(entries[3])/a0
            basis = atomtypes.index(entries[0])
            grid.append(atominfo(i,reg,m,n,t,basis))
            
    return grid,[size_1,size_12,size_123,size_in]
   
   
def grid_to_xyz(grid,atomtypes,a0,header):

    """
    Create a string that will be written to a xyz file, e.g. anisotropic code geometry file. 
    
    Parameters
    ---------- 
    grid : list of [atom index,region,m-coord,n-coord,t-coord,basis]
           for each atom in the geometry
           ** coordinates are scaled out by a factor of a0 !!
    atomtypes: list of name labels for each basis atom type
    a0 : lattice constant in Angstroms
    header : comment string for 2nd line of xyz file

    Returns
    -------
    s    : string containing the data for the xyz file
    
    """  
    
    s = "{numatoms}\n".format(numatoms=len(grid))
    s += header  + "\n"     
    for atom in grid:
        # print atom index, mnt coords
        s += "{atomtype} {mcoord:20.15f} {ncoord:20.15f} {tcoord:20.15f}\n".format(
                atomtype=atomtypes[atom[5]],mcoord=a0*atom[2],ncoord=a0*atom[3],tcoord=a0*atom[4])
    
    return s
    