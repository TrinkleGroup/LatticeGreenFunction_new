import numpy as np
from collections import namedtuple


def lammps_writedatafile(grid,a0,t_mag):

    """
    Create a string which will be written out to the LAMMPS atom data input file.
    The format is as follows (<> indicate values to be filled in):
    
    Position data for bcc Fe edge dislocation

    <#atoms>    atoms
    <#atom types>   atom types

    <xlo>   <xhi>    xlo xhi
    <ylo>   <yhi>    ylo yhi
    <zlo>   <zhi>    zlo zhi

    Atoms

    <ind>   <atom type>     <m-coord>   <n-coord>   <t-coord>
    ...
    
    Parameters
    ----------
    grid : list of [atom index,region,m-coord,n-coord,t-coord,basis]
           for each atom in the geometry
    a0 : lattice constant (angstroms)
    t_mag : magnitude of the periodic vector along the dislocation threading direction
            i.e. slab thickness

    Returns
    -------
    s    : string which will be written out to the LAMMPS atom data input file
    
    """

    index,reg,mcoords,ncoords,tcoords,basis = zip(*grid)
    
    ## set the size of the simulation box
    xlo = np.min(mcoords)*a0 - 20  ## add at least 20 Angstroms of vacuum around the whole slab
    xhi = np.max(mcoords)*a0 + 20
    ylo = np.min(ncoords)*a0 - 20
    yhi = np.max(ncoords)*a0 + 20
    zlo = 0.
    zhi = t_mag*a0

    s = """Position data for dislocation

{atoms:>6}   atoms
{types:>6}   atom types
    
{xlo:>24.16f} {xhi:>24.16f}    xlo xhi
{ylo:>24.16f} {yhi:>24.16f}    ylo yhi
{zlo:>24.16f} {zhi:>24.16f}    zlo zhi
 
Atoms
   
""".format(atoms=len(grid),types=int(np.max(basis)+1),xlo=xlo,xhi=xhi,ylo=ylo,yhi=yhi,zlo=zlo,zhi=zhi)
    
    for atom in grid:
        ## write out atom index, mnt coords
        s += "{index:<8d} {atomtype} {mcoord:24.16f} {ncoord:24.16f} {tcoord:24.16f}\n".format(index=int(atom.ind+1),
                atomtype=int(atom.basis+1),mcoord=atom.m*a0,ncoord=atom.n*a0,tcoord=atom.t*a0)

    return s
