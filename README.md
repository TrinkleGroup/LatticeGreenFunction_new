# Computing the lattice Green function for dislocation topologies

Scripts for computing the force-constant matrix and lattice Green function (LGF) for a dislocation. The LGF is then used within the flexible boundary conditions approach coupled with DFT to relax the dislocation geometry.

The methodology is described in detail in the paper: A. M. Z. Tan and D. R. Trinkle, “Computation of the lattice Green function for a dislocation”, *Phys. Rev. E* **94**, 023308 (2016). All the scripts here were written by A. M. Z. Tan. 

There are essentially 2 parts to this process:
1. Generating the dislocation force-constant matrix
2. Evaluating the lattice Green function

Before we get started, prepare the following files:
- an input file that contains the crystal and dislocation setup info
```
<crystal class label>
<a1.x> <a1.y> <a1.z>
<a2.x> <a2.y> <a2.z>
<a3.x> <a3.y> <a3.z>
<# basis atoms per unit cell>
<basis atom coordinates in unit cell> (each basis atom is listed on a separate line)
...
<a0 (in Angstroms)>
<Cijs (in GPa)>
<m.x> <m.y> <m.z>
<n.x> <n.y> <n.z>
<t.x> <t.y> <t.z>
<squared magnitude of periodic vector t>
```
- an xyz file that contains the atom positions, ordered by regions
```
<total # atoms>
<# atoms in region 1> <# atoms in regions 1+2> <# atoms in regions 1+2+3> <# atoms in regions 1+2+3+buffer> any other comments
<basis atom label> <m coord> <n coord> <t coord>
...
```


## 1. Generating the dislocation force-constant matrix

There are a few ways to do this -- either by approximating the dislocation FCs based on bulk FCs, or by evaluating the dislocation FCs directly in the dislocation geometry using an empirical potential (e.g. EAM, GAP).

[more to come in this section...]

## 2. Evaluating the lattice Green function

Now that we have the dislocation force-constant matrix, we can go ahead and compute the LGF! :tada:

The main script for this is `calc_LGF.py`.
```
usage: calc_LGF.py [-h] [-LGF_jmin LGF_JMIN] [-LGF_jmax LGF_JMAX]
                   inputfile atomxyzfile Dfile Gfile logfile

Computes the dislocation lattice Green function.

positional arguments:
  inputfile           input file that contains the crystal and dislocation setup info
  atomxyzfile         xyz file that contains the atom positions
  Dfile               .mtx file to read the FC matrix D from
  Gfile               .npy file to save the computed G to
  logfile             logfile to save to

optional arguments:
  -h, --help          show this help message and exit
  -LGF_jmin LGF_JMIN  (int) first atom index to compute LGF for. Default is the first atom in region 2.
  -LGF_jmax LGF_JMAX  (int) last atom index to compute LGF for. Default is the last atom in region 2.
```

Note that the current version of this code calls functions from `elastic.py` to evaluate the far-field displacements according to the bulk elastic Green function. If you require other boundary conditions, e.g. elastic Green function for a bicrystal, you will have to code those up yourself and call them within `calc_LGF.py` at the part where we set the displacement of atoms in the far-field boundary.

Finally, you will need to write out the LGF into an LGFCAR file which will be read in and applied in VASP. The LGFCAR file must have the format:
```
<header commenet>
<min DFT index reg 2> <max DFT index reg 2> <min DFT index reg 123> <max DFT index reg 123> <total # entries>
<DFT atom index j> <DFT atom index i> <Gxx> <Gxy> <Gxz> <Gyx> <Gyy> <Gyz> <Gzx> <Gzy> <Gzz> (separate line for G between every pair of atoms)
...
```
Note that the DFT atom index is not the same as the atom index used in calculating the LGF! For one, the DFT atom index starts from 1 (since VASP is written in fortran), while we have been using 0-based indexing (python). Furthermore, the atoms in our calculation have been sorted by regions, while the atoms in the VASP calculation are ordered by element. Therefore, if you have a system with more than 1 element, you will need to reorder the LGF entries accordingly before writing to the LGFCAR file.

I have written a script `write_LGFCAR.py` which *should* help you do all this.
```
usage: write_LGFCAR.py [-h]
                       atomxyzfile Gfile LGFCARfile elementlist [elementlist ...] header

Write the LGFCAR.

positional arguments:
  atomxyzfile  xyz file that contains the atom positions
  Gfile        .npy file with the computed G
  LGFCARfile   LGFCAR file to write to
  elementlist  list of element indices (ints) corresponding to each basis atom
  header       header for LGFCAR (string)

optional arguments:
  -h, --help   show this help message and exit
```





