# vasp_patches

Updated patchfiles for reading and applying a lattice Green function based position update in VASP.

The original patch was written by J. A. Yasi and D. R. Trinkle in 2011. D. R. Trinkle updated it in 2016 to use ALLOCATABLE arrays rather than POINTER arrays.

There are two versions of the patchfile, one for patching VASP 4.6.34 and one for patching VASP 5.3.3. If you want to patch other versions of VASP, you will most likely have to do it manually, using the patchfile for the more similar version of VASP as a reference.

After applying the patch, the Makefile must be changed manually before re-compiling:
* lgf.o needs to be added to the SOURCE line
* -DSUPPORT_LGF must be added to the preprocessor line

To use, construct an LGFCAR file and place with other input files. Add ILGF=1 entry to INCAR. Choose a relaxation method for atoms with selective dynamics, where atoms in region I are free to move, and all others are fixed.