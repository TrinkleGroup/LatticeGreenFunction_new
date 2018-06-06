import argparse
import numpy as np
import IO_vasp
import IO_xyz


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Write the LGFCAR.')
    parser.add_argument('atomxyzfile',
                        help='xyz file that contains the atom positions')
    parser.add_argument('-atomlabel', action='append', required=True,
                        help='name label for each basis atom type as used in xyz file; '
                        'may be passed multiple times as required. '
                        'Place the flag -atomlabel before each entry. '
                        'Despite the flag, this is a REQUIRED (not optional) argument!')
    parser.add_argument('-elementindex', type=int, action='append', required=True,
                        help='element index (int) corresponding to each basis atom; '
                        'may be passed multiple times as required. There should be as many '
                        'entries for this as there are atom labels, and passed in the same order. '
                        'Place the flag -elementindex before each entry. '
                        'Despite the flag, this is a REQUIRED (not optional) argument! '
                        'The elements must be numbered in the same order as in the POSCAR/POTCAR '
                        'as this will be used to map atoms onto the DFT ordering.')
    parser.add_argument('Gfile',
                        help='.npy file with the computed G')
    parser.add_argument('LGFCARfile',
                        help='LGFCAR file to write to')
    parser.add_argument('header',
                        help='header for LGFCAR (string)')
          
    ## read in the above arguments from command line
    args = parser.parse_args()
    if len(args.atomlabel) != len(args.elementindex):
        raise ValueError('number of atom labels and element indices do not match!')
    
    ## read in grid
    with open(args.atomxyzfile,'r') as f2:
        grid,[size_1,size_12,size_123,size_in] = IO_xyz.grid_from_xyz_reg(f2.read(),args.atomlabel)
    
    ## load the G matrix that we computed with calc_LGF.py
    G0 = np.load(args.Gfile)

    ## map atom indices used in calculating LGF to atom indices in VASP
    mapping = IO_vasp.map_indices(grid[:size_123],args.elementindex)

    ## write to LGFCAR file 
    with open(args.LGFCARfile, 'w') as f:
        f.write(IO_vasp.write_LGFCAR(G0,mapping,size_1,size_12,size_123,args.header))

