import argparse
import numpy as np
import IO_vasp
import IO_xyz


if __name__ == '__main__':
    
    
    parser = argparse.ArgumentParser(description='Write the LGFCAR.')
    parser.add_argument('atomxyzfile',
                        help='xyz file that contains the atom positions')
    parser.add_argument('Gfile',
                        help='.npy file with the computed G')
    parser.add_argument('LGFCARfile',
                        help='LGFCAR file to write to')
    parser.add_argument('elementlist', type=int, nargs='+',
                        help='list of element indices (ints) corresponding to each basis atom')
    parser.add_argument('header',
                        help='header for LGFCAR (string)')
      
    
    ## read in the above arguments from command line
    args = parser.parse_args()   
    
    ## read in grid
    with open(args.atomxyzfile,'r') as f2:
        grid,[size_1,size_12,size_123,size_in] = IO_xyz.grid_from_xyz_reg(f2.read(),['Fe'])
    
    ## load the G matrix that we computed with calc_LGF.py
    G0 = np.load(args.Gfile)

    ## map atom indices used in calculating LGF to atom indices in VASP
    mapping = IO_vasp.map_indices(grid[:size_123],args.elementlist)

    ## write to LGFCAR file 
    with open(args.LGFCARfile, 'w') as f:
        f.write(IO_vasp.write_LGFCAR(G0,mapping,size_1,size_12,size_123,args.header))

