import numpy as np
import numpy.linalg as la
from scipy.constants import elementary_charge  


def convert_to_GPa(x):
    
    """ convert eV/A^3 to GPa """
    
    return x*elementary_charge*(10**10)**3/(10**9)


def convert_from_GPa(x):
    
    """ convert GPa to eV/A^3 """
    
    return x/(elementary_charge*(10**10)**3/(10**9))


def construct_C(crystalclass,Cijs):
    
    """
    given the crystal class and elastic constants, construct the elastic stiffness tensor
    I followed the same numbering system as in elastic.H in the anisotropic elasticity codes
    ** I have currently only implemented this for cubic (4), hexagonal (9) and isotropic (10) !!
    
    Parameters
    ----------
    crystalclass :  0 = triclinic                        (a != b != c, alpha!=beta!=gamma)
                    1 = monoclinic, diad || x_2          (a != b != c, alpha==gamma==90!=beta)
                    2: monoclinic, diad || x_3           (a != b != c, alpha==beta==90!=gamma)
                    3: orthorhombic                      (a != b != c, alpha==beta==gamma==90)
                    4: cubic          _                  (a == b == c, alpha==beta==gamma==90)
                    5: tetragonal (4  4  4|m)       _    (a == b != c, alpha==beta==gamma==90)
                    6: tetragonal (4mm _422  4|mmm  42m) (a == b != c, alpha==beta==gamma==90)
                    7: trigonal,  (3   3)  _        (a == b != c, alpha==beta==90, gamma==120)
                    8: trigonal,  (32  3m  3m)      (a == b != c, alpha==beta==90, gamma==120)
                    9: hexagonal                    (a == b != c, alpha==beta==90, gamma==120)
                   10: isotropic
    Cijs : list of elastic constants in GPa

    Returns
    -------
    C : stiffness tensor C in Voigt notation, cartesian basis, GPa
    
    """
    
    if crystalclass == 10: ## isotropic
        C11,C12 = Cijs
        C = np.array([[C11,C12,C12,0,0,0],
                      [C12,C11,C12,0,0,0],
                      [C12,C12,C11,0,0,0],
                      [0,0,0,(C11-C12)/2.,0,0],
                      [0,0,0,0,(C11-C12)/2.,0],
                      [0,0,0,0,0,(C11-C12)/2.]])
    
    elif crystalclass == 4: ## cubic
        C11,C12,C44 = Cijs
        C = np.array([[C11,C12,C12,0,0,0],
                      [C12,C11,C12,0,0,0],
                      [C12,C12,C11,0,0,0],
                      [0,0,0,C44,0,0],
                      [0,0,0,0,C44,0],
                      [0,0,0,0,0,C44]])
    
    elif crystalclass == 9: ## hexagonal
        C11,C12,C13,C33,C44 = Cijs
        C = np.array([[C11,C12,C13,0,0,0],
                      [C12,C11,C13,0,0,0],
                      [C13,C13,C33,0,0,0],
                      [0,0,0,C44,0,0],
                      [0,0,0,0,C44,0],
                      [0,0,0,0,0,(C11-C12)/2.]])
    
    else:
        raise ValueError('invalid crystal class!')
    
    return C

    
def expand_C(C_voigt):      
   
    """ 
    expand the 6x6 C in Voigt notation to full 3x3x3x3 C tensor
	
    Parameters
    ----------
    C_voigt : stiffness tensor C in Voight notation, in cartesian basis

    Returns
    -------
    C : 3x3x3x3 stiffness tensor C in cartesian basis
    
    """
    
    ## Voight notation: xx, yy, zz, yz, xz, xy
    mapping = [[0,0],[1,1],[2,2],[1,2],[0,2],[0,1]]   
     
    C = np.zeros((3,3,3,3)) 
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    C[i,j,k,l] = C_voigt[mapping.index([min(i,j),max(i,j)]),
                                         mapping.index([min(k,l),max(k,l)])]    
                            
    return C

    
def rotate_C(C,M):      
   
    """ 
    given the stiffness tensor C in cubic basis (pqrs indices),
    rotate it to mnt basis (ijkl indices)
    C_ijkl = R_ip.R_jq.R_kr.R_ls.C_pqrs
	
    Parameters
    ----------
    C : 3x3x3x3 stiffness tensor C in cartesian basis
    M : 3x3 matrix for rotating from mnt basis to cartesian basis

    Returns
    -------
    C_mnt : 3x3x3x3 stiffness tensor C in mnt basis
    
    """
    
    R = M.T     
    C_mnt = np.tensordot(R,np.tensordot(R,np.tensordot(R,np.tensordot(R,C,
                         axes=([1],[0])),axes=([1],[1])),axes=([1],[2])),axes=([1],[3]))    
                            
    return C_mnt
 

#def Lambda2(k,RD,Dflat,M,A,a0):
#
#    """
#    Calculates Lambda2(k) = leading term in D(k)
#    Lambda2(k) = -0.5 * summation_over_R(D(R)*(k_hat.R)^2)
#    
#    ** THIS VERSION BASED ON FORCE-CONSTANTS INSTEAD OF ELASTIC CONSTANTS
#        IS NOT APPLICABLE TO MULTIPLE ATOM BASIS AND IS DEPRECIATED!! **
#    
#    Parameters
#    ----------
#    k     : k_hat vector (direction only, magnitude 1)
#    RD    : list of vectors between pairs of atoms within the thin slab,
#            given in primitive cell basis
#    Dflat : list of "flattened" 3x3 force constant matrices corresponding to the atom pairs,
#            listed in the same order as RD
#    M     : 3x3 matrix for rotating from mnt basis to cartesian basis
#    A     : 3x3 matrix for rotating from primitive cell basis to cartesian basis
#    a0    : lattice constant in angstroms
#
#    Returns
#    -------
#    L2 : 3x3 Lambda2(k) matrix evaluated for the given k vector
#    
#    Notes
#    -----
#    Refer to: D.R. Trinkle, Phys. Rev. B 78, 014110 (2008)
#    
#    """ 
#    
#    L2 = np.zeros((3,3))
#
#    for RDi,Di in zip(RD,Dflat):
#        ## convert RD vectors to mnt basis for this
#        L2 -= 0.5 * Di * (np.dot(k,a0*np.dot(M.T,np.dot(A,RDi))))**2
#
#    return L2
    

#def EGF_Fcoeffs(N,RD,Dflat,M,A,a0):
#
#    """
#    Calculates EGF(k) = inv(D(k)) = inv(Lambda2(k)),
#    then expands EGF as a fourier series in the mn plane.
#    
#    ** THIS VERSION BASED ON FORCE-CONSTANTS INSTEAD OF ELASTIC CONSTANTS
#        IS NOT APPLICABLE TO MULTIPLE ATOM BASIS AND IS DEPRECIATED!! **
#    
#    Parameters
#    ----------
#    N     : number of k vectors to generate and compute EGF(k) for
#    RD    : list of vectors between pairs of atoms within the thin slab,
#            given in primitive cell basis
#    Dflat : list of "flattened" 3x3 force constant matrices corresponding to the atom pairs,
#            listed in the same order as RD
#    M     : 3x3 matrix for rotating from mnt basis to cartesian basis
#    A     : 3x3 matrix for rotating from primitive cell basis to cartesian basis
#    a0    : lattice constant in angstroms
#
#    Returns
#    -------
#    GEn : list of complex ndarrays of fourier coefficients
#          evaluated for each component of the EGF, i.e.
#          [coeffs for xx component, coeffs for yy component, coeffs for zz component,
#           coeffs for yz component, coeffs for xz component, coeffs for xy component]
#    
#    Notes
#    -----
#    Refer to: D.R. Trinkle, Phys. Rev. B 78, 014110 (2008)
#    
#    """ 
#
#    ## generate list of k vectors that lie on m-n plane; angle is wrt +m axis    
#    ## calculates Lambda2(k) = leading term in D(k), then inverts Lambda2(k) to get EGF(k)  
#    EGF = [la.inv(Lambda2(k,RD,Dflat,M,A,a0)) 
#           for k in [[np.cos(2*np.pi*n/N),np.sin(2*np.pi*n/N),0] for n in range(N)]]
#    
#    ## create separate lists for each independent component of the 3x3 EGF matrices
#    EGF_xx,EGF_xy,EGF_xz = list(zip(*(list(zip(*EGF))[0])))
#    EGF_yx,EGF_yy,EGF_yz = list(zip(*(list(zip(*EGF))[1])))
#    EGF_zx,EGF_zy,EGF_zz = list(zip(*(list(zip(*EGF))[2])))
#
#    return np.fft.fft([EGF_xx,EGF_yy,EGF_zz,EGF_yz,EGF_xz,EGF_xy])/len(EGF)

    
def Lambda2_kCk(k,C,M):      
   
    """    
    compute Lambda2(k) = V*[k.C.k]
    [k.C.k]_qr = sum_p,s (k[p]*C_pqrs*k[s])
     
    Parameters
    ----------
    k : unit vector in the mn plane, expressed in mnt coords
    C : 3x3x3x3 stiffness tensor C in cartesian basis
    M : 3x3 matrix for rotating from mnt basis to cartesian basis

    Returns
    -------
    L2 : 3x3 Lambda2(k) matrix evaluated for the given k vector
	
    """

    k = np.dot(M,k)  ## convert k in the mn plane to cartesian coords
    L2 = np.tensordot(k,np.tensordot(C,k,axes=([3],[0])),axes=([0],[0]))
    ## I've ignored the V here as it will cancel out later... 
                            
    return L2


def EGF_Fcoeffs(N,C,M):

    """
    Calculates EGF(k) = inv(D(k)) = inv(Lambda2(k)),
    then expands EGF as a fourier series in the mn plane.
    
    Parameters
    ----------
    N  : number of k vectors to generate and compute EGF(k) for
    C  : 3x3x3x3 stiffness tensor C in cartesian basis
    M  : 3x3 matrix for rotating from mnt basis to cartesian basis

    Returns
    -------
    GEn : list of complex ndarrays of fourier coefficients
          evaluated for each component of the EGF, i.e.
          [coeffs for xx component, coeffs for yy component, coeffs for zz component,
           coeffs for yz component, coeffs for xz component, coeffs for xy component]
    
    Notes
    -----
    Refer to: D.R. Trinkle, Phys. Rev. B 78, 014110 (2008)
    
    """ 

    ## generate list of k vectors that lie on m-n plane; angle is wrt +m axis    
    ## calculates Lambda2(k) = leading term in D(k), then inverts Lambda2(k) to get EGF(k)  
    EGF = [la.inv(Lambda2_kCk(k,C,M)) 
           for k in [[np.cos(2*np.pi*n/N),np.sin(2*np.pi*n/N),0] for n in range(N)]]
    
    ## create separate lists for each independent component of the 3x3 EGF matrices
    EGF_xx,EGF_xy,EGF_xz = list(zip(*(list(zip(*EGF))[0])))
    EGF_yx,EGF_yy,EGF_yz = list(zip(*(list(zip(*EGF))[1])))
    EGF_zx,EGF_zy,EGF_zz = list(zip(*(list(zip(*EGF))[2])))

    return np.fft.fft([EGF_xx,EGF_yy,EGF_zz,EGF_yz,EGF_xz,EGF_xy])/len(EGF)


def G_largeR_ang(GEn,N,N_max):

    """
    Calculates the angular term in the real space large R LGF
    for a given finite number of angular values only.
    
    Parameters
    ----------
    GEn   : list of complex ndarrays of fourier coefficients evaluated for each component of EGF
    N     : total number of fourier components in the fourier series; 
            also equal to the total number of angular values for which the
            angular term in the real space large R LGF is to be explicitly computed
    N_max : maximum number of fourier components to consider in the truncated fourier series
          
    Returns
    -------
    phi_R_grid : list of angular terms in the real space large R LGF, 
                 computed for N equally-spaced angular (phi) values.
                 Same as GEn, each entry in the list corresponds to
                 the values for the different components of the LGF:                
                 [xx components, yy comps, zz comps, yz comps, xz comps, xy comps]
    
    Notes
    -----
    Refer to: D.R. Trinkle, Phys. Rev. B 78, 014110 (2008)
    
    """ 

    ang_coeffs = np.zeros([6,N],dtype=np.complex128)
    
    for ang_coeffs_comp,GEn_comp in zip(ang_coeffs,GEn):
        ## evaluate separately for each of the 6 independent components
        for n in range (1,N_max+1):
            ## include coeffs of exp(i*n*phi) as well as exp(-i*n*phi) in the truncated series
            ang_coeffs_comp[n] += (((-1.)**(n/2))/n)*GEn_comp[n]
            ang_coeffs_comp[-n] += (((-1.)**(n/2))/n)*GEn_comp[-n]

    return np.fft.ifft(ang_coeffs)*N


def G_largeR(GEn,phi_R_grid,R,phi,N,a0,t_mag):

    """
    Calculates the real space large R LGF for given R, phi values.
    
    Parameters
    ----------
    GEn        : list of complex ndarrays of fourier coefficients evaluated for each component of EGF
    phi_R_grid : list of angular terms in the real space large R LGF, computed for 
                 N equally-spaced angular (phi) values. Same as GEn, each entry in 
                 the list corresponds to the values for the different components of LGF.
    R          : in-plane distance (on the m-n plane) between 2 atoms
    phi        : in-plane angle between 2 atoms, measured relative to +m axis
    N          : number of angular values for which the angular term in the real space large R LGF 
                 has been explicitly computed
    a0         : lattice constant in angstroms
    t_mag      : magnitude of the periodic vector along the dislocation threading direction
          
    Returns
    -------
    G_largeR   : 3x3 real space large R LGF matrix between 2 points
                 separated by distance R and angle phi
    
    Notes
    -----
    Refer to: D.R. Trinkle, Phys. Rev. B 78, 014110 (2008)
    
    """ 

    phi_ongrid = phi/(2*np.pi/N)
    ## angular values between which to interpolate
    lower = int(np.floor(phi_ongrid))
    upper = int(lower + 1)
    lever_lower = phi_ongrid-lower
    lever_upper = upper-phi_ongrid
    logR = np.log(R)
    
    ## evaluate separately for each of the 6 independent components
    ## I've ignored the V here as it cancels out the V in the Lambda(2) term
    G_largeR_comps = [((1./(2*np.pi*a0*t_mag)) * (-GEn_comp[0]*logR + 
                      lever_lower*phi_R_grid_comp[upper%N] + lever_upper*phi_R_grid_comp[lower])) 
                      for GEn_comp,phi_R_grid_comp in zip(GEn,phi_R_grid)]
        
    return [[G_largeR_comps[0].real,G_largeR_comps[5].real,G_largeR_comps[4].real],
            [G_largeR_comps[5].real,G_largeR_comps[1].real,G_largeR_comps[3].real],
            [G_largeR_comps[4].real,G_largeR_comps[3].real,G_largeR_comps[2].real]]
    
       