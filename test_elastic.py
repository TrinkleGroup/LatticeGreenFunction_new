import numpy as np
import unittest
import elastic                  


def AnalyticEGF(R,phi):

    ## analytic solution for EGF in isotropic medium
    ## expressions from Balluffi and evaluated in mathematica
    ## v and u are the poisson ratio and shear modulus
    ## corresponding to the square lattice system with force constants k1=k2=1

    v = 0.25  ## poisson ratio
    u = 1.0  ## shear modulus 

    Gxx = ((3 - 4*v)*np.log(R) + (np.sin(phi))**2)/(8*np.pi*u*(v-1))
    Gyy = ((3 - 4*v)*np.log(R) - (np.sin(phi))**2)/(8*np.pi*u*(v-1))
    Gxy = -(np.sin(2*phi))/(16*np.pi*u*(v-1))
    Gzz = -(np.log(R))/(2*np.pi*u)

    return [[Gxx,Gxy,0.],[Gxy,Gyy,0.],[0.,0.,Gzz]]

        
class TestLGF(unittest.TestCase):

    def setUp(self):
        
        ## square lattice system with force constants k1=k2=1
        
        self.crystalclass = 0  ## isotropic
        self.a0 = 1.0
        self.V = 1.0       
        self.Cijs = [3,1]  ## the corresponding C11,C12
        self.C = elastic.expand_C(elastic.construct_C(self.crystalclass,self.Cijs))
        
        self.M = np.array([[1,0,0],[0,1,0],[0,0,1]])
        self.t_mag = 1.0 
        
        self.N = 256


    def test_analytic(self):
        
        ## runs check_analytic test for multiple random R, phi values
        
        self.GEn = elastic.EGF_Fcoeffs(self.N,self.C,self.M,self.V)
        self.phi_R_grid = elastic.G_largeR_ang(self.GEn,self.N,int(self.N/2))
        
        R = 100*np.random.rand(100)
        phi = np.random.rand(100)
        for R_i,phi_i in zip(R,phi):
            TestLGF.check_analytic(self,R_i,2*np.pi*phi_i)
    
            
    def check_analytic(self,R,phi):  
                
        ## test if the large R LGF computed agrees with the analytic solution for isotropic EGF
        ## (up to a constant shift?)
       
        LGF = np.array(elastic.G_largeR(self.GEn,self.phi_R_grid,R,phi,self.N,self.a0,self.V,self.t_mag))
        EGF = np.array(AnalyticEGF(R,phi))
                   
        ## the shift is independent of R, phi. So I just choose arbitrary values for R, phi to evaluate the shift.    
        shift = (np.array(elastic.G_largeR(self.GEn,self.phi_R_grid,10,0.5,self.N,self.a0,self.V,self.t_mag)) - 
                 np.array(AnalyticEGF(10,0.5)))

        for i in range(3):
            for j in range(3):  
                self.assertAlmostEqual((LGF-EGF)[i,j],shift[i,j],places=4)  

    
if __name__ == '__main__':

  
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLGF)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    