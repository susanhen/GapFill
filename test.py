import unittest
import numpy as np
from deconvolution import DeconvolutionSetup, DeconvolutionFramework


class TestDeconvolution(unittest.TestCase):

    def setUp(self):
        self.N = 100
        self.t = np.linspace(0, 500, self.N)
        dt = abs(self.t[-1]-self.t[-2])    
        wmin = -np.pi/dt
        dw = 2*np.pi/(dt*self.N)        
        self.w = wmin + dw*np.arange(0, self.N)
        self.w1 = 0.03
        self.w2 = 0.2
        self.dec_set = DeconvolutionSetup(self.w, self.w1, self.w2)
        
    def test_convolution(self):
        a = np.sign(np.sin(self.t*10))
        b = np.cos(self.t)
        fft_a = np.fft.fftshift(np.fft.fft(a))/self.N
        fft_b = np.fft.fftshift(np.fft.fft(b))/self.N
        A = self.dec_set.build_convolution_matrix(a)
        matrix_convolution = np.flipud(np.dot(A, np.flipud(fft_b)))
        
        N_half = int(0.5*self.N)
        numpy_convolution = np.convolve(fft_a, fft_b)[N_half:N_half+self.N]
        for i in range(0,self.N):
            self.assertAlmostEqual(matrix_convolution[i], numpy_convolution[i])     
    
    def test_deconvolution(self):
        a = np.ones(self.N)
        ind1 = int(0.45*self.N)
        ind2 = int(0.55*self.N)
        a[ind1:ind2] = 0
        RMS_expected = 0.008411               
        b = np.cos(0.05*self.t) + 0.2*np.cos(0.07*self.t+0.3*np.pi)   
        b -= np.mean(b)     
        b_dec = self.dec_set.interpolate(b, a)
        diff = np.abs(b[ind1:ind2] - b_dec[ind1:ind2])**2
        RMS = np.round(np.sqrt(np.mean(diff)), 6)
        self.assertEqual(RMS, RMS_expected)    

    def test_deconvolution_non_symmetric(self):
        a = np.ones(self.N)
        ind1 = int(0.45*self.N)
        ind2 = int(0.55*self.N)
        a[ind1:ind2] = 0
        RMS_expected = 0.008411
        b = np.cos(0.05*self.t) + 0.2*np.cos(0.07*self.t+0.3*np.pi)   
        b -= np.mean(b)     
        b_dec = self.dec_set.interpolate_non_symmetric(b, a)
        diff = np.abs(b[ind1:ind2] - b_dec[ind1:ind2])**2
        RMS = np.round(np.sqrt(np.mean(diff)), 6)
        self.assertEqual(RMS, RMS_expected)


class TestDeconvolutionFramework(unittest.TestCase): 
 
    def setUp(self):  
        self.N = 200
        self.t = np.linspace(0, 1000, self.N)
        self.y = np.cos(0.05*self.t) + 0.2*np.cos(0.07*self.t+0.3*np.pi)   
        self.y -= np.mean(self.y)
        self.m = np.ones(self.N)
        ind1 = int(0.22*self.N)
        ind2 = int(0.27*self.N)
        self.m[ind1:ind2] = 0
        ind1 = int(0.47*self.N)
        ind2 = int(0.52*self.N)
        self.m[ind1:ind2] = 0
        self.N_intervals = 2
        self.dec_framework = DeconvolutionFramework(self.t, self.y*self.m, self.m, self.N_intervals, plot_it=False)    
        
    def test_data_chopping(self):
        N_int = self.dec_framework.N_intervals
        N_per_int = self.dec_framework.N_per_interval
        start1_compare = 0
        start2_compare = N_per_int//2
        for i in range(0, N_int-1):         
            start1 = self.dec_framework.start_points1[i]
            self.assertEqual(start1, start1_compare)
            start1_compare += N_per_int
            end1 = self.dec_framework.end_points1[i]
            self.assertEqual(start1_compare, end1)
                     
            start2 = self.dec_framework.start_points2[i]            
            self.assertEqual(start2, start2_compare)
            start2_compare += N_per_int
            end2 = self.dec_framework.end_points2[i]
            self.assertEqual(start2_compare, end2)            
                    
        start1 = self.dec_framework.start_points1[-1]
        self.assertEqual(start1, start1_compare)
        start1_compare += N_per_int
        end1 = self.dec_framework.end_points1[-1]
        self.assertEqual(start1_compare, end1)
        
    def test_deconvolution(self):
        dec = self.dec_framework.deconvolve()
        diff = np.abs(dec - self.y)**2
        RMS = np.round(np.sqrt(np.mean(diff)), 6)
        RMS_expected = 0.004085
        self.assertEqual(RMS, RMS_expected)    

if __name__ == '__main__':
    unittest.main()
