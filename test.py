import unittest
import numpy as np
from deconvolution import Deconvolution


class TestDeconvolution(unittest.TestCase):

    def setUp(self):
        self.N = 100
        self.t = np.linspace(0, 500, self.N)
        self.w1 = 0.03
        self.w2 = 0.2
        self.dec = Deconvolution(self.t, self.w1, self.w2)
        
    def test_convolution(self):
        a = np.sign(np.sin(self.t))
        b = np.cos(self.t)
        fft_a = np.fft.fftshift(np.fft.fft(a))/self.N
        fft_b = np.fft.fftshift(np.fft.fft(b))/self.N
        A = self.dec.build_convolution_matrix(a)
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
        RMS_expected = 0.027875                
        b = np.cos(0.05*self.t) + 0.2*np.cos(0.07*self.t+0.3*np.pi)        
        b_dec = self.dec.interpolate(b, a)
        diff = np.abs(b[ind1:ind2] - b_dec[ind1:ind2])**2
        RMS = np.round(np.sqrt(np.mean(diff)), 6)
        self.assertEqual(RMS, RMS_expected)
        
    def test_exploit_symmetry(self):
        a = np.sign(np.sin(self.t))
        b = np.cos(self.t)
        fft_a = np.fft.fftshift(np.fft.fft(a))/self.N
        fft_b = np.fft.fftshift(np.fft.fft(b))/self.N
        A = self.dec.build_convolution_matrix(a)
        input_vector = np.flipud(fft_b)
        
        N = len(a)
        N_half = N//2
        row_indices = list(np.arange(0, N_half))
        row_indices.extend([N-1])
        
        L = A[row_indices, :N_half-1]
        R = A[row_indices, N_half:-1]
        C = np.fliplr(R)
        M = A[row_indices, N_half-1].reshape((len(row_indices), 1))
        E = A[row_indices, -1].reshape((len(row_indices), 1))
        u = input_vector[:N_half-1]
        m = input_vector[N_half-1]
        e = input_vector[-1]
        Ax = np.dot(A, input_vector)
 
        mat = np.block([[L.real + C.real, M.real, E.real, -L.imag + C.imag, -M.imag, -E.imag],
                        [L.imag + C.imag, M.imag, E.imag,  L.real - C.real,  M.real,  E.real]])
        vec = np.block([u.real, m.real, e.real, u.imag, m.imag, e.imag])
        conv2 = np.dot(mat, vec)
        conv_complex = conv2[:N_half+1] + 1j*conv2[N_half+1:]
        for i in range(0,N_half):
            self.assertAlmostEqual(conv_complex[i], Ax[i])
        
        self.assertAlmostEqual(conv_complex[-1], Ax[-1])

if __name__ == '__main__':
    unittest.main()
