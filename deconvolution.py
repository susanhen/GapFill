#!/usr/bin/env python

import numpy as np
from scipy.linalg import toeplitz, circulant

def solve_with_options_reduced(A_reduced, b, method):
    def square_A_reduced(A_reduced):
        return np.dot(np.conjugate(np.transpose(A_reduced)),A_reduced )        
        
    def square_b(A_reduced, b):
        return np.dot(np.conjugate(np.transpose(A_reduced)), b)
        
    method_dict = { 'solve': np.linalg.solve(square_A_reduced(A_reduced), square_b(A_reduced,b)),
                    'lstsq': np.linalg.lstsq(A_reduced ,b, rcond=0.2)[0]}
    return method_dict.get(method)    
    
def select_indices(w, w1, w2, w0):  
    N = len(w)
    N_lower_cut_off = np.max([np.argmin(abs(w[N//2:] - w1)), 1])    
    if w2<w[-1]:
        last_ind = np.argwhere(w>w2)[0][0] - 1
        N_upper_cut_off = N - last_ind    
    else:
        N_upper_cut_off = 0
    selected_indices = list(np.arange(N_upper_cut_off, N//2-1-N_lower_cut_off)) 
    refill_selected_indices = list(np.arange(N_upper_cut_off, N//2-1-N_lower_cut_off)) 
    if w0:      
        selected_indices.extend([N//2-1])
        refill_selected_indices.extend([N//2-1])
    selected_indices.extend(list(np.arange(N//2 + N_lower_cut_off, N-1-N_upper_cut_off)))     
    if w0: # to allow even number of columns in matrix
        selected_indices.extend([N-1])
        refill_selected_indices.extend([N-1])
        
    return selected_indices, refill_selected_indices
    
    
class Deconvolution:
    """
    A class to perform the deconvolution as interpolation

    ...

    Attributes
    ----------
    t : array
        time array for measurements (assumed to be uniform)
    N : int
        length of measurement and time array
    N_half : int
        half the length of measurement and time array
    w : array
        frequency array for t
    N_upper_cut_off_off : int
        the number of points to be removed in the center of the 
        frequency spectrum (symmetrically on both sides)
    N_upper_cut_off : int
        the number of points above the cut of frequency
    method : string
        method for solving deconvolution options: 'solve', 'lstsq'
    

    Methods
    -------
    build_convolution_matrix(illu)
        forms the convolution matrix for the observation function illu
    direct_interpolate(self, eta_shad, illu, plot_spec=False, replace_all=False)
        interpolates the missing points directly without reducing the matrix
    interpolate(self, eta_shad, illu, plot_spec=False, replace_all=False)
        interpolates the missing points by reducing the matrix
    """
    
    def __init__(self, t, w1, w2, w0=False, method='solve'):   
    
        """
        Parameters
        ----------
        t : array
            time array for measurements (assumed to be uniform)
        w1 : float
            lower limit on the frequency line for the bandwidth limitation
        w2 : float
            upper limit on the frequency line for the bandwidth limitation            
        method : string, optional
            The method for interpolating (default is 'solve')
        """
        self.t = t
        self.N = len(t)
        self.N_half = int(0.5*self.N)
        dt = abs(t[-1]-t[-2])    
        wmin = -np.pi/dt
        dw = 2*np.pi/(dt*self.N)        
        self.w = wmin + dw*np.arange(0,self.N)
        if w2>self.w[-1]:
            w2 = self.w[-1]
            print("w2 was set to {0:.4f}, input value was too large".format(self.w[-1]))
        if w2<w1:
            w2 = self.w[-1]
            print("w2 was set to {0:.4f}, since given w2<w1".format(self.w[-1]))
            
        if w1<0:
            w1 = 0
            print("w2 was set to 0, since the w1 was given with a negative value".format(self.w[-1]))
      
        self.selected_indices, self.refill_selected_indices = select_indices(self.w, w1, w2, w0)
        
        if w0:
            self.block_parameter = 'with_center'
        else:
            self.block_parameter = 'without_center'
        
        
        self.method=method      
     
    def __define_block_matrix(self, A_reduced):
        N_rows, N_cols = A_reduced.shape
        def block_matrix_with_center(A_reduced):
            L = A_reduced[:, :N_cols//2-1]
            C = np.flipud(A_reduced[:, N_cols//2:-1])
            M = (A_reduced[:, N_cols//2-1]).reshape((N_rows, 1))
            E = (A_reduced[:, -1]).reshape((N_rows, 1))
            B = np.block([[L.real + C.real, M.real, E.real, -L.imag + C.imag, -M.imag, -E.imag],
                          [L.imag + C.imag, M.imag, E.imag,  L.real - C.real,  M.real,  E.real]])
            return B
                             
        def block_matrix_without_center(A_reduced):            
            L = A_reduced[:, :N_cols//2]
            C = np.fliplr(A_reduced[:, N_cols//2:])
            B = np.block([[L.real + C.real, -L.imag + C.imag],
                          [L.imag + C.imag, L.real - C.real]])
            return B                    
        block_dict = { 'with_center': block_matrix_with_center(A_reduced),
                       'without_center': block_matrix_without_center(A_reduced)}                 
        return block_dict.get(self.block_parameter)                      
        
    def build_convolution_matrix(self, illu):    
        """
        Parameters
        ----------
        illu : array
            observation function consiting of ones and zeros, deconvolution will be active at zeros
        """    
        illu = illu.astype(int) # this operation is important in case illu includes booleans
        fft_illu = np.flipud(np.fft.fftshift(np.fft.fft(illu)))/self.N
        lower_fft_illu = np.zeros(self.N, dtype=complex)
        upper_fft_illu = np.zeros(self.N, dtype=complex)
        upper_fft_illu[:self.N_half] = np.flipud(fft_illu[:self.N_half])
        lower_fft_illu[:self.N_half+1] = fft_illu[self.N_half-1:]
        return toeplitz((lower_fft_illu), (upper_fft_illu))

        
    def __plot_spectrum(self, fft_eta_shad, fft_deconv):
        import pylab
        pylab.plot(abs(fft_eta_shad)**2, label='fft_eta_shad')
        pylab.plot(abs(fft_deconv)**2, label='fft_deconv')
        pylab.legend()
        pylab.show()
    
    def direct_interpolate(self, eta_shad, illu, plot_spec=False, replace_all=False):    
        """
        Parameters
        ----------
        eta_shad : array
            data with invalid points (the invalid points will be set to zero)
        illu : array
            observation function consiting of ones and zeros, deconvolution will be active at zeros
        plot_spec : bool, optional
            set true to plot the spectra before and after deconvolution (default is False)
        replace_all : bool, optional
            set True if all values should be replaced, False means new values are returned only in gaps (default is False)
        """
        A = build_convolution_matrix(illu)
        Nr = len(illu)
        fft_eta_shad = np.flipud(np.fft.fftshift(np.fft.fft(eta_shad)))/self.N
        fft_deconv = np.flipud(np.linalg.solve(A, fft_eta_shad))
        eta_deconv = np.real(np.fft.ifft(np.fft.ifftshift(fft_deconv*self.N)))
        eta_deconv[1:Nr] = eta_deconv[1:self.N]
        if replace_all:
            eta_out = eta_deconv
        else:
            eta_out = np.where(illu, eta_shad, eta_deconv)
        
        if plot_spec:
            self.__plot_spectrum(np.flipud(fft_eta_shad), fft_deconv)
            
        return eta_out
        
        
    def interpolate(self, eta_shad, illu, plot_spec=False, replace_all=False):  
        """
        Parameters
        ----------
        eta_shad : array
            data with invalid points (the invalid points will be set to zero)
        illu : array
            observation function consiting of ones and zeros, deconvolution will be active at zeros
        plot_spec : bool, optional
            set true to plot the spectra before and after deconvolution (default is False)
        replace_all : bool, optional
            set True if all values should be replaced, False means new values are returned only in gaps (default is False)
        """
        A = self.build_convolution_matrix(illu)
        fft_eta_shad = np.flipud(np.fft.fftshift(np.fft.fft(eta_shad)))/self.N        
        
        # Reduce matrix
        A_reduced = A[:, self.selected_indices] 
        
        # decompose complex system into real system
        B = self.__define_block_matrix(A_reduced)
        vec = np.block([fft_eta_shad.real, fft_eta_shad.imag]) 
        fft_deconv_reduced = solve_with_options_reduced(B, vec, self.method)
        
        # compose complex coefficients
        N_red_half = len(fft_deconv_reduced) // 2
        fft_deconv_reduced = fft_deconv_reduced[:N_red_half] + 1j*fft_deconv_reduced[N_red_half:]
        
        # synthesize coefficient vector from reduced coefficient vector
        fft_deconv = np.zeros(self.N, dtype=complex)
        fft_deconv[self.refill_selected_indices] = fft_deconv_reduced
        fft_deconv[self.N_half:-1] = np.conjugate(np.flipud(fft_deconv[:self.N_half-1])) 

        # revert initial flip
        fft_deconv = np.flipud(fft_deconv) 
        
        # From spectral to physical domain
        eta_deconv = np.real(np.fft.ifft(np.fft.ifftshift(fft_deconv*self.N)))        
        
        if plot_spec:
            self.__plot_spectrum(np.flipud(fft_eta_shad), fft_deconv)           

        if replace_all:
            eta_out = eta_deconv
        else:
            eta_out = np.where(illu, eta_shad, eta_deconv)
        return eta_out


    def interpolate_non_symmetric(self, eta_shad, illu, plot_spec=False, replace_all=False):  
        """
        Parameters
        ----------
        eta_shad : array
            data with invalid points (the invalid points will be set to zero)
        illu : array
            observation function consiting of ones and zeros, deconvolution will be active at zeros
        plot_spec : bool, optional
            set true to plot the spectra before and after deconvolution (default is False)
        replace_all : bool, optional
            set True if all values should be replaced, False means new values are returned only in gaps (default is False)
        """
        A = self.build_convolution_matrix(illu)
        fft_eta_shad = np.flipud(np.fft.fftshift(np.fft.fft(eta_shad)))/self.N
        
        # Reduce matrix    
        A_reduced = A[:, self.selected_indices]         
    
        # Solve
        fft_deconv_reduced = solve_with_options_reduced(A_reduced, fft_eta_shad, self.method)
        
        # Merge solution into full system 
        fft_deconv = np.zeros(self.N, dtype=complex)
                
        fft_deconv[self.selected_indices] = fft_deconv_reduced
        fft_deconv = np.flipud(fft_deconv) # revert initial flip
            
        # From spectral to physical domain
        eta_deconv = np.real(np.fft.ifft(np.fft.ifftshift(fft_deconv*self.N)))
        
        if plot_spec:
            self.__plot_spectrum(np.flipud(fft_eta_shad), fft_deconv)           

        if replace_all:
            eta_out = eta_deconv
        else:
            eta_out = np.where(illu, eta_shad, eta_deconv)
        return eta_out


        
if __name__ == '__main__':
    import pylab as plt
    from scipy.stats import rice
    
    def construct_mask(N, N_gaps, mean_length):
        illu = np.ones(N)
        # vary position of start indices around a uniform grid based on uniform distribution
        start_indices =  np.arange(1,N+1)*int(N/(N_gaps+2))+ int(np.random.uniform(0.05*N))
        for i in range(0, N_gaps):
           illu[start_indices[i]:start_indices[i]+int(np.random.normal(mean_length, 2))] = 0
        return illu  

    # define wave
    N = 390          
    N_modes = 20
    b = 0.775
    phi = np.random.uniform(0,2*np.pi, N_modes)
    r = np.linspace(100, 2100, N)
    k = np.linspace(0.01, 0.2, N_modes) 
    x = np.linspace(-1,5, N_modes)
    amp = rice.pdf(x, b)
    eta = np.dot(amp, np.sin(np.outer(k, r) + np.outer(phi, np.ones(len(r))) ))

    # define random masks  
    N_gaps = 15
    mean_gap_length = 8
    illu = construct_mask(N, N_gaps, mean_gap_length)    

    
    w1 = 0.02
    w2 = 0.16
    Dec = Deconvolution(r, w1, w2)
    
    
    eta_dec = Dec.interpolate(eta*illu, illu, replace_all=True, plot_spec=True)
    
    # plot result
    plt.figure()
    plt.plot(r, eta, 'k', label='original')
    plt.plot(r, illu, ':k', label='observation function')
    plt.plot(r, eta_dec, 'r--', label='deconvolved')
    plt.ylim([-3.,3.])
    gap_indices = np.argwhere(illu).transpose()[0]
    RMS_by_sig = np.sqrt(np.mean((eta[gap_indices] - eta_dec[gap_indices])**2)) / np.sqrt(np.var(eta))
    plt.text(100, -2.6, r'RMS/sigma = {0:1.3f}'.format(RMS_by_sig),
         {'color': 'black', 'fontsize': 12, 'ha': 'left', 'va': 'center',
          'bbox': dict(boxstyle="round", fc="white", ec="black", pad=0.2)})
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,  ncol=4, mode="expand", borderaxespad=0.)
    plt.show()
