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
    N_remove_central : int
        the number of points to be removed in the center of the 
        frequency spectrum (symmetrically on both sides)
    N_cut : int
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
    
    def __init__(self, t, w1, w2, method='solve'):   
    
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
        self.N_remove_central = np.max([np.argmin(abs(self.w[self.N//2:] - w1)), 1])    
        last_ind = np.argwhere(self.w>w2)[0][0] - 1
        self.N_cut = self.N - last_ind    
        self.method=method      
        self.__start = int(self.N_remove_central>0) + self.N_cut
                   
        
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
        fft_eta_shad = np.fft.fftshift(np.fft.fft(eta_shad))/self.N
        fft_deconv = np.linalg.solve(A, fft_eta_shad)            
        eta_deconv = np.real(np.fft.ifft(np.fft.ifftshift(fft_deconv*self.N)))
        eta_deconv[1:Nr] = eta_deconv[1:self.N]
        if replace_all:
            eta_out = eta_deconv
        else:
            eta_out = np.where(illu, eta_shad, eta_deconv)
        
        if plot_spec:
            self.__plot_spectrum(fft_eta_shad, fft_deconv)
            
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
        lower_half = np.arange(self.N_cut, self.N_half-self.N_remove_central)
        upper_half = np.arange(self.N_half+1+(self.N_remove_central), self.N-self.__start)
        choose = list(lower_half.copy())        
        choose.extend([self.N_half])
        choose.extend(list(upper_half))
        A_reduced = A[:, choose]         
        
        # Solve
        fft_deconv_reduced = solve_with_options_reduced(A_reduced, fft_eta_shad, self.method)
        
        fft_deconv_reduced = np.flipud(fft_deconv_reduced)
        
        # Merge solution into full system 
        fft_deconv = np.zeros(self.N, dtype=complex)
        N_red_half = len(fft_deconv_reduced) // 2
        fft_deconv[self.__start:self.N_half-self.N_remove_central] = fft_deconv_reduced[: N_red_half]
        # Ensure symmetric spectrum: TODO: check if improvement is possible here
        fft_deconv[self.N_half+1:] = np.conjugate(np.flipud(fft_deconv[1:self.N_half]))             

        # From spectral to physical domain
        eta_deconv = np.real(np.fft.ifft(np.fft.ifftshift(fft_deconv*self.N)))
        eta_deconv[1:self.N] = eta_deconv[1:self.N]
        if plot_spec:
            self.__plot_spectrum(fft_eta_shad, fft_deconv)           

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
