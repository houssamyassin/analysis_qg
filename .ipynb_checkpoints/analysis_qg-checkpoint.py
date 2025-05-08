import numpy as np
import numpy.fft as npfft
from matplotlib import pyplot as plt
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os
import colorcet as cc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm
import re
from sympy import sympify
from scipy.integrate import simps



def linear_step(t, period, tw, ts, a, b):
    """
    Generates a linear step function over a given period.
    
    Parameters:
    t (float): Current time
    period (float): Total period of the step function
    tl (float): Transition time between steps
    a (float): Initial value
    b (float): Peak value
    
    Returns:
    float: The value of the step function at time t
    """
    tc = 0.5 * (period - tw - ts)
    t = (t+0.25*period) % period 
    
    if 0 <= t <= 0.5*tc:
        return a
    elif 0.5*tc < t <= 0.5*tc + tw:
        return a + (b-a) * (t - 0.5*tc) / tw
    elif 0.5*tc + tw < t <= 1.5*tc + tw:
        return b
    elif 1.5*tc + tw < t <= 1.5*tc + tw + ts:
        return b + (a-b) * (t - 1.5*tc - tw) / ts
    else:
        return a


h1_sin = lambda t: 62.5 + 37.5 * np.sin(2. * np.pi * t / (365. * 86400.))
h2_sin = lambda t: 500. - h1_sin(t)
H_sin = lambda t: np.asarray([h1_sin(t),h2_sin(t)])



def invert_L_matrix(L):
    inv_L = np.zeros_like(L)
    for i in range(L.shape[0]):
        inv_L[i] = np.linalg.inv(L[i])
    return inv_L


def invert_matrix(e):
    inv_e = np.zeros_like(e)
    for i in range(e.shape[0]):
        try:
            inv_e[i] = np.linalg.inv(e[i])
        except np.linalg.LinAlgError:
            print(f"Matrix at index {i} is not invertible. Setting to substitute value.")
            inv_e[i] = np.full_like(e[i], 0.)
    return inv_e

def create_L_from_em(e,M):
    L = np.zeros_like(e)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            M_matrix = np.diag(M[i,j])
            inv_e = np.linalg.inv(e[i,j])
            L[i,j] = -M_matrix@inv_e
    return L

def get_psi_hat(phi_hat,e):
    # phi_hat are the streamfunction for each layer
    # psi_hat are the total streamfunctions evaluated at each layer
    nz = phi_hat.shape[-1]
    psi_hat = np.zeros_like(phi_hat,dtype=np.complex128)
    for i in range(nz):
        for ii in range(nz):
            psi_hat[:,:,i] += e[:,:,i,ii]*phi_hat[:,:,ii]
    return psi_hat

def get_phi_hat(qhat,M):
    return -qhat/M

def get_vabs(field,t=-1):
    """
    Get the maximum absolute value of a field at a specific time step.

    Parameters:
    - field (numpy.ndarray): The input field, assumed to have a time dimension.
    - t (int, optional): The time step index to extract the maximum absolute value.
    Defaults to the last time step (t=-1).

    Returns:
    - float: The maximum absolute value of the field at the specified time step.
    """
    return np.max(np.abs(field[t]))

def correct_sign(array,atol=1e-8):
    # Check if the first element is close to zero
    if not np.isclose(array[0], 0,atol=atol):
        # Check if the first element is negative
        if array[0] < 0:
            # Multiply the array by a negative number
            array *= -1
    else:
        # Check if the last element is negative
        if array[-1] < 0:
            # Multiply the array by a negative number
            array *= -1
    
    return array

def find_spectral_slope(kr,spec,bounds):
    i0 = bounds[0]
    i1 = bounds[1]
    
    print("L_min = {0:.2f} km".format(2*np.pi/kr[i1]/1e3))
    print("L_max = {0:.2f} km".format(2*np.pi/kr[i0]/1e3))
    coef = np.polyfit(np.log(kr[i0:i1]),np.log(spec[i0:i1]),1)
    fit_exp = coef[0]
    print("fit_exp = {0:.2f}".format(fit_exp))
    return coef

def int_to_padded_str(num, padding='XXXXXXXXX'):
    """
    Converts an integer to a string with left padding.
    
    Args:
        num (int): The integer to convert.
        padding (str, optional): The padding string to determine the desired length.
            Default is 'XXXXXXXXX'.
    
    Returns:
        str: The string representation of the integer with left padding.
    """
    num_str = str(num)
    padding_len = len(padding)
    return num_str.zfill(padding_len)

def time_to_step(N0,days,hours=0,dt=25):
    delta = days*24*3600 + hours*3600
    dstep = delta/dt
    N = int(N0) + dstep
    if N.is_integer():
        result = int_to_padded_str(int(N))
    else:
        print("Did not result in an integer time step")
        result = None
    return result

## N is the number of numbers to generate. They will be spaced by inc.
## Generate num_N numbers spaced by inc
def generate_number_list(initial_number, N, inc=1728):
    # Convert the initial number to an integer
    initial_number_int = int(initial_number)

    # Generate the list by adding integers from initial_number to initial_number + N
    number_list = [str(initial_number_int + i*inc).zfill(len(initial_number)) for i in range(N)]

    return number_list


def plot_spectrum(x,spec,figsize=(7,5),xlims=None,ylims=None,label=None,
                 c='k',lw=1,ls='-',title='',xlabel=r'$|k|$ ($m^{-1}$)',ylabel='',
                 points=False,point_size=10):
    
    figure = plt.figure(figsize=figsize)
    ax = plt.subplot(111)
        
    ax.loglog(x,spec,c=c,lw=lw,ls=ls,label=label)
    
    if points:
        # Add scatter points at each data point
        ax.scatter(x, spec, c=c, s=point_size)  # Adjust s (size) as needed

    if xlims:
        ax.set_xlims(xlims)
    else:
        ax.set_xlim([x[0],x[-1]]);
    if ylims:
        ax.set_ylim(ylims)
    
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)



def plot_spectrum_with_phase(x, spec,figsize=(7, 5), xlims=None, ylims=None, label=None,
                              c='k', lw=1, ls='-', title='', xlabel=r'$|k|$ ($m^{-1}$)', ylabel='',
                              line=True, line_alpha=1, points=False, point_size=10, 
                              phase_plot=False, mixed_plot=True, phase=None,
                              phase_radius=0.2,phase_x=0.8,phase_y=0.8, phase0=None,
                              H=None,t=None,t0=0):
    fig = plt.figure(figsize=figsize)
    
    # Main spectrum plot
    ax1 = fig.add_subplot(111)  # Full figure
    if line:
        ax1.plot(x, spec, c=c, lw=lw, ls=ls, alpha=line_alpha, label=label)
    if points:
        ax1.scatter(x, spec, c=c, s=point_size)

    ax1.set_yscale('log')
    ax1.set_xscale('log')
        
    if xlims:
        ax1.set_xlim(xlims)
    else:
        ax1.set_xlim([0.95*x[0], x[-1]])
        
    if ylims:
        ax1.set_ylim(ylims)
        
    ax1.set_title(title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)

    if phase_plot:
        if phase:
            # Phase plot (circular)
            ax2 = fig.add_axes([phase_x, phase_y, phase_radius, phase_radius], projection='polar')  # Position and size
            ax2.plot([0,phase],[0,1],c='k')
            if phase0 is not None:
                ax2.plot([0,phase0],[0,1],c='k',alpha=0.5,lw=0.5)
            
            ax2.set_theta_zero_location('N')
            ax2.set_theta_direction(-1)
            ax2.set_xticklabels([])
            ax2.set_yticklabels([])
            ax2.grid(False,which='major')
        else:
            "Print phase argument required!"
            return
    elif mixed_plot:
        if (H is not None) and (t is not None):
            ax2 = fig.add_axes([phase_x, phase_y, 2*phase_radius, phase_radius])  # Position and size            
            year = 3600*24*365
            
            T = np.linspace(t0%year, t0%year + year,1000)
            H_arr = np.asarray([H(t)[0] for t in T])
            ax2.plot(T/year,-H_arr,c='k',lw=1)
            
            ax2.axvline(x=(t-t0)%year/year,c='k')

            ax2.set_xlim(([t0%year/year,(t0%year+year)/year]))
            #ax2.axhline(y=0,lw=0.5,c='k')
            ax2.set_ylim([-1.1*H_arr.max(),0])
    
            plt.tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False) # labels along the bottom edge are off
    
            plt.tick_params(
                axis='y',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                left=False,      # ticks along the bottom edge are off
                right=False,         # ticks along the top edge are off
                labelleft=False) # labels along the bottom edge are off
        else:
            "H and t arguments required!"
            return
    plt.tight_layout()
    #plt.show()

## Sum a real fourier series with coefficients given by Fh.
## Assumes Fh is real (e.g., for energy calculations or budgets etc).
## Assumes that the original array is even, so that Fh has dimensions of [n/2+1,n,...].
## Because this is a real FFT, the negative frequency terms are not calculated.
## The negative frequency terms should be the complex conjugates of the positive frequency terms
## but because we assume Fh is real, we can just double the series.
## For even series, both F[0] and F[-1] represent both the positive and negative Nyquest frequencies,
## so they do not need to be counted twice.
def sum_series(Fh):
    Fh = 2*Fh
    Fh[0,...] /=2
    Fh[-1,...]/2
    return Fh.sum(axis=(0,1))

class QG3:
    def __init__(self, number=None, H=None, case_dir=None,data=None, sigma=None,nz=3,
                 mode='default', averaged_vars = ['t_diag','E']):
        """
        Initialize the SQG simulation class with the provided parameters.

        Parameters:
        - Lx (float): Length of the domain in the zonal direction [m].
        - Ly (float): Length of the domain in the meridional direction [m].
        - nx (int): Number of grid points in the zonal direction.
        - ny (int): Number of grid points in the meridional direction.
        - sigma (nz array, optional): N^2/f^2, where N^2 is the buoyancy frequency at each surface 
        and f is the constant Coriolis frequency (nz array).
        - H (nz array, optional): [H1, H2](t) array where H1 is the thickness of the mixed layer and
        H2 is the thickness of the thermocline. H is a function of time.
        - U (nz array, optional): [U0, U1, U2](t) The background zonal velocity at each surface. U is a 
        function of time.
        - case_dir (str, optional): Directory path for the case. If provided, 'data' is set to
        'case_dir/data' if 'data' is not specified.
        - data (str, optional): Path to the data directory. If not provided and 'case_dir' is specified,
        it defaults to 'case_dir/data'.

        Attributes:
        - Lx, Ly, nx, ny, nz, nk, dx, dy: Domain parameters and grid information.
        - sigma, H, U: Model parameters related to stratification, layer thickness, and zonal velocity.
        - dims, kdims: Physical and Fourier space dimensions.
        - Q, P, T: Arrays for potential vorticity, streamfunction, and time.
        - E: QG total energy array for each time step.
        - m: Inversion functions at each surface for the multi streamfunction formulation.
        - e: The streamfunction coupling matrix in the multi streamfunction formulation
        - L: Inversion matrix for the single streamfunction formulation.        
        """
        self.case_dir = case_dir
        self.data = data
        if self.case_dir is not None:
            if self.data is None:
                self.data = os.path.join(self.case_dir, 'data')
            self.read_parameters()
            
            self.nk = self.nx//2+1 
            
            self.dx = self.Lx/self.nx   
            self.dy = self.Ly/self.ny   
    
            self.read_Stratification()
            self.read_MeanFlow() 
            self.dims = (self.nx, self.ny, self.nz)  
            self.kdims = (self.nk, self.ny, self.nz)
        else:
            self.sigma = sigma
            self.nz = nz
        
        self.H = H   
        self.Q = None
        self.P = None
        self.t = 0

        ## diagnostic averaged vars 
        self.t_diag = None
        self.E = None
        self.Th = None
        self.Sh = None
        self.Dh = None
        self.Rh = None
        self.Ch = None
        self.Gh = None
        self.Edot_adv = None
        self.Edot_source = None
        self.Edot_smag = None
        self.Edot_leith = None
        self.Edot_strat = None
        self.Edot_APE = None
        self.total_KE = None

        ## analysis variable
        self.m = None          
        self.e = None
        self.L = None

        self.Qh = None
        self.Ph = None

        self.Phi = None

        self.evalues = None 
        self.evectors = None
        self.growth_rate = None

        self.speed = None
        
        self.Pmh = None
        self.Qmh = None
        self.Pm = None
        self.Qm = None

        if self.case_dir is not None:
            self.initialize_global_variables()


        if mode not in ['averaged', 'default']:
            print("The 'mode' argument must be either 'default' or 'averaged'")
            return

        self.available_averaged_vars = ['t_diag', 'KE', 'total_KE', 'E', 'Th', 'Sh', 'Dh', 'Rh', 'Ch', 'Gh',
                                   'Edot_adv', 'Edot_source', 'Edot_smag', 'Edot_leith','Edot_strat','Edot_APE']
        
        if self.case_dir is not None:
            if number:
                if mode == 'default':
                    self.load_Q(number=number)
                    self.load_P(number=number)
                    self.load_T(number=number)
                elif mode == 'averaged':
                    for var_name in averaged_vars:
                        if var_name in self.available_averaged_vars:
                            # Fetch the value by calling the load_data function
                            data = self.load_data(var_name, number=number)
                            # Dynamically set the attribute with the loaded data
                            setattr(self, var_name, data)
                        else:
                            print("The list 'averaged_vars' must be a subset of  ", self.available_averaged_vars)
                if self.t_diag is not None:
                    self.t = self.t_diag[0]
                

            if ((self.Lambda is not None) and
                (self.H is not None)) : 
                U2 = 0
                U1 = U2 + self.Lambda[1] * self.H(self.t)[1]
                U0 = U1 + self.Lambda[0] * self.H(self.t)[0]
                self.U = np.asarray([U0, U1, U2])
        

    def initialize_global_variables(self,print=False):
        """
        Initialize global variables used in the simulation.

        Initializes spatial grids (x, y, x_grid, y_grid) and wavenumber variables (kx, ky, k0, k1, k2, k8).
        Also, computes additional parameters related to wavenumbers (nkx, nky, kmax, dk, dl, dkr, kr).

        Prints the initialized variables and the case directory.
        """
        
        # Generate x and y grid
        self.x = np.linspace(0, self.Lx, self.nx, endpoint=False)
        self.y = np.linspace(0, self.Ly, self.ny, endpoint=False)

        # Create 2D grid
        self.x_grid, self.y_grid = np.meshgrid(self.x, self.y, indexing='ij')

        # Initialize wavenumbers
        self.kx = np.zeros((self.nx // 2 + 1, self.ny), dtype=np.float64)
        self.ky = np.zeros((self.nx // 2 + 1, self.ny), dtype=np.float64)
        self.k0 = np.zeros((self.nx // 2 + 1, self.ny), dtype=np.float64)
        self.k1 = np.zeros((self.nx // 2 + 1, self.ny), dtype=np.float64)
        self.k2 = np.zeros((self.nx // 2 + 1, self.ny), dtype=np.float64)
        self.k8 = np.zeros((self.nx // 2 + 1, self.ny), dtype=np.float64)

        # Initialize kx
        for i in range(1, self.nx // 2 + 2):
            self.kx[i - 1, :] = (2.0 * np.pi / self.Lx) * np.real(i - 1)

        # Initialize ky
        for j in range(1, self.ny // 2 + 2):
            self.ky[:, j - 1] = (2.0 * np.pi / self.Ly) * np.real(j - 1)

        for j in range(self.ny // 2 + 2, self.ny + 1):
            self.ky[:, j - 1] = (2.0 * np.pi / self.Ly) * np.real(j - self.ny - 1)

        self.k2 = self.kx ** 2 + self.ky ** 2
        self.k0[self.k2 > 0] = 1.0 / self.k2[self.k2 > 0]
        self.k1 = np.sqrt(self.k2)
        self.k8 = self.k2 ** 4

        self.nkx = self.k1.shape[0]
        self.nky = self.k1.shape[1]

        self.kmax = self.k1[:, 0].max()
        self.dk = 2.0 * np.pi / self.Lx
        self.dl = 2.0 * np.pi / self.Ly
        self.dkr = np.sqrt(self.dk ** 2 + self.dl ** 2)
        #self.kr = np.arange(1 * self.dkr / 2., self.kmax + self.dkr, self.dkr)
        self.kr = np.linspace(1 * self.dkr / 2., self.kmax, num=int((self.kmax - 1 * self.dkr / 2.) / self.dkr) + 1)
        self.nkr = len(self.kr)
        
        if print:
            print("Initialized the following variables:")
            print("\tx, y, x_grid, y_grid")
            print("\tkx, ky, k0, k1, k2, k8")
            print("\tnkx, nky, kmax, dk, dl, dkr, kr")
            print("\tcase_dir")
            print("Case Directory: ", self.case_dir)
  
    def read_parameters(self):
        if self.case_dir is None: return
            
        file_path = self.case_dir+"/QG3/parameters.f90"
        selected_parameters = ['Lx', 'Ly', 'nx', 'ny', 'nz', 
                           'f0', 'g', 'rho0', 'A0', 'A2',
                           'A8', 'A20', 'r0', 'C_d',
                            'use_bi_hypo_smag','C_smag','B_smag',
                            'use_bi_leith', 'C_leith','B_leith','L_filter']
        parameters = {}

        with open(file_path, 'r') as file:
            content = file.read()

        # Regular expression pattern to match selected parameter lines
        pattern_parameter = re.compile(r'\s*([\w\d_]+)\s*=\s*([^!,\n]+)')

        matches = pattern_parameter.findall(content)

        for name, value in matches:
            if name in selected_parameters:
                value = value.strip().replace('_dp', '')  # Remove trailing _dp and leading/trailing whitespaces
                try:
                    # Attempt to evaluate the expression
                    parameters[name] = float(sympify(value))
                except (ValueError, TypeError):
                    # If evaluation fails, keep the original value as a string
                    parameters[name] = value

        self.Lx = parameters['Lx']
        self.Ly = parameters['Ly']
        self.nx = int(parameters['nx'])
        self.ny = int(parameters['ny'])
        self.nz = int(parameters['nz'])
        self.f0 = parameters['f0']
        self.g = parameters['g']
        self.rho0 = parameters['rho0']
        self.A0 = parameters['A0']
        self.A2 = parameters['A2']
        self.A8 = parameters['A8']
        self.A20 = parameters['A20']
        self.r0 = parameters['r0']
        self.C_d = parameters['C_d']
        self.use_bi_hypo_smag = True if parameters['use_bi_hypo_smag'] == '.TRUE.' else False
        self.C_smag = parameters.get('C_smag',0)
        self.B_smag = parameters.get('B_smag',0)
        self.use_bi_leith = True if parameters.get('use_bi_leith','.FALSE.') == '.TRUE.' else False
        self.C_leith = parameters.get('C_leith',0)
        self.B_leith = parameters.get('B_leith',0)
        self.L_filter = parameters.get('L_filter',0)

        read_parameters = {'Lx':self.Lx,'Ly':self.Ly,'nx':self.nx,'ny':self.ny,'nz':self.nz,
                           'f0':self.f0,'g':self.g,'rho0':self.rho0,
                           'A0':self.A0,'A2':self.A2,'A8':self.A8,'A20':self.A20,
                           'r0':self.r0,'C_d':self.C_d}
        for name,value in read_parameters.items():
            if not isinstance(value, (int,float)):
                print(f"Parameter '{name}' is not a float. Got type {type(value)}")
    
    def read_file(self, file_path, parameter_patterns):
        parameters = {}
        with open(file_path, 'r') as file:
            for line in file:
                for name, pattern in parameter_patterns.items():
                    match = pattern.search(line)
                    if match:
                        parameters[name] = float(match.group(1))
        return np.asarray(list(parameters.values())), parameters

    def read_MeanFlow(self):
        file_path = self.case_dir + '/QG3/MeanFlow.f90'
        parameter_patterns = {
            'Lambda(1)': re.compile(r'Lambda\(1\) = ([\d.eE+-]+)'),
            'Lambda(2)': re.compile(r'Lambda\(2\) = ([\d.eE+-]+)')
        }
        self.Lambda, read_parameters = self.read_file(file_path, parameter_patterns)
        
        for name, value in read_parameters.items():
            if not isinstance(value, (int, float)):
                print(f"Parameter '{name}' is not a float. Got type {type(value)}")

    def read_Stratification(self):
        file_path = self.case_dir + '/QG3/Stratification.f90'
        parameter_patterns = {
            'sigma(1)': re.compile(r'sigma\(1\) = (\d+(\.\d+)?)'),
            'sigma(2)': re.compile(r'sigma\(2\) = (\d+(\.\d+)?)')
        }
        self.sigma, read_parameters = self.read_file(file_path, parameter_patterns)

        for name, value in read_parameters.items():
            if not isinstance(value, (int, float)):
                print(f"Parameter '{name}' is not a float. Got type {type(value)}")
    
    def print_parameters(self):
        print("Physical Dimensions:")
        print(f"  Lx: {self.Lx/1e3} km")
        print(f"  Ly: {self.Ly/1e3} km")
        print(f"  f0: {self.f0}")
        print(f"  g: {self.g}")
        print(f"  rho0: {self.rho0}")
        print("Grid Points:")
        print(f"  nx: {self.nx}")
        print(f"  ny: {self.ny}")
        print(f"  nz: {self.nz}")
        print("Stratification:")
        print(f"  sigma0: {self.sigma[0]}")
        print(f"  sigma1: {self.sigma[1]}")
        print("Mean Flow:")
        print(f"  Lambda0: {self.Lambda[0]}")
        print(f"  Lambda1: {self.Lambda[1]}")
        print("Viscosity and friction:")
        print(f"  A0: {self.A0}")
        print(f"  A2: {self.A2}")
        print(f"  A8: {self.A8}")
        print(f"  A20: {self.A20}")
        print(f"  r0: {self.r0}")
        print(f"  C_d: {self.C_d}")
        print(f"  use_bi_hypo_smag: {self.use_bi_hypo_smag}")
        print(f"  C_smag: {self.C_smag}")
        print(f"  B_smag: {self.B_smag}")
        print(f"  use_bi_leith: {self.use_bi_leith}")
        print(f"  C_leith: {self.C_leith}")
        print(f"  B_leith: {self.B_leith}")
        print(f"  L_filter: {self.L_filter/1e3} km")

    
    ###################################################################
    ### Loading data ##################################################
    ################################################################### 

    def load_data(self,quantity,first_n=None,last_n=None,skip=None,number=None,
                  desc='Processing files',dtype=np.float64):
        """
        Load and stack time-dependent simulation data for a given quantity ('q', 'p', or 't').

        Parameters:
        - quantity (str): The type of quantity to load ('q', 'p', or 't').
        - first_n (int or None): The number of initial files to load. If None, all files are considered.
        - last_n (int or None): The number of final files to load. If None, all files are considered.
        - dtype: NumPy data type to use when reading binary files (default is np.float64).

        Returns:
        - result: NumPy array
            The stacked and reshaped array representing the loaded quantity.
            For 'q' and 'p', the array has dimensions (nx, ny, nz, nt).
            For 't', the array has shape (nt,) representing a time series.
        """
        if quantity not in ('q', 'p', 't', 'E', 'KE', 'total_KE', 't_diag',
                            'Dh', 'Edot_adv', 'Edot_leith', 'Edot_smag', 'Edot_source', 'Edot_strat','Edot_APE',
                            'Rh', 'Sh', 'Th','Ch','Gh'):
            print("quantity must be one of 'q', 'p', 'E', 'KE', 'total_KE', 't', 't_diag', "
                  "'Dh', 'Edot_adv', 'Edot_leith', 'Edot_smag', 'Edot_source', 'Edot_strat','Edot_APE', "
                  "'Rh', 'Sh', 'Th', 'Ch'")
            return
        
        if (quantity == 't' or 
            quantity == 't_diag' or 
            quantity == 'E' or 
            quantity == 'total_KE' or
            quantity.startswith('Edot_')):
            dims = (1)
        elif quantity == 'KE':
            dims = (self.nkr, self.nz)
        elif quantity.endswith('h'):
            dims = (self.nkr,)
        else:
            dims = self.dims


        # load a range of files
        if number is None:
            file_pattern = quantity+'.*.dat'
    
            # Get a sorted list of file names
            path = os.path.join(self.data,file_pattern)
            file_list = sorted(glob.glob(path), key=lambda x: int(x.split('.')[1]))
            if first_n is not None:
                file_list = file_list[:first_n]
            if last_n is not None:
                file_list = file_list[-last_n:]
            if skip is not None:
                file_list = file_list[::skip]
    
            # Initialize an empty list to store the reshaped arrays
            reshaped_arrays = []
    
            # Iterate over the files and read/reshape each array
            for file_name in tqdm(file_list[1:], desc=desc, unit="file"):
                data = np.fromfile(file_name, dtype=dtype)  # Adjust dtype if needed
                reshaped_data = data.reshape(dims, order='F')  # Use 'F' for Fortran order
                reshaped_arrays.append(reshaped_data)
    
            # Stack the reshaped arrays along a new axis to create a 4D array
            result = np.stack(reshaped_arrays, axis=len(dims))
    
            # Now 'result' contains the stacked and reshaped array with shape [nx, ny, nz, nt]
            if len(file_list)!=result.shape[-1]:
                print("Warning: number of files does not match time dimension")
            if quantity == 't' and len(result)>0: result = result[0]
        # load a single file
        else:
            file_name = quantity+'.'+number+'.dat'
            file_path = os.path.join(self.data,file_name)
            data = np.fromfile(file_path, dtype=dtype)  
            result = data.reshape(dims, order='F')  # Use 'F' for Fortran order        
        return result
    
    def load_Q(self,**kwargs):
        """
        Load potential vorticity (Q) data using the load_data method for quantity 'q'.
        
        Parameters:
        - kwargs: Additional keyword arguments to pass to the load_data method.

        Output:
        - self.Q: NumPy array
            The resulting 4D array representing potential vorticity (Q).
            Dimensions are (nt, nx, ny, nz) following a common convention for array dimensions.
        """
        self.Q = self.load_data('q',desc='Loading potential vorticity',**kwargs)
        if len(self.Q.shape)==4:
            self.Q = np.transpose(self.Q, (3, 0, 1, 2))
    
    def load_P(self,**kwargs):
        """
        Load potential vorticity (Q) data using the load_data method for quantity 'q'.
        
        Parameters:
        - kwargs: Additional keyword arguments to pass to the load_data method.

        Output:
        - self.P: NumPy array
            The resulting 4D array representing streamfunction (P).
            Dimensions are (nt, nx, ny, nz) following a common convention for array dimensions.
        """
        self.P = self.load_data('p',desc='Loading streamfunction',**kwargs)
        if len(self.P.shape)==4:
            self.P = np.transpose(self.P, (3, 0, 1, 2))
    
    def load_T(self,**kwargs):
        """
        Load time (T) data using the load_data method for quantity 't'.
        
        Parameters:
        - kwargs: Additional keyword arguments to pass to the load_data method.
        
        Output:
        - self.T: NumPy array
            The resulting array represents time values corresponding to each time step in the simulation.
        """
        self.t = self.load_data('t',desc='Loading time',**kwargs)[0]

    def set_t(self,t):
        self.t = t
        
        if (self.Lambda is not None) and (self.H is not None):
            U2 = 0
            U1 = U2 + self.Lambda[1] * self.H(self.t)[1]
            U0 = U1 + self.Lambda[0] * self.H(self.t)[0]
            self.U = np.asarray([U0, U1, U2])

    

    ###################################################################
    ### Computing Operators ############################################
    ################################################################### 

    def create_m_matrix(self,K=None,t=None,return_m=False,H=None):
        """
        Create the m matrix used in the inversion process. The m matrix is used in the inversion process
        in the multi streamfunction formulation. The streamfunction P_i at surface i is realated to the surface
        potential vorticity Q_i through Qi=-m[i]*P_i.

        Parameters:
        - K (array): Array of wavenumbers. The shape can be either (nkx,) for a 1D array or
        (nkx, nky) for a 2D array, where nkx is the number of zonal wavenumbers, and nky is the
        number of meridional wavenumbers.
        - H (nz array): Array of layer thicknesses.

        Returns:
        - m (array): The m matrix for the inversion process. The shape is determined by the shape
        of the input wavenumber array K. For a 1D K, m has shape (n_kx, 3), and for a 2D K, m has shape
        (nkx, nky, 3).

        The function handles different shapes of the wavenumber array K to allow flexibility in the
        application. If K is a 1D array, it is assumed to represent the zonal wavenumbers only. If K is
        a 2D array, it is assumed to represent both zonal and meridional wavenumbers.
        """
        if H is not None:
            H = H
        elif t is None:
            H = self.H(self.t)
        else:
            H = self.H(t)

        if K is None:
            K = self.k1
        
        sigma = self.sigma
        # Initialize arrays
        T0 = np.tanh(sigma[0] * H[0] * K)
        T1 = np.tanh(sigma[1] * H[1] * K)

        if len(K.shape)==1:
            m = np.zeros((len(K),self.nz),dtype=np.float64)
            #T0[0] = 1.0 # Avoids divide by zero
            #T1[0] = 1.0 # Avoids divide by zero
        elif len(K.shape)==2:
            m = np.zeros((self.nkx,self.nky,self.nz), dtype=np.float64)
            #T0[0, 0] = 1.0  # Avoids divide by zero
            #T1[0, 0] = 1.0  # Avoids divide by zero
        else: 
            print("K has the wrong shape! Must be 1D or 2D")
            return
        # Compute m values
        m[..., 0] = (K / sigma[0]) * (sigma[1] * T0 + sigma[0] * T1) / (sigma[1] + sigma[0] * T0 * T1)
        m[..., 1] = K * ((T0 / sigma[0]) + (T1 / sigma[1]))
        m[..., 2] = K * (sigma[1] * T0 + sigma[0] * T1) / (sigma[1] * (sigma[0] + sigma[1] * T0 * T1))

        # Avoid divide by zero for m(1, 1, :)
        if len(K.shape)==2:
            m[0,0,:] = 1.0

        if return_m is True:
            return m
        else:
            self.m = m

    ## Compute the inversion function at each time step.
    def create_all_m(self,K=None):
        if K is None:
            K = self.k1

        if len(K.shape)==1:
            self.m = np.zeros((self.nt,len(K),self.nz))
        elif len(K.shape)==2:
            self.m = np.zeros((self.nt,self.nkx,self.nky,self.nz))
        
        for i in tqdm(range(self.nt),desc='Calculating inversion functions'):
            t = self.T[i]
            self.m[i] = self.create_m_matrix(K,self.H(t))
            
    def create_e_matrix(self,K=None,t=None,H=None,return_e=False):
        """
        Create the e matrix, which is 3x3 square matrix and depends on wavenumber. In the multi-
        streamfunction formulation, the e matrix represents the strength of the coupling between
        each layer at different wavenumbers. e[j,l] is represents the efficiency that the 
        streamfunction from the l-th layer contributes to layer j: 
        - e[j,l]=1 means the streamfunction at layer l is barotropic and so its contributions at 
        layer l and layer j are equal. 
        - e[j,l]<<1 means that the streamfunction at layer l only contributes to layer l and not
        the other layers.

        Parameters:
        - K (array): Array of wavenumbers. The shape can be either (n_kx,) for a 1D array or
        (nkx, nky) for a 2D array, where nkx is the number of zonal wavenumbers, and nky is the
        number of meridional wavenumbers.
        - H (array): Array of layer thicknesses. The shape should be (3,) representing the three
        layers: surface layer, mixed layer, and thermocline.

        Returns:
        - e (array): The e matrix for the inversion process. The shape is determined by the shape
        of the input wavenumber array K. For a 1D K, e has shape (n_kx, 3, 3), and for a 2D K, e has
        shape (nkx, nky, 3, 3).
        """
        sigma = self.sigma
        
        if H is not None:
            H = H
        elif t is None:
            H = self.H(self.t)
        else:
            H = self.H(t)

        if K is None:
            K = self.k1
        
        # Initialize arrays
        C0 = np.cosh(sigma[0] * H[0] * K)
        C1 = np.cosh(sigma[1] * H[1] * K)
        T0 = np.tanh(sigma[0] * H[0] * K)
        T1 = np.tanh(sigma[1] * H[1] * K)

        # Initialize e array
        if len(K.shape)==1:
            e = np.zeros((len(K), self.nz, self.nz), dtype=np.float64)
            #T0[0] = 1.0 # Avoids divide by zero
            #T1[0] = 1.0  # Avoids divide by zero
        if len(K.shape)==2:
            e = np.zeros((self.nkx, self.nky, self.nz, self.nz), dtype=np.float64)
            T0[0, 0] = 1.0  # Avoids divide by zero
            T1[0, 0] = 1.0  # Avoids divide by zero

        # Assign values to e
        e[..., 0, 0] = 1.0
        e[..., 1, 1] = 1.0
        e[..., 2, 2] = 1.0
        e[..., 1, 0] = sigma[1] / (C0 * (sigma[1] + sigma[0] * T0 * T1))
        e[..., 2, 0] = e[..., 1, 0] / C1
        e[..., 0, 1] = 1.0 / C0
        e[..., 2, 1] = 1.0 / C1
        e[..., 1, 2] = sigma[0] / (C1 * (sigma[0] + sigma[1] * T0 * T1))
        e[..., 0, 2] = e[..., 1, 2] / C0

        if return_e is True:
            return e
        else:
            self.e = e    
    ## Compute the e matrix at each time step.
    def create_all_e(self,K=None):
        if K is None:
            K = self.k1

        if len(K.shape)==1:
            self.e =  np.zeros((self.nt, len(K), self.nz, self.nz), dtype=np.float64)
        elif len(K.shape)==2:
            self.e = np.zeros((self.nt,self.nkx, self.nky, self.nz, self.nz), dtype=np.float64)
        
        for i in tqdm(range(self.nt),desc='Calculating e matrices'):
            t = self.T[i]
            self.e[i] = self.create_e_matrix(K,self.H(t))

    def create_L_matrix_alt(self, K, H):
        """
        Create the L matrix used in the inversion process in the single streamfunction formulation.
        To relate the single streamfunction formulation with the multistreamfunction formulation, we 
        have L = -M@e^{-1} where @ is matrix multiplication.

        Parameters:
        - K (array): Array of wavenumbers. The shape can be either (nkx,) for a 1D array or
        (nkx, nky) for a 2D array, where nkx is the number of zonal wavenumbers, and nky is the
        number of meridional wavenumbers.
        - H (array): Array of layer thicknesses. The shape should be (3,) representing the three
        layers: surface layer, mixed layer, and thermocline.

        Returns:
        - L (numpy.ndarray): The L matrix for the inversion process. The shape is determined by the shape
        of the input wavenumber array K. For a 1D K, L has shape (n_kx, 3, 3), and for a 2D K, L has
        shape (n_kx, n_ky, 3, 3).

        The L matrix is used in the Quasi-Geostrophic (QG) inversion process to relate the potential
        vorticity (Q) to the streamfunction (P). The inversion equation is given by: Q = L@P where L 
        is the inversion matrix, P is the streamfunction, and Q is the potential vorticity.

        The function handles different shapes of the wavenumber array K to allow flexibility in the
        application. If K is a 1D array, it is assumed to represent the zonal wavenumbers only. If K is
        a 2D array, it is assumed to represent both zonal and meridional wavenumbers.
        """
        sigma = self.sigma
        
        # Initialize arrays
        S0 = np.sinh(sigma[0] * H[0] * K)
        #S0[0] = 1

        S1 = np.sinh(sigma[1] * H[1] * K)
        #S1[0] = 1

        T0 = np.tanh(sigma[0] * H[0] * K)
        #T0[0] = 1.0  # Avoids divide by zero

        T1 = np.tanh(sigma[1] * H[1] * K)
        #T1[0] = 1.0  # Avoids divide by zero
        
        # Initialize e array
        if len(K.shape)==1:
            L = np.zeros((len(K), self.nz, self.nz), dtype=np.float64)
            if K[0]==0:
                S0[0] = 1.0  # Avoids divide by zero
                S1[0] = 1.0  # Avoids divide by zero
                T0[0] = 1.0  # Avoids divide by zero
                T1[0] = 1.0  # Avoids divide by zero
        if len(K.shape)==2:
            L = np.zeros((self.nkx, self.nky, self.nz, self.nz), dtype=np.float64)
            S0[0, 0] = 1.0  # Avoids divide by zero
            S1[0, 0] = 1.0  # Avoids divide by zero
            T0[0, 0] = 1.0  # Avoids divide by zero
            T1[0, 0] = 1.0  # Avoids divide by zero
        
        # construct matrix
        L[...,0,0] = -(K/sigma[0])/T0
        L[...,0,1] = (K/sigma[0])/S0
        L[...,0,2] = 0
        L[...,1,0] = (K/sigma[0])/S0
        L[...,1,1] = -(K/sigma[0])/T0 - (K/sigma[1])/T1
        L[...,1,2] = (K/sigma[1])/S1
        L[...,2,0] = 0
        L[...,2,1] = (K/sigma[1])/S1
        L[...,2,2] = -(K/sigma[1])/T1

        self.L = L
        return L

    ## Compute the L matrix at each time step.
    def create_all_L_alt(self,K=None):
        if K is None:
            K = self.k1

        if len(K.shape)==1:
            L =  np.zeros((self.nt, len(K), self.nz, self.nz), dtype=np.float64)
        elif len(K.shape)==2:
            L = np.zeros((self.nt,self.nkx, self.nky, self.nz, self.nz), dtype=np.float64)
        
        for i in tqdm(range(self.nt),desc='Calculating L matrices'):
            t = self.T[i]
            L[i] = self.create_L_matrix(K,self.H(t))
        
        return L

    # For testing reasons only
    def create_all_L(self):
        if len(self.m.shape)==4:
            self.L = np.zeros((self.nt,self.nkx,self.nky,self.nz,self.nz))
            for t in tqdm(range(self.nt),desc='Computing L'):
                for i in range(self.nkx):
                    for j in range(self.nky):
                        M_matrix = np.diag(self.m[t,i,j])
                        inv_e = np.linalg.inv(self.e[t,i,j])
                        self.L[t,i,j] = -M_matrix@inv_e
        elif len(self.m.shape)==3:
            nk = self.m.shape[1]
            self.L = np.zeros((self.nt,nk,self.nz,self.nz))
            for t in tqdm(range(self.nt),desc='Computing L'):
                for i in range(nk):
                    M_matrix = np.diag(self.m[t,i])
                    inv_e = np.linalg.inv(self.e[t,i])
                    self.L[t,i] = -M_matrix@inv_e

    #
    def create_L_matrix(self,return_L=False):
        if len(self.m.shape)==3:
            self.L = np.zeros((self.nkx,self.nky,self.nz,self.nz))
            for i in range(self.nkx):
                for j in range(self.nky):
                    M_matrix = np.diag(self.m[i,j])
                    inv_e = np.linalg.inv(self.e[i,j])
                    self.L[i,j] = -M_matrix@inv_e
        elif len(self.m.shape)==2:
            nk = self.m.shape[0]
            self.L = np.zeros((nk,self.nz,self.nz))
            for i in range(nk):
                M_matrix = np.diag(self.m[i])
                inv_e = np.linalg.inv(self.e[i])
                self.L[i] = -M_matrix@inv_e
        if return_L:
            return L

    
    def consistency_check(self,K=None,tol=0.001):
        if self.m is None or self.e is None or self.L is None:
            print("Compute m, e, and L first!")
            return
       
        L_alt =self.create_all_L_alt(K=K)
       
        are_equal = True
        for t in tqdm(range(self.nt),'Comparing L and L_alt'):
            if len(self.L.shape)==5:
                are_equal = np.allclose(self.L[t,1:,1:], L_alt[t,1:,1:], atol=tol)
            elif len(self.L.shape)==4:
                are_equal = np.allclose(self.L[t,1:], L_alt[t,1:], atol=tol)

        if are_equal:
            print("Matrices L and L_alt are approximately equal.")
        else:
            print("Matrices L and L_alt are NOT approximately equal.")
    
    ###################################################################
    ### Calculations ######################################################
    ################################################################### 
            
    def calculate_energy(self):
        """
        Calculate the QG total energy.

        The total energy is computed using the potential vorticity (Q) and streamfunction (P) fields.
        The integral is performed over the entire domain in both the zonal (x) and meridional (y) directions.

        The calculated energy is stored in the class attribute self.E.
        """
        #if (self.Q is None) or (self.P is None):
        #    print("Load Q and P data first. Not computing energy.")
        #    return
        #Eden = -0.5*self.Q*self.P/(self.Lx*self.Ly)
        #intx_Eden = np.trapz(Eden,x=self.x,axis=0)
        #inty_intx_Eden = np.trapz(intx_Eden,x=self.y,axis=0)
        #self.E = inty_intx_Eden.sum(axis=0)
        if (self.Qh is None) or (self.Ph is None):
            print("Calculate Qh and Ph first. Not computing energy.")
            return
        E_series = -0.5*np.real(np.conjugate(self.Ph)*self.Qh)
        #E_series[1:-1, :, :] *= 2 # (Because this is an rfft)
        self.E =  sum_series(E_series).sum()/(self.nx*self.ny)**2


    def fft(self,field):
        if len(field.shape)==3:
            return npfft.rfftn(field,axes=(-2,-3))
        elif len(field.shape)==2:
            return npfft.rfftn(field,axes=(-1,-2))
        else:
            print("field must be 2D or 3D")

    def ifft(self,field):
        if len(field.shape)==3:
            return npfft.irfftn(field,axes=(-2,-3))
        elif len(field.shape)==2:
            return npfft.irfftn(field,axes=(-1,-2))

    def compute_Qh(self):
        self.Qh = self.fft(self.Q)

    def compute_Ph(self):
        self.Ph = self.fft(self.P)

    def compute_Phih(self):
        if (self.m is None):
            print("Compute m first!")
            return
        if len(self.m.shape) != 3:
            print("First recompute operators with a 2D wavenumber field.")
            return
        if (self.Qh is None):
            print("Compute FFT of Q first!")
            return

        Phih = -self.Qh/self.m
        return Phih

    # phi_hat are the streamfunctions for each layer
    def compute_Phi(self):
        if (self.m is None):
            print("Compute m first!")
            return
        if len(self.m.shape) != 3:
            print("First recompute operators with a 2D wavenumber field.")
            return
        if (self.Qh is None):
            print("Compute FFT of Q first!")
            return

        Phih = -self.Qh/self.m
        self.Phi = self.ifft(Phih)

    # For testing only. Reconstruct Psi from Phi.
    def compute_P_from_phi(self):
        Phih = -self.Qh/self.m
        Ph = np.zeros_like(Phih,dtype=np.complex128)
        for i in range(self.nz):
            for ii in range(self.nz):
                Ph[:,:,i] += self.e[:,:,i,ii]*Phih[:,:,ii]
        return self.ifft(Ph)

    def invert_matrix(self,A):
        inv_A = np.zeros_like(A)
        for i in range(A.shape[0]):
            try:
                inv_A[i] = np.linalg.inv(A[i])
            except np.linalg.LinAlgError:
                print(f"Matrix at index {i} is not invertible. Setting to substitute value.")
                inv_A[i] = np.full_like(A[i], 0.)
        return inv_A
    
    # For testing only. Reconstruct Psi using L.
    def compute_P_from_Q(self):
        isL = self.L is None
        isQh = self.Qh is None
        if isL or isQh:
            print("Compute L and Q first.")
            return
        
        inv_L = self.invert_matrix(self.L)

        PLh = np.zeros_like(self.Qh)
        for i in range(self.nkx):
            for j in range(self.nky):
                PLh[i,j] = inv_L[i,j]@self.Qh[i,j]
        return self.ifft(PLh)

    def compute_Q_from_P(self):
        isL = self.L is  None
        isPh = self.Ph is  None
        if isL or isPh:
            print("Compute L and P first.")
            return

        QLh = np.zeros_like(self.Ph)
        for i in range(self.nkx):
            for j in range(self.nky):
                QLh[i,j] = self.L[i,j]@self.Ph[i,j]
        return self.ifft(QLh)

    def compute_U(self):
        if self.Ph is None:
            print("Compute Ph first.")
            return
        Uh = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        for z in range(self.nz):
            Uh[:,:,z] =  -1j*self.ky*self.Ph[:,:,z]
        return self.ifft(Uh)

    def compute_V(self):
        if self.Ph is None:
            print("Compute Ph first.")
            return
        Vh = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        for z in range(self.nz):
            Vh[:,:,z] =  1j*self.kx*self.Ph[:,:,z]
        return self.ifft(Vh)

    def compute_speed(self,U=None,V=None):
        if self.Ph is None:
            print("Compute Ph first.")
            return
        if U is None or V is None:
            U = self.compute_U()
            V = self.compute_V()
        self.speed = np.sqrt(U**2+V**2)
    
    def compute_velocity(self,U=None,V=None):
        if self.Ph is None:
            print("Compute Ph first.")
            return
        if U is None or V is None:
            U = self.compute_U()
            V = self.compute_V()
        return U,V

    # Computes the isotropic spectrum of fh 
    # fh must be a 2D array
    def isotropic_spectrum(self,fh):
        fhr = np.zeros(self.nkr,dtype=np.complex128)
        dkr= self.kr[1]-self.kr[0]
    
        for i in range(self.nkr):
            k1_mask =  (self.k1>=self.kr[i]-dkr/2) & (self.k1<=self.kr[i]+dkr/2)
            dtheta = np.pi / (k1_mask.sum()-1)
            fhr[i] = fh[k1_mask].sum() * self.kr[i] * dtheta
        return fhr


    ### Instability calculations

    def create_instability_matrix(self,K=None):
        if K is None:
            K = self.k1

        Lambda = self.Lambda
        sigma = self.sigma
        U = self.U
        Qy = [Lambda[0]/sigma[0]**2,Lambda[1]/sigma[1]**2-Lambda[0]/sigma[0]**2,-Lambda[1]/sigma[1]**2]
        
        if len(K.shape)==1:
            inv_L = self.invert_matrix(self.L)
            A =  np.zeros((len(K), self.nz, self.nz), dtype=np.float64)
            for i in tqdm(range(len(K)),desc='Computing instability matrix'):
                U_matrix = np.diag(U)
                Qy_matrix = np.diag(Qy)
                A[i,:,:] = U_matrix + np.dot(Qy_matrix,inv_L[i])
        elif len(K.shape)==2:
            A = np.zeros((self.nkx, self.nky, self.nz, self.nz), dtype=np.float64)
            for i in tqdm(range(self.nkx),desc='Computing instability matrix'):
                inv_L = self.invert_matrix(self.L[i])
                for j in range(self.nky):
                    U_matrix = np.diag(U)
                    Qy_matrix = np.diag(Qy)
                    A[i,j,:,:] = U_matrix + np.dot(Qy_matrix,inv_L[j])
        else:
            print("K must either one or two dimensional!")
            return None
        return A

    def solve_instability(self,K=None):
        if K is None:
            K = self.k1
        A = self.create_instability_matrix(K=K)
        evalues, evectors = np.linalg.eig(A)

        unstable = np.any(np.abs(np.imag(evalues)) > 1e-10)
        print("Unstable?: ", unstable)
        self.evalues = evalues 
        self.evectors = evectors
        
        self.growth_rate = K[...,np.newaxis]*evalues.imag*3600*24 # in units of day^{-1}

    def max_growth_rate(self,K=None):
        if K is None:
            K = self.k1
        
        max_index = np.argmax(self.growth_rate)
    
        k_max_ind, branch = np.unravel_index(max_index, self.growth_rate.shape)
    
        k_max = K[k_max_ind]
        max_growth =  self.growth_rate[k_max_ind,branch]
    
        ## Find largest non-zero growth rate
    
        growth_maxb = np.max(self.growth_rate,axis=1)
        non_zero_indices = np.nonzero(growth_maxb)[0]
        k_nonzero_index = non_zero_indices[-1]
        k_nonzero = K[k_nonzero_index]
    
        return max_growth, k_max, k_nonzero

    #### Compute modes  ## 

    # Compute the eigenvalues and eigenvectors of -L for each time.
    # The eigenvalue problem has the form -L@e=lambda*e where e 
    # is the eigenvector.
    # -L is the energy matrix in the total streamfunction coordinates
    # so that -(1/2)<psi,L*psi> gives the energy.
    def compute_modes(self, K=None, return_modes = False):
        if self.L is None:
            print("Calcualte L first.")
            return
        if K is None:
            K = self.k1
            matrix = np.zeros((self.nkx, self.nky, self.nz, self.nz))
            self.evalues = np.zeros((self.nkx, self.nky, self.nz))
            self.evectors = np.zeros((self.nkx, self.nky, self.nz, self.nz))
        else:    
            matrix = np.zeros((len(K), self.nz, self.nz))
            self.evalues = np.zeros((len(K), self.nz))
            self.evectors = np.zeros((len(K), self.nz, self.nz))
        
        eigenvalues, eigenvectors = np.linalg.eig(-self.L)
        sorted_indices = np.argsort(eigenvalues)  # Get indices to sort eigenvalues

        if len(K.shape)==1:
            # broadcasting magic... can also be done analogously to the len(K.shape)==2 case.
            self.evalues[:] = eigenvalues[np.arange(len(eigenvalues))[:, None], sorted_indices] 
            for j in range(len(K)):
                # Rearrange eigenvectors accordingly
                self.evectors[j,:,:] = eigenvectors[j][:, sorted_indices[j]]  
        elif len(K.shape)==2:
            for i in range(self.nkx):
                for j in range(self.nky):
                    self.evalues[i,j,:] = eigenvalues[i,j][sorted_indices[i,j]]
                    self.evectors[i,j,:,:] = eigenvectors[i,j][:,sorted_indices[i,j]]
        if return_modes:
            return self.evalues, self.evectors
        
    # These three functions define the values of the coefficents
    # a,b for the vertical structure in the case of a surface
    # anomaly (Psi0), a mixed-layer anomaly (Psi1), and 
    # pycnocline anomaly (Psi2).
    # They also return the argument of sinh and cosh.
    def Psi0_nz3(self,z,k,t=None,H=None):
        if t is not None:
            H_arr = self.H(t)
        if H is not None:
            H_arr = H
        if t is None and H is None:
            return
        # Define vertical coordinate of the surfaces
        z0 = 0
        z1 = -H_arr[0]
        z2 = -H_arr[0] - H_arr[1]
        
        sigma = self.sigma
        C0 = np.cosh(k*sigma[0]*(z0-z1))
        S0 = np.sinh(k*sigma[0]*(z0-z1))
        C1 = np.cosh(k*sigma[1]*(z1-z2))
        S1 = np.sinh(k*sigma[1]*(z1-z2))
        Cth0 = C0/S0
        Cth1 = C1/S1 
        T0 = S0/C0
        T1 = S1/C1
        
        if z0 >= z > z1: #l=0
            a = 1
            b = (sigma[1]*C1 + sigma[0]*Cth0*S1)/(sigma[1]*C1*Cth0 + sigma[0]*S1)
            arg = sigma[0]*k*(z-z0)
        else: #z1 >= z > z2, l=1
            a = sigma[1]/(sigma[1]*C0 + sigma[0]*S0*T1)
            b = sigma[1]/(sigma[1]*C0*Cth1 + sigma[0]*S0)
            arg = sigma[1]*k*(z-z1)
        return a,b,arg

    def Psi1_nz3(self,z,k,t=None,H=None):
        if t is not None:
            H_arr = self.H(t)
        if H is not None:
            H_arr = H
        if t is None and H is None:
            return
            
        # Define vertical coordinate of the surfaces
        z0 = 0
        z1 = -H_arr[0]
        z2 = -H_arr[0] - H_arr[1]
        
        sigma = self.sigma
        C0 = np.cosh(k*sigma[0]*(z0-z1))
        S0 = np.sinh(k*sigma[0]*(z0-z1))
        C1 = np.cosh(k*sigma[1]*(z1-z2))
        S1 = np.sinh(k*sigma[1]*(z1-z2))
        Cth0 = C0/S0
        Cth1 = C1/S1 
        T0 = S0/C0
        T1 = S1/C1
        
        if z0 >= z > z1: #l0
            a = 1/C0
            b = 0
            arg = sigma[0]*k*(z-z0)
        else: #z1 >= z > z2, l=1
            a = 1
            b = T1
            arg = sigma[1]*k*(z-z1)
        return a,b,arg

    def Psi2_nz3(self,z,k,t=None,H=None):
        if t is not None:
            H_arr = self.H(t)
        if H is not None:
            H_arr = H
        if t is None and H is None:
            return
        # Define vertical coordinate of the surfaces
        z0 = 0
        z1 = -H_arr[0]
        z2 = -H_arr[0] - H_arr[1]
        
        sigma = self.sigma
        C0 = np.cosh(k*sigma[0]*(z0-z1))
        S0 = np.sinh(k*sigma[0]*(z0-z1))
        C1 = np.cosh(k*sigma[1]*(z1-z2))
        S1 = np.sinh(k*sigma[1]*(z1-z2))
        Cth0 = C0/S0
        Cth1 = C1/S1 
        T0 = S0/C0
        T1 = S1/C1
        
        if z0 >= z > z1: #l0
            a = sigma[0]/(sigma[0]*C0*C1 + sigma[1]*S0*S1)
            b = 0
            arg = sigma[0]*k*(z-z0)
        else: #z1 >= z > z2, l=1
            a = sigma[0]/(sigma[0]*C1 + sigma[1]*S1*T0)
            b = - sigma[1]*(1/S1)/(sigma[1] + sigma[0]*Cth0*Cth1)
            arg = sigma[1]*k*(z-z1)
        return a,b,arg

    
    # The vertical structure induced a buoyancy anomaly theta_n at the nth
    # surface from the surface ocean, with the surface ocean being n=0 and
    # the ocean bottom being n=N+1, and N the number of interior surfaces.
    # At a given wavenumber k, the vertical structure for the jth surface
    # has the form
    # Psi_{jl}(z) = a_{jl}*cosh(sigma_l*k(z-z_l)) + b_{jl}*sinh(sigma_l*k(z-z_l))
    # for z_l > z > z_{l+1} where l is the layer index (l=0,...,N).
    # z is the vertical coordinate value
    # t is the index of the time axis
    # k is the horizontal wavenumber
    # n is the index of the surface
    def Psi(self,z,k,n,t=None,H=None):
        if t is not None:
            H_arr = self.H(t)
        if H is not None:
            H_arr = H
        if t is None and H is None:
            print("One of the parameters t or H should not be None")
            return
        # Define vertical coordinate of the surfaces
        z0 = 0
        z1 = -H_arr[0]
        z2 = -H_arr[0] - H_arr[1]
        
        if np.isscalar(k):
            k = np.asarray([k])

        mask = k == 0
        
        # Initialize the result array with the same shape as k
        result = np.zeros(k.shape)

        # For elements in k that are equal to 0, set the corresponding result to 1
        result[mask] = 1.

        
        if self.nz==3:
            if n==0:
                a,b,arg = self.Psi0_nz3(z,k[~mask],t=t,H=H)
            if n==1:
                a,b,arg = self.Psi1_nz3(z,k[~mask],t=t,H=H)
            if n==2:
                a,b,arg = self.Psi2_nz3(z,k[~mask],t=t,H=H)
        else:
            print("This function requires nz==3!")
            return

        result[~mask] = a*np.cosh(arg) + b*np.sinh(arg)

        return result

    # order is the number of derivatives to take.
    # order must be 1 or 2.
    def dPsidz(self,z,k,n,t=None,H=None,order=1):
        if t is not None:
            H_arr = H(t)
        if H is not None:
            H_arr = H
        if t is None and H is None:
            print("One of the parameters t or H should not be None")
            return

        if np.isscalar(k):
            k = np.asarray([k])
        
        mask = k == 0
        # Initialize the result array with the same shape as k
        result = np.zeros(k.shape)
        result[mask] = 0.
        
        # Define vertical coordinate of the surfaces
        z0 = 0
        z1 = -H_arr[0]
        z2 = -H_arr[0] - H_arr[1]
        
        if self.nz==3:
            if n==0:
                a,b,arg = self.Psi0_nz3(z,k[~mask],t=t,H=H)
            if n==1:
                a,b,arg = self.Psi1_nz3(z,k[~mask],t=t,H=H)
            if n==2:
                a,b,arg = self.Psi2_nz3(z,k[~mask],t=t,H=H)
        else:
            print("This function requires nz==3!")
            return

        if z0 >= z > z1: #l0
            sigma = self.sigma[0]
        else: #z1 >= z > z2, l=1
            sigma = self.sigma[1]
        
        if order == 1:
            result[~mask] = k[~mask]*sigma*(b*np.cosh(arg) + a*np.sinh(arg))
            return result
        elif order == 2:
            result[~mask] = k[~mask]**2*sigma**2*(a*np.cosh(arg) + b*np.sinh(arg))
            return result
        else:
            print("order must be 1 or 2.")
            return None

    # t is the index of the time axis
    # k is the wavenumber
    # evector[:,i] is the eigenvector corresponding to the eigenvalue evalues[i]
    def Psi_modes(self,z,t,ik,k,n):
        if self.evectors is None:
            print("Compute modes first!")
            return

        if len(self.evectors.shape)==3: # 1D wavenumber axis
            evector = self.evectors[ik,:,n]
        elif len(self.evectors.shape)==4: # 2D wavenumber axis
            evector = self.evectors[ik,0,:,n]
        else:
            print("Wrong evector size")
            return
        
        inv_e = np.linalg.inv(self.e[ik])
        phi = inv_e@evector
    
        return np.asarray([phi[i]*self.Psi(z,t,k,i) for i in range(self.nz)]).sum()

    def project_Ph(self):
        if self.evectors is None:
            print("Compute modes first")
            return
        self.Pmh = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        for i in range(self.nkx):
            for j in range(self.nky):
                for n in range(self.nz):
                    self.Pmh[i,j,n] = self.evectors[i,j,:,n]@self.Ph[i,j,:]

    def compute_Pm(self):
        if self.Pmh is None:
            print("Compute Pmh first.")
            return
        self.Pm = self.ifft(self.Pmh)

    def compute_Qm(self):
        if self.evalues is None:
            print("Compute modes first")
            return
        if self.Pmh is None:
            print("Compute Pmh first.")
            return
        self.Qm = self.ifft(self.evalues*self.Pmh)

    def compute_KE_spectrum(self,z=0):
        if self.Ph is None:
            self.compute_Ph()
        KE_spec = np.abs(self.k1*self.Ph[:,:,z])**2/(self.nx*self.ny)**2
        KEr = self.isotropic_spectrum(KE_spec)
        return KEr

    def volume_integrated_KE(self,z_upper=None,z_lower=None,Nz=100):
        if z_upper is None:
            z_upper = 0
        if z_lower is None:
            z_lower = -self.H(self.t).sum()
        
        # First compute the coefficients, which consist
        # of integrals of the form Int(Psi_i*Psi_j*dz)
        Z = np.linspace(z_lower, z_upper, Nz)
    
        Psi0 = np.asarray([self.Psi(z, self.k1, 0, t=self.t) for z in Z ], dtype=np.float64)
        Psi1 = np.asarray([self.Psi(z, self.k1, 1, t=self.t) for z in Z ], dtype=np.float64)
        Psi2 = np.asarray([self.Psi(z, self.k1, 2, t=self.t) for z in Z ], dtype=np.float64)
    
        PsiPsidz_00 = simps(Psi0*Psi0,Z,axis=0)
        PsiPsidz_01 = simps(Psi0*Psi1,Z,axis=0)
        PsiPsidz_02 = simps(Psi0*Psi2,Z,axis=0)
        PsiPsidz_11 = simps(Psi1*Psi1,Z,axis=0)
        PsiPsidz_12 = simps(Psi1*Psi2,Z,axis=0)
        PsiPsidz_22 = simps(Psi2*Psi2,Z,axis=0)
    
        # Compute Phih = thetah/m
        if self.m is None: self.create_m_matrix()
        if self.Qh is None: self.compute_Qh()
        Phih = self.compute_Phih()
    
        # interaction terms
        inter00 = (1./2) * Phih[:,:,0] * np.conj(Phih[:,:,0])
        inter11 = (1./2) * Phih[:,:,1] * np.conj(Phih[:,:,1])
        inter22 = (1./2) * Phih[:,:,2] * np.conj(Phih[:,:,2])
        inter01 = (1./2) * (Phih[:,:,0]*np.conj(Phih[:,:,1]) + np.conj(Phih[:,:,0])*Phih[:,:,1])
        inter02 = (1./2) * (Phih[:,:,0]*np.conj(Phih[:,:,2]) + np.conj(Phih[:,:,0])*Phih[:,:,2])
        inter12 = (1./2) * (Phih[:,:,1]*np.conj(Phih[:,:,2]) + np.conj(Phih[:,:,1])*Phih[:,:,2])
    
        KEh =  self.k1**2 * (
            inter00*PsiPsidz_00 + inter11*PsiPsidz_11 + inter22*PsiPsidz_22 +
            inter01*PsiPsidz_01 + inter02*PsiPsidz_02 + inter12*PsiPsidz_12
        )
    
        return sum_series(KEh).real/(self.nx*self.ny)**2
    
    
    def volume_integrated_APE(self,z_upper=None,z_lower=None,Nz=100):
        if z_upper is None:
            z_upper = 0
        if z_lower is None:
            z_lower = -self.H(self.t).sum()
            
        Z = np.linspace(z_lower, z_upper, Nz)
    
        Psi0 = np.asarray([self.Psi(z, self.k1, 0, t=self.t)for z in Z ])
        Psi1 = np.asarray([self.Psi(z, self.k1, 1, t=self.t) for z in Z ])
        Psi2 = np.asarray([self.Psi(z, self.k1, 2, t=self.t) for z in Z ])
    
        dPsi0 = np.gradient(Psi0, Z, axis=0)
        dPsi1 = np.gradient(Psi1, Z, axis=0)
        dPsi2 = np.gradient(Psi2, Z, axis=0)
        sigma_n2 = np.where(Z > -self.H(self.t)[0], self.sigma[0]**(-2.), self.sigma[1]**(-2.))
        sigma_n2 = sigma_n2[:,np.newaxis,np.newaxis]
    
        dPsidPsidz_00 = simps(sigma_n2*dPsi0*dPsi0, Z, axis=0)
        dPsidPsidz_01 = simps(sigma_n2*dPsi0*dPsi1, Z, axis=0)
        dPsidPsidz_02 = simps(sigma_n2*dPsi0*dPsi2, Z, axis=0)
        dPsidPsidz_11 = simps(sigma_n2*dPsi1*dPsi1, Z, axis=0)
        dPsidPsidz_12 = simps(sigma_n2*dPsi1*dPsi2, Z, axis=0)
        dPsidPsidz_22 = simps(sigma_n2*dPsi2*dPsi2, Z, axis=0)
    
        # Compute Phih = thetah/m
        if self.m is None: self.create_m_matrix()
        if self.Qh is None: self.compute_Qh()
        Phih = self.compute_Phih()
    
        # interaction terms
        inter00 = (1./2) * Phih[:,:,0] * np.conj(Phih[:,:,0])
        inter11 = (1./2) * Phih[:,:,1] * np.conj(Phih[:,:,1])
        inter22 = (1./2) * Phih[:,:,2] * np.conj(Phih[:,:,2])
        inter01 = (1./2) * (Phih[:,:,0]*np.conj(Phih[:,:,1]) + np.conj(Phih[:,:,0])*Phih[:,:,1])
        inter02 = (1./2) * (Phih[:,:,0]*np.conj(Phih[:,:,2]) + np.conj(Phih[:,:,0])*Phih[:,:,2])
        inter12 = (1./2) * (Phih[:,:,1]*np.conj(Phih[:,:,2]) + np.conj(Phih[:,:,1])*Phih[:,:,2])
    
        APEh = (
            inter00*dPsidPsidz_00 + inter11*dPsidPsidz_11 + inter22*dPsidPsidz_22 +
            inter01*dPsidPsidz_01 + inter02*dPsidPsidz_02 + inter12*dPsidPsidz_12
        )
    
        return sum_series(APEh).real/(self.nx*self.ny)**2
    
    # layer 0 is the upper most layer. There are nz-1 layers.
    def volume_integrated_KE_layer(self,layer_index,Nz=100):
        H = self.H(self.t)
        if layer_index == 0:
            z_upper = 0
        else:
            z_upper = -H[layer_index-1]
        
        if layer_index == self.nz:
            z_lower = -H.sum()
        else:
            z_lower = -H[layer_index]
    
        return self.volume_integrated_KE(z_upper=z_upper,z_lower=z_lower,Nz=Nz)
    
    # layer 0 is the upper most layer. There are nz-1 layers.
    def volume_integrated_APE_layer(self,layer_index,Nz=100):
        H = self.H(self.t)
        if layer_index == 0:
            z_upper = 0
        else:
            z_upper = -H[layer_index-1]
        
        if layer_index == self.nz:
            z_lower = -H.sum()
        else:
            z_lower = -H[layer_index]
    
        return self.volume_integrated_APE(z_upper=z_upper,z_lower=z_lower,Nz=Nz)

    #### The following functions are to compute the Kinetic energy
    #### spectrum budget

    def compute_dmdt(self,dt=1e-4):
        t_minus = self.t - dt
        t_plus = self.t + dt
    
        m_plus = self.create_m_matrix(t=t_plus,return_m=True)
        m_minus = self.create_m_matrix(t=t_minus,return_m=True)
        dmdt = (m_plus-m_minus)/(2*dt)

        return dmdt

    ## Compute the time tendency of Phi
    def compute_Phi_t(self):
        ## Compute the Jacobian J(psi,theta)
        ## Recast as u*theta_x + v*theta_y
        
        # Compute the velocity
        self.compute_Ph()
        U, V = self.compute_velocity()
        
        #  Compute the horizontal gradients of theta
        self.compute_Qh()
        Qhx = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        Qhy = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        for z in range(self.nz):
            Qhx[:,:,z] = 1j*self.kx*self.Qh[:,:,z]
            Qhy[:,:,z] = 1j*self.ky*self.Qh[:,:,z]
        Qx = self.ifft(Qhx)
        Qy = self.ifft(Qhy)
        
        ## Compute the jacobian
        J_psi_theta = U*Qx + V*Qy
        J_psi_theta_h = self.fft(J_psi_theta)
        
        ## Compute the term depending on the time-tendency of m
        
        ## First find the streamfunction generated by theta at each
        ## layer (Phi)
        self.create_m_matrix()
        Phih = self.compute_Phih()
        
        ## Find dmdt
        dmdt = self.compute_dmdt()
        dmdt_Phi_h = dmdt*Phih
        
        ## Compute the dissipation
        Dh = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        if self.use_bi_hypo_smag:
            A4_hypo_smag = self.compute_A4_hypo_smag()
            for z in range(self.nz):
                inv_lap_Q = self.ifft(self.k0 * self.Qh[:,:,z])
                A4_inv_lap_Q_hat = self.fft(A4_hypo_smag * inv_lap_Q)
                Dh[:,:,z] = self.k0 * A4_inv_lap_Q_hat
        else:
            print("Not calculating hypoviscous contribution in the Phih calculation!")
    
        Rh = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        if self.use_bi_leith:
            A4_leith = self.compute_A4_leith()
            for z in range(self.nz):
                Rh[:,:,z] = A4_leith[z] * self.k2**2 * self.Qh[:,:,z]
        else:
            print("Not calculating hyperviscous contribution in the Phih calculation!")
    
        Disph = Dh + Rh
        
        dPhi_h_dt = (1/self.m)*(J_psi_theta_h - dmdt_Phi_h - Disph)
        return dPhi_h_dt
    
    # compute the time-tendency of buoyancy
    def compute_bh_t(self,z,t=None,dt=1e-4):
        ## The time tendency consists of two terms. The first 
        ## is a product between the dPhi_h_dt and dPsidz whereas
        ## the second terms is a product between Phi_h and 
        ## d^2_Psi_dz_dt
    
        if t is None:
            t = self.t
        
        ## Find dPsidz and d2_Psi_dz_dt
        t_plus = t + dt
        t_minus = t - dt
        
        dPsidz = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        dPsidz_plus = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        dPsidz_minus = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        
        for n in range(self.nz):
            dPsidz[:,:,n] = self.dPsidz(z,t,self.k1,n,order=1)
            ## the next two are needed to compute the time derivative
            dPsidz_plus[:,:,n] = self.dPsidz(z,t_plus,self.k1,n,order=1)
            dPsidz_minus[:,:,n] = self.dPsidz(z,t_minus,self.k1,n,order=1)
    
        # the time derivative
        d2_Psi_dz_dt = (dPsidz_plus-dPsidz_minus)/(2*dt)
        
        ## Compute quantities necessary in the calculation of b_t
        dPhi_h_dt = self.compute_Phi_t()
        self.create_m_matrix()
        Phih = self.compute_Phih()
        
        ## Now find the time tendency of b
        bh_t = np.zeros((self.nkx,self.nky),dtype=np.complex128)
        for n in range(self.nz):
            bh_t += self.f0*( dPhi_h_dt[:,:,n]*dPsidz[:,:,n] +  Phih[:,:,n]*d2_Psi_dz_dt[:,:,n]  )
    
        # replace the nan at k=0 with a zero
        bh_t[0,0] = 0.0
        return bh_t

    def compute_vertical_velocity(self, z):
        ## Choose the correct startification
        layer_index = -1
        current_sigma = 0
        if (0 >= z) and (z >- self.H(self.t)[0]):
            layer_index = 0
        elif (-self.H(self.t)[0] >= z) and (z >= -self.H(self.t).sum()):
            layer_index = 1
        else:
            print("z must be between -self.H(t)[0] and -self.H(t).sum()")
            return None
        current_sigma = self.sigma[layer_index]
        
        N = self.f0*current_sigma
    
        bh_t = self.compute_bh_t(z)
    
        ## Compute bh and and the velocities at an arbitrary z
        self.create_m_matrix()
        Phih = -self.Qh/self.m
    
        Psi = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        dPsidz = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        for n in range(self.nz):
            dPsidz[:,:,n] = self.dPsidz(z,self.t,self.k1,n,order=1)
            Psi[:,:,n] = self.Psi(z,self.t,self.k1,n) 
    
        ## Now Find b and velocity
        ph = np.zeros((self.nkx,self.nky),dtype=np.complex128)
        bh = np.zeros((self.nkx,self.nky),dtype=np.complex128)
        for n in range(self.nz):
            bh += self.f0 * Phih[:,:,n] * dPsidz[:,:,n]
            ph += Phih[:,:,n] * Psi[:,:,n]
    
        Uh = -1j*self.ky*ph
        Vh = +1j*self.kx*ph
    
        U = self.ifft(Uh)
        V = self.ifft(Vh)
        
        #  Compute the horizontal gradients of b
        bhx = 1j*self.kx*bh
        bhy = 1j*self.ky*bh
        
        bx = self.ifft(bhx)
        by = self.ifft(bhy)
        
        ## Compute the jacobian
        J_psi_b = U*bx + V*by
        J_psi_b_h = self.fft(J_psi_b)
    
        ## Compute the viscous terms
        Dh = np.zeros((self.nkx,self.nky),dtype=np.complex128)
        if self.use_bi_hypo_smag:
            A4_hypo_smag = self.compute_A4_hypo_smag()        
            inv_lap_b = self.ifft(self.k0 * bh)
            A4_inv_lap_b_hat = self.fft(A4_hypo_smag * inv_lap_b)
            Dh = self.k0 * A4_inv_lap_b_hat
        else:
            print("Not calculating hypoviscous contribution in the vertical velocity calculation!")
    
        Rh = np.zeros((self.nkx,self.nky),dtype=np.complex128)
        if self.use_bi_leith:
            A4_leith = self.compute_A4_leith()
            Rh = A4_leith[layer_index] * self.k2**2 * bh
        else:
            print("Not calculating hyperviscous contribution in the vertical velocity calculation!")
    
        Disph = Dh + Rh
        
        #wh = -(1./N**2)*( bh_t +  J_psi_b_h - Disph )
        wh = -(1./N**2)*( bh_t +  J_psi_b_h - Disph)
        return self.ifft(wh)

    def compute_vertical_velocity_derivative(self, z):

        zetah_t = compute_zetah_t(self, z)
    
        ## Compute bh and and the velocities at an arbitrary z
        self.create_m_matrix()
        Phih = -self.Qh/self.m
    
        Psi = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        for n in range(self.nz):
            Psi[:,:,n] = self.Psi(z,self.t,self.k1,n) 
    
        ## Now find zeta and the velocity
        ph = np.zeros((self.nkx,self.nky),dtype=np.complex128)
        zeta_h = np.zeros((self.nkx,self.nky),dtype=np.complex128)
        for n in range(self.nz):
            zeta_h += -self.k2 * Phih[:,:,n] * Psi[:,:,n]
            ph += Phih[:,:,n] * Psi[:,:,n]
    
        Uh = -1j*self.ky*ph
        Vh = +1j*self.kx*ph
    
        U = self.ifft(Uh)
        V = self.ifft(Vh)
        
        #  Compute the horizontal gradients of b
        zetax_h = 1j*self.kx*zeta_h
        zetay_h = 1j*self.ky*zeta_h
        
        zetax = self.ifft(zetax_h)
        zetay = self.ifft(zetay_h)
        
        ## Compute the jacobian
        J_psi_zeta = U*zetax + V*zetay
        J_psi_zeta_h = self.fft(J_psi_zeta)
    
        wz_h = (1./self.f0) * (zetah_t + J_psi_zeta_h)
        return wz_h

    ### Compute the horizontally homogeneous Smagorisnky 
    ### biharmonic hypoviscosity
    def compute_A4_hypo_smag(self):
        if self.Ph is None:
            self.compute_Ph()
        
        delta = max(self.Lx,self.Ly)
        k_star = 2*np.pi/(self.C_smag*delta)
    
        sigma = self.sigma
        H = self.H(self.t)
    
        T0 = np.tanh(sigma[0]*H[0]*k_star)
        T1 = np.tanh(sigma[1]*H[1]*k_star)
        m_value = (k_star / sigma[0]) * (sigma[1] * T0 + sigma[0] * T1) / (sigma[1] + sigma[0] * T0 * T1)
    
        ikx = np.zeros_like(self.kx)
        ikx[self.kx>0] = 1.0 / self.kx[self.kx>0]
        iky = np.zeros_like(self.ky)
        iky[self.ky>0] = 1.0 / self.ky[self.ky>0]
    
        ikx = ikx[...,np.newaxis]
        iky = iky[...,np.newaxis]
    
        gradx_psi = self.ifft(ikx*1j*self.Ph)
        grady_psi = self.ifft(iky*1j*self.Ph)
    
        grad_psi2 = gradx_psi**2 + grady_psi**2
        grad_psi2_hat = self.fft(grad_psi2)
    
        ekf = np.exp( - 6. * (self.L_filter/(2*np.pi))**2 * self.k2)
        smoothed_grad_psi2 = self.ifft(ekf[...,np.newaxis]*grad_psi2_hat)
    
        h_star = H[0] + (sigma[1]*k_star)**(-1)
    
        ## Make the viscosity depth dependent
        A4_hypo_smag = self.B_smag * k_star**8 * np.sqrt(h_star * smoothed_grad_psi2[:,:,0]/m_value)
        return A4_hypo_smag

    def compute_A4_leith(self):
        if self.Qh is None:
            self.compute_Qh()
    
        delta = min(self.dx,self.dy)
        k_star = np.pi/(self.C_leith*delta)
    
        sigma = self.sigma
        H = self.H(self.t)
    
        C0 = np.cosh(sigma[0]*H[0]*k_star)
        C1 = np.cosh(sigma[1]*H[1]*k_star)
        T0 = np.tanh(sigma[0]*H[0]*k_star)
        T1 = np.tanh(sigma[1]*H[1]*k_star) 
    
        m_value = np.zeros(self.nz)
        m_value[0] = (k_star / sigma[0]) * (sigma[1] * T0 + sigma[0] * T1) / (sigma[1] + sigma[0] * T0 * T1)
        m_value[1] = k_star * ((T0 / sigma[0]) + (T1 / sigma[1]))
        m_value[2] = k_star * (sigma[1] * T0 + sigma[0] * T1) / (sigma[1] * (sigma[0] + sigma[1] * T0 * T1))
    
        ## Turn the square wavenumbers from a 2D to a 3D fields
        k2 = self.k2[...,np.newaxis]
        
        lap_q = self.ifft(-k2*self.Qh)
        lap_q2_mean = (lap_q**2).mean(axis=(0,1))
    
        A4_leith = self.B_leith * (k_star**(-4)/m_value) * np.sqrt(lap_q2_mean)
            
        return A4_leith


    
    ### Compute the energy budget ###
    ### dE/dt + Ch + Th = Sh + Dh+Rh
    ### Ch is the term due to changing stratification
    ### Th is the advection term
    ### Sh is the source term
    ### Dh is the large-scale dissipation
    ### Rh is the small-scale dissipation
    def energy_budget(self,eps=1e-4):
        # Compute the velocity
        self.compute_Ph()
        U, V = self.compute_velocity()
        
        #  Compute the horizontal gradients of theta
        self.compute_Qh()
        Qhx = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        Qhy = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        for z in range(self.nz):
            Qhx[:,:,z] = 1j*self.kx*self.Qh[:,:,z]
            Qhy[:,:,z] = 1j*self.ky*self.Qh[:,:,z]
        Qx = self.ifft(Qhx)
        Qy = self.ifft(Qhy)
        
        ## Compute the jacobian
        J_psi_theta = U*Qx + V*Qy
        J_psi_theta_h = self.fft(J_psi_theta)/(self.nx*self.ny)**2
        
        ## Multiply the -psi_k^dagger and sum over surfaces
        Th = (np.conjugate(-self.Ph)*J_psi_theta_h).sum(axis=2).real/(self.nx*self.ny)
        
        ## Compute the source term
        Shj = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        for z in range(self.nz):
            Shj[:,:,z] = np.real(self.kx*self.U[z]*1j*np.conjugate(self.Ph[:,:,z])*self.Qh[:,:,z])
        Sh = Shj.sum(axis=2).real/(self.nx*self.ny)**2
    
        ## Compute dissipatation at large scales
        Dhj = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        if self.use_bi_hypo_smag:
            A4_hypo_smag = self.compute_A4_hypo_smag()
            for z in range(self.nz):
                inv_lap_Q = self.ifft(self.k0*self.Qh[:,:,z])
                A4_inv_lap_Q_hat = self.fft(A4_hypo_smag*inv_lap_Q)
                Dhj[:,:,z] = np.real( np.conjugate(self.Ph[:,:,z]) * self.k0*A4_inv_lap_Q_hat)
                #Dhj[:,:,z] = np.real(A4_hypo_smag[z]*self.k0**2*np.conjugate(self.Ph[:,:,z])*self.Qh[:,:,z])
        else:
            for z in range(self.nz):
                Dhj[:,:,z] = np.real(self.A0*self.k0*np.conjugate(self.Ph[:,:,z])*self.Qh[:,:,z])
        Dh = Dhj.sum(axis=2).real/(self.nx*self.ny)**2
        
        ## Compute dissipatation at small scales
        Rhj = np.zeros((self.nkx,self.nky,self.nz),dtype=np.complex128)
        if self.use_bi_leith:
            A4_leith = self.compute_A4_leith()
            for z in range(self.nz):
                Rhj[:,:,z] = np.real(A4_leith[z]*self.k2**2*np.conjugate(self.Ph[:,:,z])*self.Qh[:,:,z])
        else:
            for z in range(self.nz):
                Rhj[:,:,z] = np.real(self.A20*self.k2**10*np.conjugate(self.Ph[:,:,z])*self.Qh[:,:,z])
        Rh = Rhj.sum(axis=2).real/(self.nx*self.ny)**2
    
        ## Compute term due to changing stratification
        Ch = stratification_energy_tendency(self,eps=eps)
            
        return Ch, Th, Sh, Dh, Rh

    ### Compute the stratification time-tendencu term 
    ## in the energy budget, which is###
    ### dE/dt + Ch + Th = Sh + Dh+Rh
    ### This function computes Ch ##
    def stratification_energy_tendency_old(self,eps):

        sigma = self.sigma
        K = self.k1
        H = self.H(self.t)
    
        S0 = np.sinh(sigma[0]*H[0]*K)
        S0[0,0] = 1.0 # avoid division by zero
        S1 = np.sinh(sigma[1]*H[1]*K)
        S1[0,0] = 1.0 # avoid division by zero
        C0 = np.cosh(sigma[0]*H[0]*K)
        C1 = np.cosh(sigma[1]*H[1]*K)
        T0 = np.tanh(sigma[0]*H[0]*K)
        T1 = np.tanh(sigma[1]*H[1]*K)
    
        a_SC = sigma[0]*S0*S1 + sigma[1]*C0*C1
        a_CS = sigma[0]*C0*C1 + sigma[1]*S0*S1
        b_CS = sigma[0]*C0*S1 + sigma[1]*S0*C1
    
        # cofficinets to interaction terms
        V00 = (K*b_CS/a_SC)**2 - (K*sigma[1]*S1/a_SC)**2
        V11 = -(K*T1)**2
        V22 = - (K*sigma[1]*S0/a_CS)**2
        V01 = - K**2*sigma[1]*S1*T1/a_SC
        V02 = ( T0*T1 / (sigma[0]+sigma[1]*T0*T1) ) * (sigma[1]**2*K**2/a_SC)
        V12 = K**2*(sigma[1]/C1)*(1./a_CS)
    
        if self.m is None: self.create_m_matrix()
        if self.Qh is None: self.compute_Qh()
        Phih = self.compute_Phih()
    
        # interaction terms
        inter00 = (1./2) * Phih[:,:,0] * np.conj(Phih[:,:,0])
        inter11 = (1./2) * Phih[:,:,1] * np.conj(Phih[:,:,1])
        inter22 = (1./2) * Phih[:,:,2] * np.conj(Phih[:,:,2])
        inter01 = (1./2) * (Phih[:,:,0]*np.conj(Phih[:,:,1]) + np.conj(Phih[:,:,0])*Phih[:,:,1])
        inter02 = (1./2) * (Phih[:,:,0]*np.conj(Phih[:,:,2]) + np.conj(Phih[:,:,0])*Phih[:,:,2])
        inter12 = (1./2) * (Phih[:,:,1]*np.conj(Phih[:,:,2]) + np.conj(Phih[:,:,1])*Phih[:,:,2])
    
        ## z_1(t) = -H(self.t)[0]
        dzdt = -(self.H(self.t+eps)[0]-self.H(self.t-eps)[0])/(2*eps)
    
        return dzdt * (
                inter00*V00 + inter11*V11 + inter22*V22 +
                inter01*V01 + inter02*V02 + inter12*V12
                        ).real/(self.nx*self.ny)**2

    ### Compute the APE time-tendency term
    ##  dE/dt + G = 0 where G is the time-tendency term.
    def APE_energy_tendency_old(self,eps):
    
        sigma = self.sigma
        K = self.k1
        H = self.H(self.t)
    
        S0 = np.sinh(sigma[0]*H[0]*K)
        S0[0,0] = 1.0 # avoid division by zero
        S1 = np.sinh(sigma[1]*H[1]*K)
        S1[0,0] = 1.0 # avoid division by zero
        C0 = np.cosh(sigma[0]*H[0]*K)
        C1 = np.cosh(sigma[1]*H[1]*K)
        T0 = np.tanh(sigma[0]*H[0]*K)
        T1 = np.tanh(sigma[1]*H[1]*K)
    
        a_SC = sigma[0]*S0*S1 + sigma[1]*C0*C1
        a_CS = sigma[0]*C0*C1 + sigma[1]*S0*S1
        b_CS = sigma[0]*C0*S1 + sigma[1]*S0*C1
    
        # cofficinets to interaction terms
        G00 =  (K*sigma[0]*b_CS/a_SC)**2  + (K*sigma[1]**2*S1/a_SC)**2 
        G11 = (K*sigma[1]*T1)**2
        G22 = (K*sigma[1]**2*S0/a_CS)**2
        G01 = K**2*sigma[1]**3*S1*T1/a_SC
        G02 = -(K*sigma[1]**2)**2*S0*S1/(a_CS*a_SC)
        G12 = -K**2*sigma[1]**3*S0*T1/a_CS
    
        if self.m is None: self.create_m_matrix()
        if self.Qh is None: self.compute_Qh()
        Phih = self.compute_Phih()
    
        # interaction terms
        inter00 = (1./2) * Phih[:,:,0] * np.conj(Phih[:,:,0])
        inter11 = (1./2) * Phih[:,:,1] * np.conj(Phih[:,:,1])
        inter22 = (1./2) * Phih[:,:,2] * np.conj(Phih[:,:,2])
        inter01 = (1./2) * (Phih[:,:,0]*np.conj(Phih[:,:,1]) + np.conj(Phih[:,:,0])*Phih[:,:,1])
        inter02 = (1./2) * (Phih[:,:,0]*np.conj(Phih[:,:,2]) + np.conj(Phih[:,:,0])*Phih[:,:,2])
        inter12 = (1./2) * (Phih[:,:,1]*np.conj(Phih[:,:,2]) + np.conj(Phih[:,:,1])*Phih[:,:,2])
    
        ## z_1(t) = -H(self.t)[0]
        dzdt = -(self.H(self.t+eps)[0]-self.H(self.t-eps)[0])/(2*eps)
    
        return -(1/2) * (sigma[1]**(-2)-sigma[0]**(-2)) * dzdt * (
                inter00*G00 + inter11*G11 + inter22*G22 +
                inter01*G01 + inter02*G02 + inter12*G12
                        ).real/(self.nx*self.ny)**2

    ## dE/dt + C = S + D + R
    ## This function computes C, but does not sum
    ## over the fourier wavenumbers
    def stratification_energy_tendency(self,eps):
        
        sigma = self.sigma
        K = self.k1
        H = self.H(self.t)
        
        S0 = np.sinh(sigma[0]*H[0]*K)
        S0[0,0] = 1.0 # avoid division by zero
        S1 = np.sinh(sigma[1]*H[1]*K)
        S1[0,0] = 1.0 # avoid division by zero
        C0 = np.cosh(sigma[0]*H[0]*K)
        C1 = np.cosh(sigma[1]*H[1]*K)
        T0 = np.tanh(sigma[0]*H[0]*K)
        T1 = np.tanh(sigma[1]*H[1]*K)
    
        ## Necessary coeffecients for the vertical structure
        ## Psi_{j} = a_{jl}*cosh(sigma[l]k(z-z_l)) + b_{jl}*sinh(sigma[l]*j(z-z_l))
        a00 = 1.
        b00 = (sigma[0]*T1 + sigma[1]*T0)/(sigma[0]*T0*T1 + sigma[1])
        a10 = 1./C0
        b10 = 0.
        a20 = (sigma[0]/(C0*C1)) * (1/(sigma[0] + sigma[1]*T0*T1))
        b20 = 0.
    
        b01 = (sigma[1]/C0) * (T1/(sigma[0]*T0*T1 + sigma[1]))
        b11 = T1
        b21 = - (sigma[1]/C1)*(T0/(sigma[0] + sigma[1]*T0*T1))
    
        ## The values of the vertical structure dPsidz above 
        ## and below the interface (Psi0p=dPsi0/dz)
    
        Psi0p_plus = sigma[0]*K*(-a00*S0 + b00*C0)
        Psi1p_plus = sigma[0]*K*(-a10*S0 + b10*C0)
        Psi2p_plus = sigma[0]*K*(-a20*S0 + b20*C0)
    
        Psi0p_minus = sigma[1]*K*b01
        Psi1p_minus = sigma[1]*K*b11
        Psi2p_minus = sigma[1]*K*b21
    
        ## Calculate the jump in dPsi_i/dz*dPsi_j/dz divided by density squared
    
        delta_Psi_square_00 = (Psi0p_plus/sigma[0])**2 - (Psi0p_minus/sigma[1])**2
        delta_Psi_square_11 = (Psi1p_plus/sigma[0])**2 - (Psi1p_minus/sigma[1])**2
        delta_Psi_square_22 = (Psi2p_plus/sigma[0])**2 - (Psi2p_minus/sigma[1])**2
    
        delta_Psi_square_01 = Psi0p_plus*Psi1p_plus/sigma[0]**2 - Psi0p_minus*Psi1p_minus/sigma[1]**2
        delta_Psi_square_02 = Psi0p_plus*Psi2p_plus/sigma[0]**2 - Psi0p_minus*Psi2p_minus/sigma[1]**2
        delta_Psi_square_12 = Psi1p_plus*Psi2p_plus/sigma[0]**2 - Psi1p_minus*Psi2p_minus/sigma[1]**2
    
        ## Compute the streamfunctions phi_j      
        if self.m is None: self.create_m_matrix()
        if self.Qh is None: self.compute_Qh()
        Phih = self.compute_Phih()
    
        ## Compute the terms in the sum
        term00 = delta_Psi_square_00 * Phih[:,:,0] * np.conj(Phih[:,:,0])
        term11 = delta_Psi_square_11 * Phih[:,:,1] * np.conj(Phih[:,:,1])
        term22 = delta_Psi_square_22 * Phih[:,:,2] * np.conj(Phih[:,:,2])
        term01 = delta_Psi_square_01 * (Phih[:,:,0]*np.conj(Phih[:,:,1]) + np.conj(Phih[:,:,0])*Phih[:,:,1])
        term02 = delta_Psi_square_02 * (Phih[:,:,0]*np.conj(Phih[:,:,2]) + np.conj(Phih[:,:,0])*Phih[:,:,2])
        term12 = delta_Psi_square_12 * (Phih[:,:,1]*np.conj(Phih[:,:,2]) + np.conj(Phih[:,:,1])*Phih[:,:,2])
    
        ## Compute time derivative of interface: z_1(t) = -H(self.t)[0]
        dzdt = -(self.H(self.t+eps)[0]-self.H(self.t-eps)[0])/(2*eps)
    
        ## Compute final expression
        delta_partial_psi_dz_squared = (term00 + term11 + term22 + term01 + term02 + term12).real/(self.nx*self.ny)**2
        result = 0.5 * delta_partial_psi_dz_squared * dzdt
        return result
    
    
    ## dE/dt + G = S + D + R
    ## This function computes G, but does not sum
    ## over the fourier wavenumbers
    def APE_energy_tendency(self,eps):
        
        sigma = self.sigma
        K = self.k1
        H = self.H(self.t)
        
        S0 = np.sinh(sigma[0]*H[0]*K)
        S0[0,0] = 1.0 # avoid division by zero
        S1 = np.sinh(sigma[1]*H[1]*K)
        S1[0,0] = 1.0 # avoid division by zero
        C0 = np.cosh(sigma[0]*H[0]*K)
        C1 = np.cosh(sigma[1]*H[1]*K)
        T0 = np.tanh(sigma[0]*H[0]*K)
        T1 = np.tanh(sigma[1]*H[1]*K)
    
        ## Necessary coeffecients for the vertical structure
        ## Psi_{j} = a_{jl}*cosh(sigma[l]k(z-z_l)) + b_{jl}*sinh(sigma[l]*j(z-z_l))
        a00 = 1.
        b00 = (sigma[0]*T1 + sigma[1]*T0)/(sigma[0]*T0*T1 + sigma[1])
        a10 = 1./C0
        b10 = 0.
        a20 = (sigma[0]/(C0*C1)) * (1/(sigma[0] + sigma[1]*T0*T1))
        b20 = 0.
    
        b01 = (sigma[1]/C0) * (T1/(sigma[0]*T0*T1 + sigma[1]))
        b11 = T1
        b21 = - (sigma[1]/C1)*(T0/(sigma[0] + sigma[1]*T0*T1))
    
        ## The values of the vertical structure dPsidz above 
        ## and below the interface (Psi0p=dPsi0/dz)
    
        Psi0p_plus = sigma[0]*K*(-a00*S0 + b00*C0)
        Psi1p_plus = sigma[0]*K*(-a10*S0 + b10*C0)
        Psi2p_plus = sigma[0]*K*(-a20*S0 + b20*C0)
    
        Psi0p_minus = sigma[1]*K*b01
        Psi1p_minus = sigma[1]*K*b11
        Psi2p_minus = sigma[1]*K*b21
    
        ## Calculate the average in dPsi_i/dz*dPsi_j/dz 
    
        delta_Psi_square_00 = (Psi0p_plus)**2 + (Psi0p_minus)**2
        delta_Psi_square_11 = (Psi1p_plus)**2 + (Psi1p_minus)**2
        delta_Psi_square_22 = (Psi2p_plus)**2 + (Psi2p_minus)**2
    
        delta_Psi_square_01 = Psi0p_plus*Psi1p_plus + Psi0p_minus*Psi1p_minus
        delta_Psi_square_02 = Psi0p_plus*Psi2p_plus + Psi0p_minus*Psi2p_minus
        delta_Psi_square_12 = Psi1p_plus*Psi2p_plus + Psi1p_minus*Psi2p_minus
    
        ## Compute the streamfunctions phi_j      
        if self.m is None: self.create_m_matrix()
        if self.Qh is None: self.compute_Qh()
        Phih = self.compute_Phih()
    
        ## Compute the terms in the sum
        term00 = delta_Psi_square_00 * Phih[:,:,0] * np.conj(Phih[:,:,0])
        term11 = delta_Psi_square_11 * Phih[:,:,1] * np.conj(Phih[:,:,1])
        term22 = delta_Psi_square_22 * Phih[:,:,2] * np.conj(Phih[:,:,2])
        term01 = delta_Psi_square_01 * (Phih[:,:,0]*np.conj(Phih[:,:,1]) + np.conj(Phih[:,:,0])*Phih[:,:,1])
        term02 = delta_Psi_square_02 * (Phih[:,:,0]*np.conj(Phih[:,:,2]) + np.conj(Phih[:,:,0])*Phih[:,:,2])
        term12 = delta_Psi_square_12 * (Phih[:,:,1]*np.conj(Phih[:,:,2]) + np.conj(Phih[:,:,1])*Phih[:,:,2])
    
        ## Compute time derivative of interface: z_1(t) = -H(self.t)[0]
        dzdt = -(self.H(self.t+eps)[0]-self.H(self.t-eps)[0])/(2*eps)
    
        ## Compute final expression
        ave_partial_psi_dz_squared = (term00 + term11 + term22 + term01 + term02 + term12).real/(self.nx*self.ny)**2
        result = -0.25 * (sigma[1]**(-2) - sigma[0]**(-2)) * ave_partial_psi_dz_squared * dzdt
        return result

        
    
    ###################################################################
    ### Plotting ######################################################
    ################################################################### 
         
    def plot(self,field,size=10,title=None,save=0,number=0,path='',cb=1,
             vabs=None,vmin=None,vmax=None,cmap=cc.m_CET_D1,
             xlims=None,ylims=None):

        if vabs:
            vmin = -vabs
            vmax = vabs
            norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        elif vmax:
            norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        else:
            norm=None
        
        if cb:
            width = size*1.05
            fig = plt.figure(figsize=(width,size))
            ax = plt.subplot(111)

            #im = ax.imshow(m.q.squeeze(),cmap='binary_r')
            im = ax.pcolormesh(self.x,self.y,field.squeeze(),cmap=cmap,shading='auto',norm=norm)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            fig.colorbar(im, cax=cax, orientation='vertical')
        
        else:
            fig = plt.figure(figsize=(size,size))
            ax = plt.subplot(111)
            #im = ax.imshow(m.q.squeeze(),cmap='binary_r')
            im = ax.pcolormesh(self.x,self.y,field.squeeze(),cmap=cmap,shading='auto',norm=norm)

        ax.set_title(title)

        ax.set_xticks([])
        ax.set_yticks([])
        
        if xlims:
            ax.set_xlim(xlims)
        if ylims:
            ax.set_ylim(ylims)
        
        if save:
            name = str(number).zfill(9)
            plt.savefig(path+name+'.png',bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
                
        return im

    # t and z in the time and z index
    def plot_q(self,z,**kwargs):
        field = self.Q[:,:,z]
        self.plot(field,**kwargs)
    
    def plot_p(self,z,**kwargs):
        field = self.P[:,:,z]
        self.plot(field,**kwargs)
        
    def plot_all(self, field, size=5, title=None, save=0, save_path=None, number=0, path='', 
                cb=1, cb_label=None, vabs=None, vmin=None, vmax=None, 
                cmap=cc.m_CET_D1, xlims=None, ylims=None, vabs_arr=None,vmax_arr=None,
                fontsize=25):
        
        fig, axes = plt.subplots(1, 3, figsize=(size * 3, size),gridspec_kw={'wspace': 0.05})
        
        for i in range(3):
            if vabs_arr:
                vmin_i = -vabs_arr[i]
                vmax_i = vabs_arr[i]
            elif vmax_arr:
                vmin_i = 0
                vmax_i = vmax_arr[i]
            elif vabs:
                vmin_i = -vabs
                vmax_i = vabs
            elif vmax:
                vmax_i = vmax
                vmin_i = 0
            else:
                vabs_i = np.max(np.abs(field[:,:,i]))
                vmin_i = -vabs_i
                vmax_i = vabs_i


            ax = axes[i]
            norm = plt.Normalize(vmin=vmin_i, vmax=vmax_i)

            im = ax.pcolormesh(self.x, self.y, field[:, :, i].squeeze(), 
                                cmap=cmap, shading='auto', norm=norm)
            
            if i==0:
                ax.set_title('Surface',fontsize=fontsize)
            if i==1:
                ax.set_title('Mixed layer',fontsize=fontsize)
            if i==2:
                ax.set_title('Pycnocline',fontsize=fontsize)

            ax.set_xticks([])
            ax.set_yticks([])

            if xlims:
                ax.set_xlim(xlims)
            if ylims:
                ax.set_ylim(ylims)
            

            if cb:
                cax = fig.add_axes([ax.get_position().x0+0.01, 0.05, ax.get_position().width-0.02, 0.03])
                cbar = fig.colorbar(im, cax=cax, orientation='horizontal')
            if cb_label:
                cbar.set_label(cb_label, labelpad=5)
        #if cb:
            # Adjust the position of the color bar
            #cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
            #cbar = fig.colorbar(im, cax=cax, orientation='vertical')
        #if cb_label:
        #    cbar.set_label(cb_label, rotation=270, labelpad=15)

        fig.suptitle(title,fontsize=fontsize)

        if save:
            #name = str(number).zfill(5)
            #save_path = os.path.join(path,name + '.png')
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

        return fig
    
    def plot_all_Q(self,**kwargs):
        field = self.Q[:,:,:]
        self.plot_all(field,**kwargs)

    def plot_all_P(self,**kwargs):
        field = self.P[:,:,:]
        self.plot_all(field,**kwargs)

    def plot_all_Phi(self,**kwargs):
        field = self.Phi[:,:,:]
        self.plot_all(field,**kwargs)

    def plot_all_Qm(self,order=[0,1,2],**kwargs):
        if self.Qm is None:
            print("Compute Qm first")
            return
        field = np.zeros_like(self.Qm)
        field[:,:,0] = self.Qm[:,:,order[0]]
        field[:,:,1] = self.Qm[:,:,order[1]]
        field[:,:,2] = self.Qm[:,:,order[2]]
        self.plot_all(field,**kwargs)

    def plot_all_Pm(self,order=[0,1,2],**kwargs):
        if self.Pm is None:
            print("Compute Qm first")
            return
        field = np.zeros_like(self.Pm[t])
        field[:,:,0] = self.Pm[:,:,order[0]]
        field[:,:,1] = self.Pm[:,:,order[1]]
        field[:,:,2] = self.Pm[:,:,order[2]]
        self.plot_all(field,**kwargs)
        
    def plot_energy(self,T,E,figsize=(10,5),c='k',lw=1.5,lw_zero=0.5,points=False,point_size=10,
                    title='Energy',xlabel='Time (days)',ylabel=r'Energy $(m^3s^4)$'):
        
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        day = 24*3600
        
        ax.plot(T/day,E,c=c,lw=lw)
        if points:
             ax.scatter(T / day, E, c=c, s=point_size) 

        ax.axhline(y=0,c='k',lw=lw_zero)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    def plot_inversion(self,m,K=None,which=None,figsize=(10,5),c=None,labels=None,
               xlim=None,ylim=None,xlabel=r'$|k|$ ($m^{-1}$)',
               ylabel='',title='Inversion functions'):

        if which is None:
            which = [0,1,2]
        if c is None:
            c = ['tab:red','tab:purple','tab:blue']
        if labels is None:
            labels = ['Surface','Mixed Layer','Pycnocline']

        plt.figure(figsize=figsize)
        ax = plt.subplot(111)

        # If we computed m with 2D K array then 
        # plot only along kx axis
        if (K is not None) and len(K.shape)==1:
            K = K ## This should be a 1D K
        elif len(m.shape)==3:
            m = m[:,0,:]
            K = self.k1[:,0]
        else:
            K = self.kr
        
        for i in which:
            ax.loglog(K,m[:,i],color=c[i],label=labels[i])

        # Large scale limit
        ax.loglog(K,self.H(self.t).sum()*K**2,c='k',ls=':',lw='1')

        # Small scale limit
        if 0 in which:
            ax.loglog(K,K/self.sigma[0],c=c[0],ls=':',lw='1')
        if 1 in which:
            ax.loglog(K,K*(1./self.sigma[0]+1./self.sigma[1]),c=c[1],ls=':',lw='1')
        if 2 in which:
            ax.loglog(K,K/self.sigma[1],c=c[2],ls=':',lw='1')

        if xlim is None:
            if K[0]==0:
                xlim = [K[1],K[-1]]
            else:
                xlim = [K[0],K[-1]]

        ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_title(title)

        ax.legend()

    def plot_m(self,**kwargs):
        self.plot_inversion(self.m,**kwargs)

    def plot_evalues(self,**kwargs):
        self.plot_inversion(self.evalues,**kwargs)
    
    def plot_e(self,K=None,which=None,figsize=(10,5),c=None,labels=None,
               xlim=None,ylim=[0,1],xlabel=r'$|k|$ ($m^{-1}$)',
               ylabel='',title=r'Interaction matrix'):
        
        if which is None:
            which = [0,1,2]
        if c is None:
            c = ['tab:red','tab:purple','tab:blue']
        if labels is None:
            labels = ['Surface','Mixed Layer','Pycnocline']

        e = self.e
        # If we computed m with 2D K array then 
        # plot only along kx axis
        if K is not None:
            K = K ## This should be a 1D K
        elif len(e.shape)==4:
            e = e[:,0,:,:]
            K = self.k1[:,0]
        else:
            K = self.kr

        plt.figure(figsize=(10,5))
        ax = plt.subplot(111)

        for i in which:
            ax.semilogx(K,e[:,i,0],color=c[i],label=labels[i])
            ax.semilogx(K,e[:,i,1],color=c[i],ls='--')
            ax.semilogx(K,e[:,i,2],color=c[i],ls=':')

        if xlim is None:
            if K[0]==0:
                xlim = [K[1],K[-1]]
            else:
                xlim = [K[0],K[-1]]
        ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_title(title)

        ax.legend()

    def plot_phase_speed(self,K=None,figsize=(7,7),
               xlim=None,ylim=None,xlabel=r'$|k|$ ($m^{-1}$)',
               ylabel='Speed $(m\,s^{-1})$',title="Phase Speed"):
        
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)

        # If we computed m with 2D K array then 
        # plot only along kx axis
        if K is not None:
            K = K ## This should be a 1D K
        elif len(self.evalues.shape)==3:
            evalues = self.evalues[:,0,:]
            K = self.k1[:,0]
        else:
            K = self.kr

        ax.semilogx(K[:],self.evalues[:,0].real,c='k')
        ax.semilogx(K[:],self.evalues[:,1].real,c='k')
        ax.semilogx(K[:],self.evalues[:,2].real,c='k')

        if xlim is None:
            if K[0]==0:
                xlim = [K[1],K[-1]]
            else:
                xlim = [K[0],K[-1]]
        ax.set_xlim(xlim)

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def plot_growth_rate(self,K=None,figsize=(7,7),
               xlim=None,ylim=None,xlabel=r'$|k|$ ($m^{-1}$)',
               ylabel='Growth rate $(\mathrm{day}^{-1})$',title="Growth Rate"):
        
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)
 
        # If we computed m with 2D K array then 
        # plot only along kx axis
        if K is not None:
            K = K ## This should be a 1D K
            growth_rate = self.growth_rate
        elif len(self.evalues.shape)==3:
            growth_rate = self.growth_rate[:,0,:]
            K = self.k1[:,0]
        else:
            K = self.kr
            growth_rate = self.growth_rate

        ax.semilogx(K[:],growth_rate[:,0],c='k')
        ax.semilogx(K[:],growth_rate[:,1],c='k')
        ax.semilogx(K[:],growth_rate[:,2],c='k')

        if xlim is None:
            if K[0]==0:
                xlim = [K[1],K[-1]]
            else:
                xlim = [K[0],K[-1]]
        ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    def plot_mode_grid(self,Ls,K=None, figsize=(15,3.5),lw=2,xlim=[-1.05,1.05],
                       Nz=300):
        if K is None:
            K = self.k1[:,0]
    
        Z = np.linspace(0,-self.H(0).sum(),Nz)
        
        Psi_mode = np.zeros((self.nz,len(Ls),Nz))
        ks = np.zeros_like(Ls)
        for i in range(self.nz):
            for j in range(len(Ls)):
                L = Ls[j]
                k = (2*np.pi)/L
                ik = np.argmin(np.abs(K-k)) # find the index closest of ik
                ks[j] = K[ik]
                Psi_mode[i,j,:] = correct_sign(np.asarray([self.Psi_modes(z,0,ik,k,i) for z in Z]))

        n_plots = 1
        m_plots = len(Ls)  # Number of columns
    
        fig, axs = plt.subplots(n_plots, m_plots, figsize=figsize)  # Adjust the figsize as needed
        for j in range(m_plots):
            ax = axs[j]
            
            ax.plot(Psi_mode[1,j,:],Z,c='tab:red',lw=lw)
            ax.plot(Psi_mode[2,j,:],Z,c='tab:purple',lw=lw)
            ax.plot(Psi_mode[0,j,:],Z,c='tab:blue',lw=lw)
            
            ax.set_xlim(xlim)
            ax.set_ylim([-self.H(0).sum(), 0])
            ax.axvline(x=0.0, lw=0.5, c='k')
            ax.set_title("L = {:.0f} km".format(2*np.pi/ks[j]/1e3))
        
        
        plt.tight_layout()  # Adjust spacing between subplots
        plt.show()

