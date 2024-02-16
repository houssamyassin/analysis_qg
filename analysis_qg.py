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


def plot_q(x,y,q,size=5,title=None,save=0,number=0,path='',
           cb=1,vabs=None,vmin=None,vmax=None,cmap='binary_r',
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
        im = ax.pcolormesh(x,y,q.squeeze(),cmap=cmap,shading='auto',norm=norm)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        fig.colorbar(im, cax=cax, orientation='vertical')
    
    else:
        fig = plt.figure(figsize=(size,size))
        ax = plt.subplot(111)
        #im = ax.imshow(m.q.squeeze(),cmap='binary_r')
        im = ax.pcolormesh(x,y,q.squeeze(),cmap=cmap,shading='auto',norm=norm)

    ax.set_title(title)

    ax.set_xticks([])
    ax.set_yticks([])
    
    if xlims:
        ax.set_xlim(xlims)
    if ylims:
        ax.set_ylim(ylims)
    
    if save:
        name = str(number).zfill(5)
        plt.savefig(path+name+'.png',bbox_inches='tight')
    else:
        plt.show()
        
    plt.close()
            
    return im



def plot_3q(x, y, q, size=5, title=None, save=0, number=0, path='', 
            cb=1, cb_label=None, vabs=None, vmin=None, vmax=None, 
            cmap='binary_r', xlims=None, ylims=None,
            fontsize=25):
    fig, axes = plt.subplots(1, 3, figsize=(size * 3, size),gridspec_kw={'wspace': 0.05})
    plt.subplots_adjust(right=0.85)
    
    for i in range(3):
        ax = axes[i]

        if vabs:
            vmin = -vabs
            vmax = vabs
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
        elif vmax:
            norm = plt.Normalize(vmin=0, vmax=vmax)
        else:
            norm = None

        im = ax.pcolormesh(x, y, q[:, :, i].squeeze(), cmap=cmap, shading='auto', norm=norm)
        
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
        # Adjust the position of the color bar
        cax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
        cbar = fig.colorbar(im, cax=cax, orientation='vertical')
    if cb_label:
        cbar.set_label(cb_label, rotation=270, labelpad=15)

    fig.suptitle(title,fontsize=fontsize)

    if save:
        name = str(number).zfill(5)
        save_path = os.path.join(path,name + '.png')
        plt.savefig(save_path, bbox_inches='tight')
    #else:
    #    plt.show()

    plt.close()

    return fig

#quantity can be 'q', 'p', 't'
#dims = (nx,ny,nz) for 'q','t,' or '1' 
def load_data(case_dir,quantity,dims,dtype=np.float64):
    if quantity not in ('q','p','t'):
        print("quantity must be 'q','p', or 't'")
        return
    if quantity == 't': dims = (1,)

    file_pattern = quantity+'.*.dat'

    # Get a sorted list of file names
    path = os.path.join(case_dir,file_pattern)
    file_list = sorted(glob.glob(path), key=lambda x: int(x.split('.')[1]))
    # Initialize an empty list to store the reshaped arrays
    reshaped_arrays = []

    # Iterate over the files and read/reshape each array
    for file_name in file_list[1:]:
        data = np.fromfile(file_name, dtype=dtype)  # Adjust dtype if needed
        reshaped_data = data.reshape(dims, order='F')  # Use 'F' for Fortran order
        reshaped_arrays.append(reshaped_data)

    # Stack the reshaped arrays along a new axis to create a 4D array
    result = np.stack(reshaped_arrays, axis=len(dims))

    # Now 'result' contains the stacked and reshaped array with shape [nx, ny, nz, nt]
    if len(file_list)!=result.shape[-1]:
        print("Warning: number of files does not match time dimension")
    if quantity == 't' and len(result)>0: result = result[0]

    return result


def create_e_matrix(k1,sigma,H,dims):
    nx = dims[0]; ny=dims[1]; nz=dims[2]
    # Initialize arrays
    C0 = np.cosh(sigma[0] * H[0] * k1)
    C1 = np.cosh(sigma[1] * H[1] * k1)
    T0 = np.tanh(sigma[0] * H[0] * k1)
    T1 = np.tanh(sigma[1] * H[1] * k1)

    # Initialize e array
    if len(k1.shape)==1:
        e = np.zeros((len(k1), nz, nz), dtype=np.float64)
        T0[0] = 1.0 # Avoids divide by zero
        T1[0] = 1.0  # Avoids divide by zero
    if len(k1.shape)==2:
        e = np.zeros((nx//2 + 1, ny, nz, nz), dtype=np.float64)
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

    return e

def create_m_matrix(K,sigma,H,dims):
    nx = dims[0]; ny=dims[1]; nz=dims[2]
    # Initialize arrays
    T0 = np.tanh(sigma[0] * H[0] * K)
    T1 = np.tanh(sigma[1] * H[1] * K)

    if len(K.shape)==1:
        m = np.zeros((len(K),nz),dtype=np.float64)
        T0[0] = 1.0 # Avoids divide by zero
        T1[0] = 1.0 # Avoids divide by zero
    if len(K.shape)==2:
        m = np.zeros((nx//2 + 1, ny,nz), dtype=np.float64)
        T0[0, 0] = 1.0  # Avoids divide by zero
        T1[0, 0] = 1.0  # Avoids divide by zero
    # Compute m values
    m[..., 0] = (K / sigma[0]) * (sigma[1] * T0 + sigma[0] * T1) / (sigma[1] + sigma[0] * T0 * T1)
    m[..., 1] = K * ((T0 / sigma[0]) + (T1 / sigma[1]))
    m[..., 2] = K * (sigma[1] * T0 + sigma[0] * T1) / (sigma[1] * (sigma[0] + sigma[1] * T0 * T1))

    # Avoid divide by zero for m(1, 1, :)
    if len(K.shape)==2:
        m[0,0,:] = 1.0
    return m

def create_L_matrix(K,sigma,H,dims):
    nx = dims[0]; ny=dims[1]; nz=dims[2]
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
        L = np.zeros((len(K), nz, nz), dtype=np.float64)
    if len(K.shape)==2:
        L = np.zeros((nx//2 + 1, ny, nz, nz), dtype=np.float64)
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

    return L

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

class QG3:
    def __init__(self, H=None, U=None, case_dir=None,data=None):
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
        self.H = H            
        self.U = U           

        self.dims = (self.nx, self.ny, self.nz)  
        self.kdims = (self.nk, self.ny, self.nz) 

        self.Q = None          
        self.P = None          
        self.T = None         
        self.nt = 0

        self.E = None          

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

        self.evalues = None
        self.evectors = None
        
        self.print_parameters()
        self.initialize_global_variables()

    def initialize_global_variables(self):
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
        self.kr = np.arange(1 * self.dkr / 2., self.kmax + self.dkr, self.dkr)

        print("Initialized the following variables:")
        print("\tx, y, x_grid, y_grid")
        print("\tkx, ky, k0, k1, k2, k8")
        print("\tnkx, nky, kmax, dk, dl, dkr, kr")
        print("\tcase_dir")
        print("Case Directory: ", self.case_dir)
  
    def read_parameters(self):
        file_path = self.case_dir+"/QG3/parameters.f90"
        selected_parameters = ['Lx', 'Ly', 'nx', 'ny', 'nz', 
                           'f0', 'g', 'rho0', 'A0', 'A2',
                           'A8', 'A20', 'r0', 'C_d']
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


    
    ###################################################################
    ### Loading data ##################################################
    ################################################################### 

    def load_data(self,quantity,first_n=None,last_n=None,skip=None,
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
        if quantity not in ('q','p','t'):
            print("quantity must be 'q','p', or 't'")
            return
        if quantity == 't': dims = (1,)
        else: dims = self.dims

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
        self.T = self.load_data('t',desc='Loading time',**kwargs)
        self.nt = len(self.T)

    def set_T(self,T):
        self.T = T
        self.nt = len(self.T)

    ###################################################################
    ### Computing Operators ############################################
    ################################################################### 

    def create_m_matrix(self,K,H):
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
        
        return m

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
            
    def create_e_matrix(self,K,H):
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

        return e
    
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

    def create_L_matrix(self, K, H):
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
        if (self.Q is None) or (self.P is None):
            print("Load Q and P data first. Not computing energy.")
            return
        Eden = -0.5*self.Q*self.P/(self.Lx*self.Ly)
        intx_Eden = np.trapz(Eden,x=self.x,axis=1)
        inty_intx_Eden = np.trapz(intx_Eden,x=self.y,axis=1)
        self.E = inty_intx_Eden.sum(axis=1)

    def fft(self,field):
        return npfft.rfftn(field,axes=(-2,-3))

    def ifft(self,field):
        return npfft.irfftn(field,axes=(-2,-3))

    def compute_Qh(self):
        self.Qh = self.fft(self.Q)

    def compute_Ph(self):
        self.Ph = self.fft(self.P)

    # phi_hat are the streamfunctions for each layer
    def compute_Phi(self):
        if len(self.m.shape) != 4:
            print("First recompute operators with a 2D wavenumber field.")
            return
        if (self.Qh is None):
            print("Compute FFT of Q first!")
            return
        if (self.m is None):
            print("Compute m first!")
            return

        Phih = -self.Qh/self.m
        self.Phi = self.ifft(Phih)

    # For testing only. Reconstruct Psi from Phi.
    def compute_P_from_phi(self):
        Phih = -self.Qh/self.m
        Ph = np.zeros_like(Phih,dtype=np.complex128)
        for t in range(self.nt):
            for i in range(self.nz):
                for ii in range(self.nz):
                    Ph[t,:,:,i] += self.e[t,:,:,i,ii]*Phih[t,:,:,ii]
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
        isL = self.L is  None
        isQh = self.Qh is  None
        if isL or isQh:
            print("Compute L and Q first.")
            return
        
        inv_L = np.zeros_like(self.L)
        for i in tqdm(range(self.nt),desc='Inverting L'):
            inv_L[i] = self.invert_matrix(self.L[i])

        PLh = np.zeros_like(self.Qh)
        for t in range(self.nt):
            for i in range(self.nkx):
                for j in range(self.nky):
                    PLh[t,i,j] = inv_L[t,i,j]@self.Qh[t,i,j]
        return self.ifft(PLh)

    def compute_Q_from_P(self):
        isL = self.L is  None
        isPh = self.Ph is  None
        if isL or isPh:
            print("Compute L and P first.")
            return

        QLh = np.zeros_like(self.Ph)
        for t in range(self.nt):
            for i in range(self.nkx):
                for j in range(self.nky):
                    QLh[t,i,j] = self.L[t,i,j]@self.Ph[t,i,j]
        return self.ifft(QLh)

    def compute_U(self):
        if self.Ph is None:
            print("Compute Ph first.")
            return
        Uh = np.zeros((self.nt,self.nkx,self.nky,self.nz),dtype=np.complex128)
        for t in tqdm(range(self.nt),desc='Computing U'):
            for z in range(self.nz):
                Uh[t,:,:,z] =  -1j*self.ky*self.Ph[t,:,:,z]
        return self.ifft(Uh)

    def compute_V(self):
        if self.Ph is None:
            print("Compute Ph first.")
            return
        Vh = np.zeros((self.nt,self.nkx,self.nky,self.nz),dtype=np.complex128)
        for t in tqdm(range(self.nt),desc='Computing V'):
            for z in range(self.nz):
                Vh[t,:,:,z] =  1j*self.kx*self.Ph[t,:,:,z]
        return self.ifft(Vh)

    def compute_speed(self,U=None,V=None):
        if self.Ph is None:
            print("Compute Ph first.")
            return
        if U is None or V is None:
            U = self.compute_U()
            V = self.compute_V()
        self.speed = np.sqrt(U**2+V**2)


    ### Instability calculations

    def create_instability_matrix(self,t,K=None):
        if K is None:
            K = self.k1

        Lambda = self.Lambda
        sigma = self.sigma
        U = self.U(t)
        Qy = [Lambda[0]/sigma[0]**2,Lambda[1]/sigma[1]**2-Lambda[0]/sigma[0]**2,-Lambda[1]/sigma[1]**2]
        
        if len(K.shape)==1:
            inv_L = self.invert_matrix(self.L[t])
            A =  np.zeros((len(K), self.nz, self.nz), dtype=np.float64)
            for i in tqdm(range(len(K)),desc='Computing instability matrix'):
                U_matrix = np.diag(U)
                Qy_matrix = np.diag(Qy)
                A[i,:,:] = U_matrix + np.dot(Qy_matrix,inv_L[i])
        elif len(K.shape)==2:
            A = np.zeros((self.nkx, self.nky, self.nz, self.nz), dtype=np.float64)
            for i in tqdm(range(self.nkx),desc='Computing instability matrix'):
                inv_L = self.invert_matrix(self.L[t,i])
                for j in range(self.nky):
                    U_matrix = np.diag(U)
                    Qy_matrix = np.diag(Qy)
                    A[i,j,:,:] = U_matrix + np.dot(Qy_matrix,inv_L[j])
        else:
            print("K must either one or two dimensional!")
            return None
        return A

    def solve_instability(self,t,K=None):
        if K is None:
            K = self.k1
        A = self.create_instability_matrix(t,K=K)
        evalues, evectors = np.linalg.eig(A)

        unstable = np.any(np.abs(np.imag(evalues)) > 1e-10)
        print("Unstable?: ", unstable)
        self.evalues = evalues 
        self.evectors = evectors
        
        self.growth_rate = K[...,np.newaxis]*evalues.imag*3600*24 # in units of day^{-1}

    #### Compute modes  ## 

    # Compute the eigenvalues and eigenvectors of -L for each time.
    # The eigenvalue problem has the form -L@e=lambda*e where e 
    # is the eigenvector.
    # -L is the energy matrix in the total streamfunction coordinates
    # so that -(1/2)<psi,L*psi> gives the energy.
    def compute_modes(self, K=None):
        if K is None:
            K = self.k1
            matrix = np.zeros((self.nt, self.nkx, self.nky, self.nz, self.nz))
            self.evalues = np.zeros((self.nt, self.nkx, self.nky, self.nz))
            self.evectors = np.zeros((self.nt, self.nkx, self.nky, self.nz, self.nz))
        else:    
            matrix = np.zeros((self.nt, len(K), self.nz, self.nz))
            self.evalues = np.zeros((self.nt, len(K), self.nz))
            self.evectors = np.zeros((self.nt, len(K), self.nz, self.nz))
        
        for t in tqdm(range(self.nt), desc='Computing modes'):
            eigenvalues, eigenvectors = np.linalg.eig(-self.L[t])
            sorted_indices = np.argsort(eigenvalues)  # Get indices to sort eigenvalues

            if len(K.shape)==1:
                # broadcasting magic... can also be done analogously to the len(K.shape)==2 case.
                self.evalues[t,:] = eigenvalues[np.arange(len(eigenvalues))[:, None], sorted_indices] 
                for j in range(len(K)):
                    # Rearrange eigenvectors accordingly
                    self.evectors[t,j,:,:] = eigenvectors[j][:, sorted_indices[j]]  
            elif len(K.shape)==2:
                for i in range(self.nkx):
                    for j in range(self.nky):
                        self.evalues[t,i,j,:] = eigenvalues[i,j][sorted_indices[i,j]]
                        self.evectors[t,i,j,:,:] = eigenvectors[i,j][:,sorted_indices[i,j]]

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
    def Psi(self,z,t,k,n):
        # Define vertical coordinate of the surfaces
        z0 = 0
        z1 = -self.H(t)[0]
        z2 = -self.H(t)[0] - self.H(t)[1]
        
        sigma = self.sigma
        C0 = np.cosh(k*sigma[0]*(z0-z1))
        S0 = np.sinh(k*sigma[0]*(z0-z1))
        C1 = np.cosh(k*sigma[1]*(z1-z2))
        S1 = np.sinh(k*sigma[1]*(z1-z2))
        Cth0 = C0/S0
        Cth1 = C1/S1 
        T0 = S0/C0
        T1 = S1/C1

        # These functions define the values of the coefficents
        # a,b for the vertical structure in the case of a surface
        # anomaly (Psi0), a mixed-layer anomaly (Psi1), and 
        # pycnocline anomaly (Psi2).
        # They also return the argument of sinh and cosh.
        def Psi0_nz3():
            if z0 >= z > z1: #l=0
                a = 1
                b = (sigma[1]*C1 + sigma[0]*Cth0*S1)/(sigma[1]*C1*Cth0 + sigma[0]*S1)
                arg = sigma[0]*k*(z-z0)
            else: #z1 >= z > z2, l=1
                a = sigma[1]/(sigma[1]*C0 + sigma[0]*S0*T1)
                b = sigma[1]/(sigma[1]*C0*Cth1 + sigma[0]*S0)
                arg = sigma[1]*k*(z-z1)
            return a,b,arg

        def Psi1_nz3():
            if z0 >= z > z1: #l0
                a = 1/C0
                b = 0
                arg = sigma[0]*k*(z-z0)
            else: #z1 >= z > z2, l=1
                a = 1
                b = T1
                arg = sigma[1]*k*(z-z1)
            return a,b,arg

        def Psi2_nz3():
            if z0 >= z > z1: #l0
                a = sigma[0]/(sigma[0]*C0*C1 + sigma[1]*S0*S1)
                b = 0
                arg = sigma[0]*k*(z-z0)
            else: #z1 >= z > z2, l=1
                a = sigma[0]/(sigma[0]*C1 + sigma[1]*S1*T0)
                b = - sigma[1]*(1/S1)/(sigma[1] + sigma[0]*Cth0*Cth1)
                arg = sigma[1]*k*(z-z1)
            return a,b,arg
            
        if self.nz==3:
            if n==0:
                a,b,arg = Psi0_nz3()
            if n==1:
                a,b,arg = Psi1_nz3()
            if n==2:
                a,b,arg = Psi2_nz3()
        else:
            print("This function require nz==3!")
            return
        
        return a*np.cosh(arg) + b*np.sinh(arg)

    # t is the index of the time axis
    # k is the wavenumber
    # evector[:,i] is the eigenvector corresponding to the eigenvalue evalues[i]
    def Psi_modes(self,z,t,ik,k,n):
        if self.evectors is None:
            print("Compute modes first!")
            return

        if len(self.evectors.shape)==4: # 1D wavenumber axis
            evector = self.evectors[t,ik,:,n]
        elif len(self.evectors.shape)==5: # 2D wavenumber axis
            evector = self.evectors[t,ik,0,:,n]
        
        inv_e = np.linalg.inv(self.e[t,ik])
        phi = inv_e@evector
    
        return np.asarray([phi[i]*self.Psi(z,t,k,i) for i in range(self.nz)]).sum()

    
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
            name = str(number).zfill(5)
            plt.savefig(path+name+'.png',bbox_inches='tight')
        else:
            plt.show()
            
        plt.close()
                
        return im

    # t and z in the time and z index
    def plot_q(self,t,z,**kwargs):
        field = self.Q[t,:,:,z]
        self.plot(field,**kwargs)
    
    def plot_p(self,t,z,**kwargs):
        field = self.P[t,:,:,z]
        self.plot(field,**kwargs)
        
    def plot_all(self, field, size=5, title=None, save=0, number=0, path='', 
                cb=1, cb_label=None, vabs=None, vmin=None, vmax=None, 
                cmap=cc.m_CET_D1, xlims=None, ylims=None, vabs_arr=None,
                fontsize=25):
        
        fig, axes = plt.subplots(1, 3, figsize=(size * 3, size),gridspec_kw={'wspace': 0.05})
        
        for i in range(3):
            if vabs_arr:
                vmin_i = -vabs_arr[i]
                vmax_i = vabs_arr[i]
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
            name = str(number).zfill(5)
            save_path = os.path.join(path,name + '.png')
            plt.savefig(save_path, bbox_inches='tight')
        else:
            plt.show()

        plt.close()

        return fig
    
    def plot_all_Q(self,t,**kwargs):
        field = self.Q[t,:,:,:]
        self.plot_all(field,**kwargs)

    def plot_all_P(self,t,**kwargs):
        field = self.P[t,:,:,:]
        self.plot_all(field,**kwargs)

    def plot_all_Phi(self,t,**kwargs):
        field = self.Phi[t,:,:,:]
        self.plot_all(field,**kwargs)

    def plot_energy(self,figsize=(10,5),c='k',lw=1.5,lw_zero=0.5,
                    title='Energy',xlabel='Time (days)',ylabel=r'Energy $(m^3s^4)$'):
        
        plt.figure(figsize=figsize)
        ax = plt.subplot(111)
        day = 24*3600
        
        ax.plot(self.T/day,self.E,c=c,lw=lw)


        ax.axhline(y=0,c='k',lw=lw_zero)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    
    def plot_inversion(self,t,m,K=None,which=None,figsize=(10,5),c=None,labels=None,
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
        ax.loglog(K,self.H(t).sum()*K**2,c='k',ls=':',lw='1')

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

    def plot_m(self,t,**kwargs):
        self.plot_inversion(t,self.m[t],**kwargs)

    def plot_evalues(self,t,**kwargs):
        self.plot_inversion(t,self.evalues[t],**kwargs)
    
    def plot_e(self,t,K=None,which=None,figsize=(10,5),c=None,labels=None,
               xlim=None,ylim=[0,1],xlabel=r'$|k|$ ($m^{-1}$)',
               ylabel='',title=r'Interaction matrix'):
        
        if which is None:
            which = [0,1,2]
        if c is None:
            c = ['tab:red','tab:purple','tab:blue']
        if labels is None:
            labels = ['Surface','Mixed Layer','Pycnocline']

        e = self.e[t]
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
