# FirFilter1D.py
# Simple FIR filter class used to perform 1-D filtering of time series data.

# System packages.
import numpy as np

# ----------------------------------------------------------------------
# Globals.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Functions.
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# Classes.
# ----------------------------------------------------------------------
class FirFilter1D(object):
    """
    Simple FIR filter class used to perform 1-D filtering of time series data.
    """
    
    # ------------------------------------------------------------------
    def __init__(self, dtype=np.float):
        """
        Class constructor.
        Parameters:
            dtype : (numpy.dtype) Optional data type to use for filter operations.
        Returns:
            None
        """
        self.dtype = dtype

        # Initialize an empty coefficient array.
        self.coeffs = np.array([], dtype=self.dtype)
    
    # ------------------------------------------------------------------
    def coeff_len(self):
        """
        Return the length of the coefficient array
        """
        return self.coeffs.size

    # ------------------------------------------------------------------
    def load_coeffs_list(self, coeffs_list):
        """
        Load a 1-D array of filter coefficients from a Python list.
        
        Parameters:
            coeffs_list : (numeric) Python list containing coefficients to load.
        Returns:
            Nothing is returned; coefficients are loaded into self.coeffs
            as a 1-D numpy array.
        """            
        self.coeffs = np.array(coeffs_list, dtype=self.dtype)
        
    # ------------------------------------------------------------------
    def load_coeffs_file(self, filename):
        """
        Load a 1-D array of filter coefficients from a text file.
        
        The file should be organized as one coefficient per line.
        Blank lines and lines starting with a hash (#) are ignored.
        
        Parameters:
            filename : (str) Text file containing coefficients to load.
        Returns:
            Nothing is returned; coefficients are loaded into self.coeffs
            as a 1-D numpy array.
        """
        coeffs_list = []
        
        try:
            with open(filename, 'r') as file:
                for line in file:
                    ls = line.strip()
                    if (len(ls) == 0):
                        continue
                    elif ls.startswith('#'):
                        continue
                    coeffs_list.append(float(ls))

        except Exception as err:
            print('Error loading coefficients from {}: {}'.format(filename, err))
            coeffs_list.clear()  # Empty the list if an error occurs
            
        self.load_coeffs_list(coeffs_list)

    # ------------------------------------------------------------------
    def filter(self, x, pad=False):
        """
        Run the FIR filter on a data array.
        
        Parameters:
            x : (numpy array) Array containing data to filter.
            
            pad : (bool) Optional padding. If True, adds zeros to the 
            beginning of x to keep the length of y equal to the length of x.
            
            x array length (including optional pad) must be greater than 
            or equal to the length of the coefficient array.
            
        Returns:
            y : (numpy array) Filtered data array.
            y array length = x length + pad - coefficient length + 1
        """
        flt_len = self.coeffs.shape[0]
        if (flt_len == 0):
            raise RuntimeError('Empty coefficient array.')
        
        if pad:
            x = np.concatenate((np.zeros(flt_len-1), x), dtype=self.dtype)
        x_len = x.shape[0]
        if (x_len < flt_len):
            raise RuntimeError('Input array too small.')
        
        y = []
        for i in range(x_len - flt_len + 1):
            y.append(np.dot(x[i:i+flt_len], self.coeffs.T))
        return np.asarray(y, dtype=self.dtype)


# ----------------------------------------------------------------------
# Main program used as an example for test and debug.
# ----------------------------------------------------------------------
if __name__ == '__main__':
    print('FirFilter test program not implemented.')
    
