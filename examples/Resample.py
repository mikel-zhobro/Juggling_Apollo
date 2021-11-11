import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from fractions import Fraction, gcd
import math

def fraction(r):
    f = Fraction(r).limit_denominator(1000)
    return f.numerator, f.denominator

def size_after_shrink(n_before, shrink_factor):
    n_after = int(round((n_before / shrink_factor)))
    return n_after
    
def shrink(x, shrink_factor, kaiser=True):
    """upfirdn performs a cascade of three operations:
       - Upsample the input data in the matrix xin by a factor of the integer p (inserting zeros)
       - FIR filter the upsampled signal data with the impulse response corresponding to Kaiser's Window with beta=5
       - Downsample the result by a factor of the integer q (throwing away samples)



    Args:
        x (list or np.array): Signal of length N x[n] for n = 0, .., N-1
        factor (float): shrinking factor x_s[n] = x(factor*n) the sampling factor: factor = downsampling/upsampling
    """
    down , up = fraction(shrink_factor)

    if kaiser:
        return signal.resample_poly(x, up=up, down=down)
    else:
        print(x.size)
        return signal.resample(x, size_after_shrink(x.size, shrink_factor) )



def main():
    x = np.linspace(0, 10, 20, endpoint=False)
    y = np.cos(-x**2/6.0)
    f_fft = shrink(y, 1.2, False)
    f_poly = shrink(y, 1.2)
    xnew = np.linspace(0, 10, size_after_shrink(20, 1.2), endpoint=False)




    plt.plot(xnew, f_fft, 'b.-', xnew, f_poly, 'r.-')
    plt.plot(x, y, 'ko-')
    plt.plot(10, y[0], 'bo', 10, 0., 'ro')  # boundaries
    plt.legend(['resample', 'resamp_poly', 'data'], loc='best')
    plt.show()

if __name__ == "__main__":
    main()