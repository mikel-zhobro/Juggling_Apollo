import matplotlib.pyplot as plt
import numpy as np
from ApolloInterface.Apollo_It import ApolloInterface
from ApolloILC.settings import alpha
from math import sin

def main():
    N = 2000
    dt = 0.004
    timesteps = np.arange(0.0, dt*N,dt)
    inputs = np.zeros((N, 7))
    inputs[:,0] = 0.3 * np.sin(timesteps)

    r_arm = ApolloInterface(r_arm=True)
    r_arm.go_to_posture_array([0.0, 0.0, -np.pi/4, np.pi/2, np.pi/2, np.pi/2, 0.0], 2000, False)
    # r_arm.go_to_posture_array([np.pi/4, 0.0, np.pi/4, np.pi/4, 3*np.pi/4, 3*np.pi/4, 0.0], 2000, False)

    poses, velocities, acc, _, u_vec = r_arm.apollo_run_one_iteration(dt, T=dt*len(timesteps), u=inputs)

    plt.figure()
    plt.plot(timesteps, poses[:, 0], label='angle')
    plt.plot(timesteps, velocities[:, 0], label='velocity')
    plt.plot(timesteps, u_vec[:, 0], label='des_velocity')
    plt.plot(timesteps, acc[:, 0], label='acc')
    plt.legend()
    plt.show()


def main2():

       # define the model and draw some data
    model = lambda x: x * np.sin(x)
    xdata = np.arange(0, 12, 0.2)
    ydata = model(xdata)

    plt.plot(xdata, ydata, '-', color='k', label="psi")
    plt.fill_between(xdata, ydata -2, ydata + 2, color='gray', alpha=0.2, label="psimin psimax")
    plt.legend()
    plt.show()


def main3():
    def simulate_vel(v_des, dt, N, N1, N2, alpha=alpha):
        vels = np.zeros(N)
        for i in range(N1, N2):
            vels[i] += alpha*dt*(v_des - vels[i-1])
        return vels

    N = 2000
    dt = 0.004
    timesteps = np.arange(0.0, dt*N,dt)
    v_des = 0.2
    N1 = 100
    N2 = 600
    inputs = np.zeros((N, 7))
    inputs[N1:N2,0] = 3.0 * np.sin(np.arange(N2-N1)*dt *4)
    # inputs[N1:N2,0] = v_des
    # inputs[N1:N2,3] = v_des


    r_arm = ApolloInterface(r_arm=True)

    r_arm.go_to_home_position([0.0, 0.0, -np.pi/4, np.pi/2, np.pi/2, np.pi/2, 0.0], 4000, False)
    # r_arm.go_to_posture_array([np.pi/4, 0.0, np.pi/4, np.pi/4, 3*np.pi/4, 3*np.pi/4, 0.0], 2000, False)

    poses, velocities, acc, _, u_vec = r_arm.apollo_run_one_iteration(dt, T=dt*len(timesteps), u=inputs, repetitions=1)

    plt.figure()
    plt.plot(velocities.squeeze()[:, 0], 'b', label="Measured Velocities")
    plt.plot(inputs[:, 0], 'r', label="Desired Velocities")
    # for a in [2.0,  8.0, 15.0]:
    #     plt.plot(simulate_vel(v_des, dt, N, N1, N2, a), '-', label="Simulated Velocities, alpha="+str(a))
    plt.legend()
    plt.show()
    print()



def main4():
    def save_matrices(A,B,C, file_name):
        with open(file_name, 'wb') as f:
            np.save(f, A)
            np.save(f, B)
            np.save(f, C)

    def load_matrices(file_name):
        with open(file_name, 'rb') as f:
            A = np.load(f)
            B = np.load(f)
            C = np.load(f)
        return (A,B,C)


    A = 0.2
    B = np.eye(4)
    C = np.eye(5)

    filename ='data/abc'
    save_matrices(A,B,C, filename)
    abc = load_matrices(filename)
    print(abc)


def main5():
    # Cell 0 - Preparation: load packages, set some basic options
    from scipy import signal
    from obspy.signal.invsim import cosine_taper
    from matplotlib import rcParams
    import numpy as np
    import matplotlib.pylab as plt
    plt.style.use('ggplot')
    rcParams['figure.figsize'] = 15, 3
    rcParams["figure.subplot.hspace"] = (0.8)
    rcParams["figure.figsize"] = (15, 9)
    rcParams["axes.labelsize"] = (15)
    rcParams["axes.titlesize"] = (20)
    rcParams["font.size"] = (12)

    def fourier_series_coeff(f, Nf, complex=False):
        """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

        Given a periodic, function f(t) with period T, this function returns the
        coefficients a0, {a1,a2,...},{b1,b2,...} such that:

        f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

        Parameters
        ----------
        f : the periodic function values
        Nf : the function will return the first N + 1 Fourier coeff.

        Returns
        -------
        a0 : float
        a,b : numpy float arrays describing respectively the cosine and sine coeff.
        """
        # In order for the nyquist theorem to be satisfied N_t > 2 N_f where N_t=f.size = T/dt
        y = np.fft.rfft(f, norm=None) / f.size * 2.0
        if complex:
            return y[:Nf]
        return y[0].real, y[1:].real[0:Nf], -y[1:].imag[0:Nf]

    def series_real_coeff(a0, a, b, t, T):
        """calculates the Fourier series with period T at times t,
        from the real coeff. a0,a,b"""
        tmp = np.ones_like(t) * a0 / 2.
        for k, (ak, bk) in enumerate(zip(a, b)):
            tmp += ak * np.cos(2 * np.pi * (k + 1) * t / T) + bk * np.sin(2 * np.pi * (k + 1) * t / T)
        return tmp

   # Cell 3: create periodic, discrete, finite signal
    # number of samples
    samp = 3000
    # sample rate
    dt = 0.001
    # period
    T = samp * dt
    # time axis
    t = np.linspace(0, T, samp)

    # number of coefficients (initial value: 100)
    Nf = samp//2
    Nf= 45
   # Cell 8: FFT of signal
    # number of sample points need to be the same as in cell 3
    print('samp =',samp,' Need to be the same as in cell 3.')
    # number of sample points need to be the same as in cell 3
    print('T =',T,' Need to be the same as in cell 3.')
    # percentage of taper applied to signal (initial: 0.1)
    taper_percentage = 0.1
    taper = cosine_taper(samp,taper_percentage)

    # signal
    sig = (t-1.5)**6 - 1.5**6  # second order polynomial
    # sig = np.sin(2*np.pi/T*t)  # sinusoid

    # tapered signal
    sig_ = sig * taper

    # fourie coeficients
    Fsig0 = fourier_series_coeff(sig, samp, True)
    Fsig = fourier_series_coeff(sig, Nf, True)
    Fsig_ = fourier_series_coeff(sig, Nf, True)

    # prepare plotting
    xf = np.linspace(0.0, 2.0/(samp*T), (samp//2)+1)



    # fourier coefs
    aa0, aa, bb = fourier_series_coeff(sig, samp)
    a0, a, b = fourier_series_coeff(sig, Nf)
    a0_, a_, b_ = fourier_series_coeff(sig_, Nf)

    # reconstruction
    gg = series_real_coeff(aa0, aa, bb, t, dt)
    g = series_real_coeff(a0, a, b, t, dt)
    g_ = series_real_coeff(a0_, a_, b_, t, dt)

    #plotting
    plt.subplot(311)
    plt.title('Time Domain')
    plt.plot(t, sig, linewidth=1, label='real')
    plt.plot(t, sig_, linewidth=1, label='tapered')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')

    plt.subplot(312)
    plt.title('Frequency Domain')
    plt.plot(np.abs(Fsig0), '.-', label='real')
    plt.plot(np.abs(Fsig), '.-', label='nf_nontapered')
    plt.plot(np.abs(Fsig_), '.-', label='nf_tapered')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.legend()

    plt.subplot(313)
    plt.title('Time Domain Reconstruction')
    plt.plot(t, sig, label='real')
    plt.plot(t, gg, label='real2')
    plt.plot(t, g, alpha=0.4, label='rec_direct')
    plt.plot(t, g_, alpha=0.5, label='rec_tapered')
    # plt.xlim(0, 0.04)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main5()