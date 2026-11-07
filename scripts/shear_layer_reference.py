#!/usr/bin/env python3
"""
Pseudo-spectral reference solution for the doubly periodic shear layer
(Example 2 of the BDF2-IMEX/CIP paper) and the comparison metrics for the
finite element solutions sampled by parMooN_tnse2d_bdf2imex_cip
(bdf2_sample_grid: N, file <output_basename>_sample.txt).

Reference: vorticity form on [0,1]^2 with M x M Fourier modes, 2/3 dealiasing,
exact integration of the viscous term (integrating factor) and RK4 for the
nonlinear term.

  python3 shear_layer_reference.py --modes 512 --nu 1e-4 --t-end 1 --out ref_t1.npz
  python3 shear_layer_reference.py --compare ref_t1.npz sv_sample.txt th_sample.txt

Metrics (as in the paper): TV ratio TV(omega_h)/TV(omega_ref), vorticity
over/undershoots beyond the reference range normalized by |omega_ref|_inf,
velocity over/undershoots, relative L2 vorticity error; all evaluated on the
sample grid (the reference is restricted to the sample points by Fourier
interpolation).
"""
import argparse
import numpy as np


def initial_condition(x, y, rho=80., delta=0.05):
    u1 = np.tanh(rho*(0.25 - np.abs(y - 0.5)))
    u2 = delta*np.sin(2*np.pi*(x + 0.25))
    return u1, u2


def spectral_reference(M, nu, t_end, dt=None, cfl=0.25):
    k = 2*np.pi*np.fft.fftfreq(M, d=1./M)
    kx, ky = np.meshgrid(k, k, indexing="xy")
    k2 = kx**2 + ky**2
    k2[0, 0] = 1.
    # dealiasing mask (2/3 rule)
    kmax = M/3*2*np.pi
    mask = (np.abs(kx) < kmax) & (np.abs(ky) < kmax)
    x = (np.arange(M) + 0.5)/M
    X, Y = np.meshgrid(x, x, indexing="xy")
    u1, u2 = initial_condition(X, Y)
    w = np.gradient(u2, x, axis=1) - np.gradient(u1, x, axis=0)  # crude start
    # exact spectral vorticity of the initial condition
    u1h, u2h = np.fft.fft2(u1), np.fft.fft2(u2)
    wh = 1j*kx*u2h - 1j*ky*u1h
    wh *= mask

    def velocity(wh):
        psih = wh/k2
        psih[0, 0] = 0.
        u1h = 1j*ky*psih
        u2h = -1j*kx*psih
        return np.real(np.fft.ifft2(u1h)), np.real(np.fft.ifft2(u2h))

    def nonlinear(wh):
        u1, u2 = velocity(wh)
        w = np.real(np.fft.ifft2(wh))
        wx = np.real(np.fft.ifft2(1j*kx*wh))
        wy = np.real(np.fft.ifft2(1j*ky*wh))
        return -np.fft.fft2(u1*wx + u2*wy)*mask

    u1, u2 = velocity(wh)
    umax = max(np.abs(u1).max(), np.abs(u2).max())
    if dt is None:
        dt = cfl/(M*umax)
    n = int(np.ceil(t_end/dt))
    dt = t_end/n
    E = np.exp(-nu*k2*dt)
    E2 = np.exp(-nu*k2*dt/2)
    t = 0.
    for step in range(n):
        # integrating factor RK4
        a = nonlinear(wh)
        b = nonlinear(E2*(wh + dt/2*a))
        c = nonlinear(E2*wh + dt/2*b)
        d = nonlinear(E*wh + dt*E2*c)
        wh = E*wh + dt/6*(E*a + 2*E2*(b + c) + d)
        t += dt
    u1, u2 = velocity(wh)
    w = np.real(np.fft.ifft2(wh))
    return x, u1, u2, w, wh, n, dt


def fourier_sample(wh, pts_x, pts_y):
    """evaluate the Fourier series with coefficients wh at points (vectorised
    over a grid of sample points)"""
    M = wh.shape[0]
    k = 2*np.pi*np.fft.fftfreq(M, d=1./M)
    ex = np.exp(1j*np.outer(pts_x, k))  # (nx, M)
    ey = np.exp(1j*np.outer(pts_y, k))  # (ny, M)
    # w(x,y) = sum_kx sum_ky wh[ky,kx] e^{i kx x} e^{i ky y} / M^2
    return np.real(ey @ wh @ ex.T)/M**2


def tv(f, h):
    return (np.abs(np.roll(f, -1, axis=1) - f).sum() + np.abs(np.roll(f, -1, axis=0) - f).sum())*h


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modes", type=int, default=512)
    ap.add_argument("--nu", type=float, default=1e-4)
    ap.add_argument("--t-end", type=float, default=1.)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--out", default="shear_reference.npz")
    ap.add_argument("--compare", nargs="*", default=None,
                    help="reference .npz followed by sample files of the FE runs")
    args = ap.parse_args()

    if args.compare:
        ref = np.load(args.compare[0])
        wh = ref["wh"]
        for f in args.compare[1:]:
            data = np.loadtxt(f)
            N = int(round(np.sqrt(data.shape[0])))
            x = data[:, 0].reshape(N, N)[0, :]
            y = data[:, 1].reshape(N, N)[:, 0]
            u1 = data[:, 2].reshape(N, N)
            u2 = data[:, 3].reshape(N, N)
            om = data[:, 4].reshape(N, N)
            dv = data[:, 5].reshape(N, N)
            om_ref = fourier_sample(wh, x, y)
            u1_ref = fourier_sample(ref["u1h"], x, y)
            u2_ref = fourier_sample(ref["u2h"], x, y)
            h = 1./N
            winf = np.abs(om_ref).max()
            over = max(0., om.max() - om_ref.max())/winf
            under = max(0., om_ref.min() - om.min())/winf
            uinf = max(np.abs(u1_ref).max(), np.abs(u2_ref).max())
            uover = max(0., max(u1.max() - u1_ref.max(), u2.max() - u2_ref.max()))/uinf
            uunder = max(0., max(u1_ref.min() - u1.min(), u2_ref.min() - u2.min()))/uinf
            werr = np.sqrt(((om - om_ref)**2).sum())/np.sqrt((om_ref**2).sum())
            divn = np.sqrt((dv**2).sum()*h*h)
            print(f"{f}: TV ratio {tv(om, h)/tv(om_ref, h):.2f}, omega over/under "
                  f"{over:.2f}/{under:.2f}, u over/under {uover:.2f}/{uunder:.2f}, "
                  f"omega error {werr:.2f}, ||div u_h|| {divn:.1e}")
        return

    x, u1, u2, w, wh, n, dt = spectral_reference(args.modes, args.nu, args.t_end, args.dt)
    u1h, u2h = np.fft.fft2(u1), np.fft.fft2(u2)
    np.savez(args.out, x=x, u1=u1, u2=u2, omega=w, wh=wh, u1h=u1h, u2h=u2h,
             nu=args.nu, t_end=args.t_end)
    print(f"reference with {args.modes}^2 modes, {n} steps of dt = {dt:.2e}; "
          f"vorticity range [{w.min():.3f}, {w.max():.3f}]; written to {args.out}")


if __name__ == "__main__":
    main()
