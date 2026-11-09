"""Mean-transverse versus parallel energy in a cyclotron-frequency band."""
import numpy as np
from scipy.signal import periodogram

ENERGY_RATIO_MIN = 5.0

def cyclotron_band_energy(bx, by, bz, fci_hz, *, fs=1.0, full_event=False):
    """Three half-length Hann estimates, or one full-event Hann if requested.

    Rectangular PSD-bin integration is used for either estimator.

    fci is the mean of positive instantaneous frequencies; sigma is population SD.
    No background subtraction or frequency interpolation is performed.
    """
    arrays=[np.asarray(v,dtype=float).ravel() for v in (bx,by,bz,fci_hz)]
    n=len(arrays[0])
    if n<32 or any(len(v)!=n for v in arrays) or not all(np.isfinite(v).all() for v in arrays):
        raise ValueError('Band energy needs >=32 matched finite samples')
    fc=arrays[3]
    if np.any(fc<=0):raise ValueError('Cyclotron frequencies must be positive')
    center=float(fc.mean());sigma=float(fc.std(ddof=0))
    low=max(0.0,.8*(center-sigma));high=1.2*(center+sigma)
    length=n if full_event else n//2;estimates=[]
    for off in ((0,) if full_event else (0,(n-length)//2,n-length)):
        ps=[]
        for v in arrays[:3]:
            freq,p=periodogram(v[off:off+length],fs=fs,window=np.hanning(length),detrend='constant',scaling='density')
            ps.append(p)
        estimates.append([(ps[0]+ps[1])/2,ps[2]])
    perp,parallel=np.mean(estimates,axis=0)
    mask=(freq>0)&(freq>=low)&(freq<=high)
    if not mask.any():raise ValueError('Cyclotron band contains no PSD bins')
    df=freq[1]-freq[0];ip=float(perp[mask].sum()*df);iz=float(parallel[mask].sum()*df)
    # Undefined or zero parallel energy fails closed instead of creating infinity.
    ratio=ip/iz if iz>0 and np.isfinite(ip/iz) else None
    scale=1e-18/(2*4*np.pi*1e-7)
    return dict(energy_ratio=ratio,energy_perp_J_m3=ip*scale,energy_parallel_J_m3=iz*scale,
                energy_band_low_hz=low,energy_band_high_hz=high,energy_fci_hz=center,
                energy_sigma_fci_hz=sigma,energy_band_bin_count=int(mask.sum()))
