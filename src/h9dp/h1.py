"""Conditional environmental water-vapour spectra at the inner cloud's B16 temperature."""

import numpy as np

from h9dp.d1 import locate


def background_curve(environment, temperatures):
    """Median water-vapour BT at equal B16 quantiles, with saturated end extrapolation."""
    finite = np.isfinite(environment).all(axis=0)
    data = environment[:,finite]
    if data.shape[1] < 120:
        return np.full((3,*temperatures.shape), np.nan)
    order = np.argsort(data[3], kind="stable")
    # Preserve co-location by summarizing the same pixels in all channels.
    bands = np.stack([np.median(data[:,indices],axis=1) for indices in np.array_split(order,18)],axis=1)
    values, inverse = np.unique(bands[3],return_inverse=True)
    return np.stack([np.interp(temperatures,values,np.bincount(inverse,weights=b)/np.bincount(inverse))
                     for b in bands[:3]])


def signature(image, mask, center):
    eye, _ = locate(image,mask,tuple(np.rint(np.asarray(center)[::-1]-0.5).astype(int)))
    dy,dx = np.mgrid[-28:29,-28:29]
    yy,xx = eye[0]+dy,eye[1]+dx
    yn,xn = yy.clip(0,mask.shape[0]-1),xx.clip(0,mask.shape[1]-1)
    local = image[:,yn,xn].astype(float)*0.01+150
    local[:,~(mask[yn,xn] & (yn==yy) & (xn==xx))] = np.nan
    gy,gx = np.ogrid[:mask.shape[0]:3,:mask.shape[1]:3]
    radius2 = (gy-eye[0])**2+(gx-eye[1])**2
    env = image[:,::3,::3].astype(float)*0.01+150
    good = mask[::3,::3]
    def area(lo,hi):
        return env[:,good & (radius2>=lo**2) & (radius2<hi**2)]
    inner,outer = area(35,110),area(110,230)
    profile = []
    for surround in (inner,outer):
        anomaly = local[:3] - background_curve(surround,local[3])
        profile.append(anomaly)
    features = np.concatenate((local,local[1:]-local[:-1],*profile))
    radial = np.hypot(dx,dy)
    output = np.full((5,5,features.shape[0]),np.nan)
    for i,(lo,hi) in enumerate(((0,3),(0,7),(3,12),(7,18),(15,28))):
        valid = (radial>=lo)&(radial<hi)&np.isfinite(local[3])
        if valid.sum()<0.75*np.count_nonzero((radial>=lo)&(radial<hi)):
            continue
        sampled = features[:,valid]
        order = np.argsort(local[3,valid],kind="stable")
        for j,part in enumerate(np.array_split(order,5)):
            output[i,j] = sampled[:,part].mean(1)
    # In the full patch the warm-core spectrum and the local cloud background
    # can be matched by B16 even if the two peaks occur at different positions.
    envtable = np.full((2,3,13),np.nan)
    for i,surround in enumerate((inner,outer)):
        envtable[i] = background_curve(surround,np.linspace(190,286,13))
    return {"thermal":output[:,:,:4].ravel(),"spectral":output[:,:,4:7].ravel(),
            "matched":output[:,:,7:].ravel(),"reference":envtable.ravel()}


def sequence(image,mask,centers,times):
    usable = np.flatnonzero(mask.any(axis=(-1,-2)) & np.isfinite(times))
    frames = [signature(image[j],mask[j],centers[j]) for j in usable]
    result = {}
    for k,v in frames[0].items():
        result[k] = np.full((6,len(v)),np.nan,dtype=np.float32)
        result[k][usable] = np.stack([f[k] for f in frames])
    return result
