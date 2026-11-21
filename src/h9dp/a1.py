"""Outer cloud-field fingerprints for pressure transfer between image sequences."""

import numpy as np
import torch


def field(image, mask, xy):
    # Divide a 480-pixel window into 20x20 averages, with no edge wrap.
    dy = np.arange(-240, 240)
    x,y = np.rint(np.asarray(xy)-0.5).astype(int)
    yy,xx = (y+dy)[:,None],x+dy
    yn,xn = yy.clip(0,mask.shape[0]-1),xx.clip(0,mask.shape[1]-1)
    ok = mask[yn,xn] & (yy==yn) & (xx==xn)
    coverage = ok.reshape(20,24,20,24).sum((1,3))
    values = image[:,yn,xn].reshape(4,20,24,20,24)
    bt = (values * ok.reshape(20,24,20,24)).sum((2,4)) / coverage.clip(1) * 0.01 + 150
    return np.where(coverage >= 0.75*24**2,bt,np.nan).astype(np.float32)


def sequence(image,mask,centers,times):
    maps = np.full((6,4,20,20),np.nan,dtype=np.float32)
    for i in np.flatnonzero(mask.any(axis=(-1,-2)) & np.isfinite(times)):
        maps[i] = field(image[i],mask[i],centers[i])
    return {"map":maps}


def translated_codes(maps):
    """Retain azimuthal departures and cross-band environmental texture."""
    x = np.stack((maps[:,:,0],maps[:,:,2]-maps[:,:,0],maps[:,:,3]),axis=2)
    y,d = np.mgrid[-9.5:10,-9.5:10]
    bins = np.floor(np.hypot(y,d)/2)
    anomaly = np.zeros_like(x)
    for i in np.unique(bins):
        ring = bins==i
        values = x[...,ring]
        keep = np.isfinite(values)
        avg = np.where(keep,values,0).sum(-1) / keep.sum(-1).clip(1)
        anomaly[...,ring] = values-avg[...,None]
    outputs = []
    for dy,dx in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):
        crop = anomaly[...,1+dy:19+dy,1+dx:19+dx]
        keep = np.isfinite(crop)
        norm = np.sqrt(np.where(keep,crop**2,0).sum((-1,-2),keepdims=True)).clip(1e-5)
        code = np.where(keep,crop/norm,0)
        outputs.append(code.reshape(-1,3*18*18)/np.sqrt(np.float32(3)))
    return np.stack(outputs)


def sequence_matches(maps, times):
    """Return symmetric similarity summaries and an antisymmetric relative lag estimate."""
    code = torch.as_tensor(translated_codes(maps),device="cuda")
    n = len(maps)
    c = code[0]@code[0].T
    for shifted in code[1:]:
        gram = code[0]@shifted.T
        c = torch.maximum(c,torch.maximum(gram,gram.T))
    c = c.reshape(n,6,n,6).permute(0,2,1,3).contiguous()
    valid = torch.as_tensor(np.isfinite(maps).any((2,3,4)),device=c.device)
    usable = valid[:,None,:,None] & valid[None,:,None,:]
    c = torch.where(usable,c,-2.)
    t = torch.as_tensor(times,device=c.device,dtype=c.dtype)
    dt = t[:,None,:,None]-t[None,:,None,:]
    # Best historical/recent scan alignment, with temporal persistence in nearby matches.
    latest = torch.where(valid,t,-torch.inf).max(1).values
    recent = ((t-latest[:,None])/100).exp().nan_to_num() * valid
    recency = recent[:,None,:,None] * recent[None,:,None,:]
    high = c.amax((2,3))
    weights = ((c-high[:,:,None,None])/0.055).exp() * usable
    lag = (weights*dt.nan_to_num()).sum((2,3))/weights.sum((2,3)).clip(1e-6)
    avg = ((c*weights).sum((2,3)) / weights.sum((2,3)).clip(1e-6))
    now = (c*recency).sum((2,3))/recency.sum((2,3)).clip(1e-6)
    return torch.stack((high, avg, now, lag/180)).double()


def analog_basis(matches,train,held):
    """Map each record to cloud-pattern affinities over training-only support records."""
    order = np.r_[train,held]
    high,avg,now,lag = matches[:,order][:,:,train]
    strength = 0.58*high + 0.27*avg + 0.15*now
    kernels = []
    for scale in (0.045,0.12,0.24):
        phi = ((strength-1)/scale).exp()
        # Euclidean dot products of same support features guarantee PSD even for max-shift matching.
        norm = torch.linalg.vector_norm(phi,dim=1).clamp_min(0.12)
        phi = phi/norm[:,None]
        kernels.extend((phi@phi.T, (phi*lag)@(phi*lag).T))
    return torch.stack(kernels)
