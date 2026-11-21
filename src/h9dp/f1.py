"""PSD kernels on time-ordered cloud spectra through learned temporal filters."""

import numpy as np
import torch

from h9dp.r1 import _normalize


def frame_groups(data):
    eye = data["eye"].reshape(-1,6,8,8,5)
    wall = data["wall"].reshape(-1,6,8,10,5)
    relief = data["relief"].reshape(-1,6,8,8,5)
    groups = {
        "eye_wv":eye[:,:,:3], "eye_ir":eye[:,:,3], "eye_diff":eye[:,:,4:],
        "eye_gradient":np.diff(eye,axis=3), "wall":wall, "relief":relief,
        "wave":data["waves"], "motion":data["motion"], "sector":data["sector"],
    }
    return {k:v.reshape(*v.shape[:2],-1) for k,v in groups.items()}


def time_filters(t, valid):
    """Three temporal averages with missing-scan support inferred per descriptor family."""
    if not valid.any(1).all():
        # Some families (e.g. motion) can be missing throughout a record.
        last = np.max(np.where(np.isfinite(t), t, -np.inf), axis=1)
    else:
        last = np.max(np.where(valid, t, -np.inf), axis=1)
    lag = t - last[:,None]
    w = np.stack([np.exp(-0.5*((lag-center)/scale)**2) for center,scale in ((0,8),(-23,30),(-185,105))],axis=1)
    w = np.where(valid[:,None,:],w,0)
    return w / w.sum(2,keepdims=True).clip(1e-10)


def compress(k,weights):
    n = len(weights)
    return torch.einsum("ipf,ifjh,jqh->pqij",weights,k.reshape(n,6,n,6),weights)


def blocks(data,times,train,held):
    order = np.r_[train,held]
    output, names = [], []
    for name, values in frame_groups(data).items():
        xfit = values[train].reshape(-1,values.shape[-1])
        xeval = values[held].reshape(-1,values.shape[-1])
        a,b = _normalize(xfit,xeval)
        if not a.shape[1]:
            continue
        x = torch.as_tensor(np.concatenate((a,b)),dtype=torch.float32,device="cuda")
        w = time_filters(times[order],np.isfinite(values[order]).any(-1))
        weight = torch.as_tensor(w,dtype=x.dtype,device=x.device)
        gram = x @ x.T / x.shape[1]
        d = (gram.diag()[:,None]+gram.diag()[None,:]-2*gram).clamp_min(0)
        r = (3*d+1e-8).sqrt()
        output.extend((compress(gram,weight),compress((1+r)*(-r).exp(),weight)))
        names.extend((name+"_linear",name+"_local"))
        if name.startswith("eye_") and name!="eye_gradient":
            # Channelwise nonlinear response before filtering allows transient hot cores.
            knots = torch.linspace(-3.,3.,7,dtype=x.dtype,device=x.device)
            phi = torch.nn.functional.normalize((-0.5*((x[:,:,None]-knots)/0.9).square()).exp(),dim=2)
            phi = phi.flatten(1)
            output.append(compress(phi@phi.T/x.shape[1],weight))
            names.append(name+"_response")
        if name == "sector":
            output.append(compress((-d/0.18).exp(),weight))
            names.append("sector_tracking")
    return torch.stack(output).double(),names


def gram(blocks,filt,scale,latitude):
    matrices = filt@filt.transpose(1,2)
    return torch.einsum("gpq,gpqij->ij",matrices,blocks) + scale.exp()*latitude + 8


def objective(filt,scale,noise,blocks,latitude,y,q,weight,prior):
    kernel = gram(blocks,filt,scale,latitude)
    chol,_ = torch.linalg.cholesky_ex(kernel+torch.diag(noise.exp()[q]+0.002))
    inv = torch.cholesky_inverse(chol)
    dual = inv@y
    loo = ((dual/inv.diag()).square()*weight).mean()
    evidence = (y@dual+2*chol.diag().log().sum())/len(y)
    return loo+0.11*evidence+0.012*(filt-prior).square().sum((-1,-2)).mean()
