"""Fit a varying-coefficient spectral regression with fold-local wavelength whitening."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from h9dp.pressure_labels import read_catalog, verify_catalog_labels
from h9dp.r1 import record_folds, scores, summarize_time
from h9dp.r1_latitude import latitude_fields, latitude_kernel, native_r3_paths
from scripts.v1 import bank


def spectrum_grams(features, channels, train, held):
    n = len(train)
    x = features[np.r_[train,held]].reshape(len(train)+len(held),-1,channels)
    finite = np.isfinite(x[:n])
    count = finite.sum(0).clip(1)
    center = np.where(finite,x[:n],0).sum(0)/count
    centered = np.where(np.isfinite(x),x-center,0)
    scale = np.sqrt((centered[:n]**2).sum(0)/count).clip(0.15)
    standardized = (centered/scale).clip(-5,5)
    fit = standardized[:n].reshape(-1,channels)
    eig,vec = np.linalg.eigh(fit.T@fit/len(fit))
    # Partial whitening exposes weak differences without inverting nearly collinear channels.
    rotation = (vec*(eig.clip(0.035)**-0.38))@vec.T
    t = torch.as_tensor(standardized @ rotation,device="cuda",dtype=torch.float64).permute(2,0,1)
    return torch.einsum("anp,bmp->abnm",t,t)/t.size(-1)


def bases(groups,lat,times,train,held):
    valid = np.isfinite(groups["eye"]).any(-1)
    morph,names = bank({k:groups[k] for k in ("eye","wall","relief","waves","motion","sector")},times,train,held)
    order = np.r_[train,held]
    thermal = groups["thermal"].reshape(len(times),6,5,5,4)
    eye = summarize_time(thermal[:,:,0,-1],times,valid)["recent"][:,3]
    annulus = summarize_time(thermal[:,:,4,0],times,valid)["recent"][:,3]
    gate = 1/(1+np.exp(-((eye-annulus-18)/7).clip(-40,40)))
    gate = np.where(np.isfinite(gate),gate,0.5)
    gates = np.stack((gate,1-gate),1)[order]
    gates /= np.linalg.norm(gates,axis=-1,keepdims=True)
    gate_gram = torch.as_tensor(gates@gates.T,device="cuda",dtype=morph.dtype)
    latitude = torch.as_tensor(latitude_kernel(lat,train,held,with_field=False),device="cuda",dtype=morph.dtype)
    blocks = []
    for key,channels in (("thermal",4),("spectral",3),("matched",6),("reference",3)):
        frames = groups[key]
        if key == "reference":
            frames = frames.reshape(len(times),6,2,3,13).transpose(0,1,2,4,3).reshape(len(times),6,-1)
        stats = summarize_time(frames,times,valid)
        for statistic in ("recent","latest","trend"):
            blocks.append(spectrum_grams(stats[statistic],channels,train,held))
    return morph,names,blocks,latitude,gate_gram


def covariance(mass, factors, extra, morph, blocks, latitude, gate):
    result = torch.einsum("g,gij->ij",mass.exp(),morph)+extra[0].exp()*latitude + 8
    for i,block in enumerate(blocks):
        factor = factors[i]
        weight = factor@factor.T
        gram = torch.einsum("ab,abij->ij",weight,block)
        result = result + gram * (1+extra[1].exp()*gate)
    return result


def loss(mass,factors,extra,noise,morph,blocks,latitude,gate,y,q,w,prior):
    gram = covariance(mass,factors,extra,morph,blocks,latitude,gate)
    chol,_ = torch.linalg.cholesky_ex(gram+torch.diag(noise.exp()[q]+0.002))
    inv = torch.cholesky_inverse(chol)
    dual = inv@y
    loo = ((dual/inv.diag()).square()*w).mean()
    evidence = (y@dual+2*chol.diag().log().sum())/len(y)
    shrink = sum(f.square().sum() for f in factors)
    return loo + 0.12*evidence + 0.005*(mass-prior).square().mean() + 0.006*shrink


def fit(morph,names,blocks,latitude,gate,target,quality):
    prior = morph.new_tensor([np.log(0.06 if "recent" in name else 0.025) for name in names])
    mass = torch.nn.Parameter(prior.clone())
    factors = torch.nn.ParameterList([torch.nn.Parameter(torch.eye(len(b),device=b.device,dtype=b.dtype)*0.06)
                                     for b in blocks])
    extra = torch.nn.Parameter(morph.new_tensor([-2.3,-1.3]))
    noise = torch.nn.Parameter(morph.new_tensor([-2.8,-1.9,-1.45]))
    y = morph.new_tensor((target-target.mean())/target.std())
    q = torch.as_tensor([{"Good":0,"Fair":1,"Poor":2}[k] for k in quality],device=y.device)
    w = morph.new_tensor([1.,0.42,0.25])[q]
    n = len(y)
    optimizer = torch.optim.Adam([{"params":[mass,noise,extra],"lr":0.04},{"params":factors,"lr":0.007}])
    compiled = torch.compile(loss,fullgraph=True,dynamic=False)
    inputs = (morph[:,:n,:n].contiguous(),[b[:,:,:n,:n].contiguous() for b in blocks],
              latitude[:n,:n].contiguous(),gate[:n,:n].contiguous(),y,q,w/w.mean(),prior)
    steps = []
    for _ in range(125):
        torch.cuda.synchronize()
        start = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        value = compiled(mass,list(factors),extra,noise,*inputs)
        value.backward()
        optimizer.step()
        torch.cuda.synchronize()
        steps.append(time.monotonic()-start)
    with torch.no_grad():
        gram = covariance(mass,factors,extra,morph,blocks,latitude,gate)
        chol = torch.linalg.cholesky(gram[:n,:n]+torch.diag(noise.exp()[q]+0.002))
        dual = torch.cholesky_solve(y[:,None],chol)
        estimate = (gram[n:,:n]@dual).flatten()*target.std()+target.mean()
    state = {k:v.detach().cpu() for k,v in {"mass":mass,"extra":extra,"noise":noise,"dual":dual}.items()}
    state["factors"] = [f.detach().cpu() for f in factors]
    return estimate.cpu().numpy(),state,[steps[0],float(np.median(steps[1:]))]


def evaluate(args):
    rows_raw = (args.source/"all_meta.json").read_bytes()
    rows = json.loads(rows_raw)["records"]
    catalog, catalog_digest = read_catalog()
    verify_catalog_labels(rows, catalog)
    ids = np.array([r["record"] for r in rows])
    times = np.array([r["times"] for r in rows])
    digest = hashlib.sha256(rows_raw).hexdigest()
    groups = {}
    for file,keys in ((args.polar,("eye","wall","relief","waves","motion","sector")),
                      (args.column,("thermal","spectral","matched","reference"))):
        with np.load(file) as f:
            if not np.array_equal(ids,f["record"]) or not np.array_equal(times,f["time"],equal_nan=True) or str(f["source_sha256"])!=digest:
                raise ValueError("Source differs from feature cache")
            groups.update({k:f[k] for k in keys})
    paths = native_r3_paths(json.loads(args.index.read_bytes()),ids,times)
    latitude = latitude_fields(paths,np.isfinite(groups["eye"]).any(-1))
    args.output.mkdir(exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",str(args.output.parent/"inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR",str(args.output.parent/"triton"))
    torch.manual_seed(1)
    torch._dynamo.config.suppress_errors = False
    y = np.array([r["dP"] for r in rows])
    quality = np.array([r["quality"] for r in rows])
    fold = record_folds(len(ids),5,1)
    pred = np.full(len(ids),np.nan)
    trace = []
    with (args.output/"predictions.csv").open("x",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed","fold","record","quality","target","prediction"])
        for k in range(5):
            train,held = np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            inputs = bases(groups,latitude,times,train,held)
            pred[held],state,timing = fit(*inputs,y[train],quality[train])
            torch.save({**state,"train_ids":ids[train].tolist(),"names":inputs[1]},args.output/f"fold{k}.pt")
            trace.append(timing)
            writer.writerows((1,k,ids[i],quality[i],y[i],pred[i]) for i in held)
            stream.flush()
            print(f"fold {k} {timing}",flush=True)
    result = {"seed":1,"catalog_sha256":catalog_digest,"metrics":scores(y,pred,quality),"first_and_steady_s":trace,"torch":torch.__version__,
              "native_latitudes":int(np.isfinite(latitude[:,1,1]).sum()),"source_sha256":digest,"device":torch.cuda.get_device_name()}
    (args.output/"metrics.json").write_text(json.dumps(result,indent=2))
    print(result["metrics"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source","column","polar","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--index",type=Path,default=Path("runs/h9dp/native500_v2/index.json"))
    evaluate(p.parse_args())
