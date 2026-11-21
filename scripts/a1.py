"""Cross-fitted image-analog regression with a radiometric trend and localized residuals."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from h9dp.a1 import analog_basis, sequence_matches
from h9dp.r1 import record_folds, scores
from scripts.v1 import bank


def system(mass, noise, coupling, morph, analog):
    trend = torch.einsum("g,gij->ij",mass[:len(morph)].exp(),morph)
    local = torch.einsum("g,gij->ij",mass[len(morph):].exp(),analog)
    scale = trend.diag().clamp_min(1e-8).rsqrt()
    return trend + local*(1+coupling.exp()*trend*scale[:,None]*scale[None,:]) + 6


def objective(mass,noise,coupling,morph,analog,y,q,weight,prior):
    base = system(mass,noise,coupling,morph,analog)
    chol,_ = torch.linalg.cholesky_ex(base + torch.diag(noise.exp()[q]+0.001))
    inv = torch.cholesky_inverse(chol)
    dual = inv@y
    err = dual/inv.diag()
    evidence = (y@dual + 2*chol.diag().log().sum())/len(y)
    return (weight*err.square()).mean() + 0.12*evidence + 0.0025*(mass-prior).square().mean()


def fit(morph,analog,names,target,quality):
    n = len(target)
    q = torch.as_tensor([{"Good":0,"Fair":1,"Poor":2}[k] for k in quality], device=morph.device)
    y = morph.new_tensor((target-target.mean())/target.std())
    weights = morph.new_tensor([1.,0.32,0.18])[q]
    prior = morph.new_tensor([np.log(0.08 if "recent" in k else 0.03) for k in names] + [-1.8,-3]*3)
    mass = torch.nn.Parameter(prior.clone())
    noise = torch.nn.Parameter(morph.new_tensor([-2.9,-1.9,-1.4]))
    coupling = torch.nn.Parameter(morph.new_tensor(-1.0))
    fn = torch.compile(objective,fullgraph=True,dynamic=False)
    opt = torch.optim.Adam([mass,noise,coupling],lr=0.04)
    train_morph,train_analog = morph[:,:n,:n].contiguous(),analog[:,:n,:n].contiguous()
    timings = []
    for _ in range(140):
        torch.cuda.synchronize()
        start = time.monotonic()
        opt.zero_grad(set_to_none=True)
        loss = fn(mass,noise,coupling,train_morph,train_analog,y,q,weights/weights.mean(),prior)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        timings.append(time.monotonic()-start)
    with torch.no_grad():
        base = system(mass,noise,coupling,morph,analog)
        chol = torch.linalg.cholesky(base[:n,:n]+torch.diag(noise.exp()[q]+0.001))
        dual = torch.cholesky_solve(y[:,None],chol)
        pred = (base[n:,:n]@dual).flatten().cpu().numpy()*target.std()+target.mean()
    state = {k:v.detach().cpu() for k,v in {"mass":mass,"noise":noise,"coupling":coupling,"dual":dual}.items()}
    return pred,state,[timings[0],float(np.median(timings[1:]))]


def evaluate(args):
    raw = (args.source/"all_meta.json").read_bytes()
    rows = json.loads(raw)["records"]
    ids = np.array([r["record"] for r in rows])
    digest = hashlib.sha256(raw).hexdigest()
    times = np.array([r["times"] for r in rows])
    groups = {}
    for file,keys in ((args.polar,("eye","wall","relief","motion","waves","sector")),(args.field,("map",))):
        with np.load(file) as f:
            if not np.array_equal(ids,f["record"]) or not np.array_equal(times,f["time"],equal_nan=True) or str(f["source_sha256"])!=digest:
                raise ValueError("Mismatch between features and source")
            groups.update({k:f[k] for k in keys})
    args.output.mkdir(exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",str(args.output.parent/"inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR",str(args.output.parent/"triton"))
    torch._dynamo.config.suppress_errors = False
    torch.manual_seed(1)
    matches = sequence_matches(groups.pop("map"),times)
    y = np.array([r["dP"] for r in rows])
    q = np.array([r["quality"] for r in rows])
    fold = record_folds(len(ids),5,1)
    pred = np.full(len(ids),np.nan)
    trace = []
    with (args.output/"predictions.csv").open("x",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed","fold","record","quality","target","prediction"])
        for k in range(5):
            train,held = np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            morph,names = bank(groups,times,train,held)
            analog = analog_basis(matches,train,held)
            pred[held],state,timing = fit(morph,analog,names,y[train],q[train])
            torch.save({**state,"names":names,"train_ids":ids[train].tolist()},args.output/f"fold{k}.pt")
            trace.append(timing)
            writer.writerows((1,k,ids[i],q[i],y[i],pred[i]) for i in held)
            stream.flush()
            print(f"fold {k} {timing}",flush=True)
    result = {"seed":1,"metrics":scores(y,pred,q),"first_and_steady_s":trace,"torch":torch.__version__,
              "source_sha256":digest,"device":torch.cuda.get_device_name()}
    (args.output/"metrics.json").write_text(json.dumps(result,indent=2))
    print(result["metrics"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source","field","polar","output"):
        p.add_argument("--"+key,type=Path,required=True)
    evaluate(p.parse_args())
