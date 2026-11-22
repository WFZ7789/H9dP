"""Estimate the temporal response of frame-level spectral features to MOTIS pressure."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from h9dp.f1 import blocks, gram, objective
from h9dp.pressure_labels import read_catalog, verify_catalog_labels
from h9dp.r1 import record_folds, scores
from h9dp.r1_latitude import latitude_fields, latitude_kernel, native_r3_paths


def fit(features,latitude,target,quality):
    prior = features.new_tensor([[0.13,0.04,0.],[0.19,-0.04,0.02],[0.045,0.,-0.02]]).repeat(len(features),1,1)
    filt = torch.nn.Parameter(prior.clone())
    scale = torch.nn.Parameter(features.new_tensor(-2.3))
    noise = torch.nn.Parameter(features.new_tensor([-2.9,-1.9,-1.4]))
    y = features.new_tensor((target-target.mean())/target.std())
    q = torch.as_tensor([{"Good":0,"Fair":1,"Poor":2}[v] for v in quality],device=features.device)
    w = features.new_tensor([1.,0.4,0.22])[q]
    n = len(y)
    fn = torch.compile(objective,fullgraph=True,dynamic=False)
    opt = torch.optim.Adam([{"params":[filt],"lr":0.012},{"params":[scale,noise],"lr":0.04}])
    args = (features[:,:,:,:n,:n].contiguous(),latitude[:n,:n].contiguous(),y,q,w/w.mean(),prior)
    times = []
    for _ in range(130):
        torch.cuda.synchronize()
        start = time.monotonic()
        opt.zero_grad(set_to_none=True)
        loss = fn(filt,scale,noise,*args)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        times.append(time.monotonic()-start)
    with torch.no_grad():
        kernel = gram(features,filt,scale,latitude)
        chol = torch.linalg.cholesky(kernel[:n,:n]+torch.diag(noise.exp()[q]+0.002))
        dual = torch.cholesky_solve(y[:,None],chol)
        estimate = (kernel[n:,:n]@dual).flatten()*target.std()+target.mean()
    state = {k:v.detach().cpu() for k,v in {"filter":filt,"scale":scale,"noise":noise,"dual":dual}.items()}
    return estimate.cpu().numpy(),state,[times[0],float(np.median(times[1:]))]


def evaluate(args):
    raw = (args.source/"all_meta.json").read_bytes()
    rows = json.loads(raw)["records"]
    catalog, catalog_digest = read_catalog()
    verify_catalog_labels(rows, catalog)
    ids = np.array([r["record"] for r in rows])
    digest = hashlib.sha256(raw).hexdigest()
    with np.load(args.polar) as f:
        if str(f["source_sha256"])!=digest or not np.array_equal(ids,f["record"]):
            raise ValueError("Feature source mismatch")
        times = f["time"]
        data = {k:f[k] for k in ("eye","wall","relief","waves","motion","sector")}
    paths = native_r3_paths(json.loads(args.index.read_bytes()),ids,times)
    fields = latitude_fields(paths,np.isfinite(data["eye"]).any(-1))
    args.output.mkdir(exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",str(args.output.parent/"inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR",str(args.output.parent/"triton"))
    torch.manual_seed(1)
    torch._dynamo.config.suppress_errors = False
    fold = record_folds(len(ids),5,1)
    y = np.array([r["dP"] for r in rows])
    q = np.array([r["quality"] for r in rows])
    pred = np.full(len(ids),np.nan)
    trace = []
    with (args.output/"predictions.csv").open("x",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed","fold","record","quality","target","prediction"])
        for k in range(5):
            train,held = np.flatnonzero(fold!=k),np.flatnonzero(fold==k)
            matrices,names = blocks(data,times,train,held)
            lat = torch.as_tensor(latitude_kernel(fields,train,held,with_field=False),device="cuda")
            pred[held],state,timing = fit(matrices,lat,y[train],q[train])
            trace.append(timing)
            torch.save({**state,"names":names,"train_ids":ids[train].tolist()},args.output/f"fold{k}.pt")
            writer.writerows((1,k,ids[i],q[i],y[i],pred[i]) for i in held)
            stream.flush()
            print(f"fold {k} {timing}",flush=True)
    result = {"seed":1,"catalog_sha256":catalog_digest,"metrics":scores(y,pred,q),"first_and_steady_s":trace,"source_sha256":digest,
              "device":torch.cuda.get_device_name(),"torch":torch.__version__}
    (args.output/"metrics.json").write_text(json.dumps(result,indent=2))
    print(result["metrics"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source","polar","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--index",type=Path,default=Path("runs/h9dp/native500_v2/index.json"))
    evaluate(p.parse_args())
