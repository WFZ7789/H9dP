"""Multi-fidelity regression with additional single-scan historical training records."""

import argparse
import csv
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from h9dp.r1 import _normalize, record_folds, scores, summarize_time
from h9dp.v1 import sequence
from scripts.audit_motis_inputs import record_key, unique_lookup
from scripts.v1 import bank

KEYS = ("eye","wall","relief","motion","waves","sector")
STATIC = ("eye","wall","relief","waves","sector")


def auxiliary(input_dir):
    meta = json.loads((input_dir/"index.json").read_bytes())
    rows = meta["records"]
    fields = {k:[] for k in STATIC}
    for entry in rows:
        with np.load(input_dir/f"{entry['index']:04d}.npz") as f:
            if json.loads(str(f["manifest"])) != entry:
                raise ValueError("Extra observation manifest differs")
            image = np.broadcast_to(f["image"],(6,4,500,500))
            mask = np.zeros((6,500,500),bool)
            mask[-1] = f["mask"]
            groups = sequence(image,mask,np.tile(entry["center"],(6,1)),np.arange(-5,1))
            for k in STATIC:
                fields[k].append(groups[k][-1])
    return {k:np.stack(v) for k,v in fields.items()},rows


def snapshot_bank(main,extra,train,held):
    # Physical samples are standardized once; matched old and current labels share features.
    bs,names = [],[]
    for key,values in main.items():
        fit_rows = np.concatenate((values[train],extra[key]))
        a,b = _normalize(fit_rows,values[held])
        x = torch.as_tensor(np.concatenate((a,b)),device="cuda",dtype=torch.float64)
        dot = x@x.T/x.shape[1]
        d = (dot.diag()[:,None]+dot.diag()[None,:]-2*dot).clamp_min(0)
        r = (3*d+1e-10).sqrt()
        bs.extend((dot, (1+r)*(-r).exp()))
        names.extend((key+"_linear",key+"_local"))
        if key in ("eye","relief","wall"):
            phi = torch.nn.functional.normalize((-0.5*(x[:,:,None]-x.new_tensor([-2.5,-1.25,0,1.25,2.5])).square()).exp(),dim=2)
            phi=phi.flatten(1)
            bs.append(phi@phi.T/x.shape[1])
            names.append(key+"_univariate")
    return torch.stack(bs),names


def joint_covariance(shared_mass,seq_mass,rho,noise,correlation,shared,series,old_positions,pairing,qc,qo):
    nc = len(qc)
    shared_k = torch.einsum("g,gij->ij",shared_mass.exp(),shared) + 4
    current = torch.einsum("g,gij->ij",seq_mass.exp(),series) + 4 + rho.square()*shared_k[:nc,:nc]
    vcur,vold = noise.exp()[0,qc]+0.002,noise.exp()[1,qo]+0.002
    old = shared_k[old_positions[:,None],old_positions] + torch.diag(vold)
    cross = rho*shared_k[:nc,old_positions] + pairing*correlation.tanh()*torch.sqrt(vcur[:,None]*vold[None,:])
    return torch.cat((torch.cat((current+torch.diag(vcur),cross),1),torch.cat((cross.T,old),1)),0)


def deletion_errors(diagonal,dual,cross,paired):
    """Exact residual after jointly deleting both pressure labels of a matched observation."""
    nc = len(paired)
    other = nc+paired.clamp_min(0)
    shared = torch.where(paired>=0,cross,0.)
    return (dual[:nc]-shared*dual[other]/diagonal[other])/(diagonal[:nc]-shared.square()/diagonal[other])


def objective(shared_mass,seq_mass,rho,noise,correlation,shared,series,old_positions,pairing,qc,qo,y,paired,w,prior,seq_prior):
    cov = joint_covariance(shared_mass,seq_mass,rho,noise,correlation,shared,series,old_positions,pairing,qc,qo)
    chol,_ = torch.linalg.cholesky_ex(cov)
    inv = torch.cholesky_inverse(chol)
    dual = inv@y
    nc = len(qc)
    cross = inv[torch.arange(nc,device=y.device),nc+paired.clamp_min(0)]
    errors = deletion_errors(inv.diag(),dual,cross,paired)
    evidence = (y@dual+2*chol.diag().log().sum())/len(y)
    return (errors.square()*w).mean()+0.18*evidence+0.003*((shared_mass-prior).square().mean()
                       +(seq_mass-seq_prior).square().mean())+0.008*(rho-0.85).square()


def fit(shared,series,names,target,old_target,qc,qo,old_ids,ntraining):
    current_n = len(target)
    y = np.concatenate(((target-target.mean())/target.std(),(old_target-old_target.mean())/old_target.std()))
    y = shared.new_tensor(y)
    tags = {"Good":0,"Fair":1,"Poor":2}
    qc,qo = [torch.as_tensor([tags[t] for t in q],device=y.device) for q in (qc,qo)]
    positions = torch.as_tensor(old_ids,device=y.device)
    pairing = (torch.arange(current_n,device=y.device)[:,None]==positions).to(y.dtype)
    paired = torch.where(pairing.any(1),pairing.argmax(1),-1)
    prior = shared.new_full((len(shared),),-2.35)
    seq_prior = shared.new_tensor([np.log(0.07 if "recent" in k else 0.025) for k in names])
    shared_mass,seq_mass = torch.nn.Parameter(prior.clone()),torch.nn.Parameter(seq_prior.clone())
    rho = torch.nn.Parameter(y.new_tensor(0.85))
    noise = torch.nn.Parameter(y.new_tensor([[-2.7,-1.9,-1.4],[-2.6,-1.9,-1.45]]))
    correlation = torch.nn.Parameter(y.new_tensor(0.45))
    params = [shared_mass,seq_mass,rho,noise,correlation]
    fn = torch.compile(objective,fullgraph=True,dynamic=False)
    opt = torch.optim.Adam(params,lr=0.035)
    weights = y.new_tensor([1,0.4,0.22])[qc]
    train_args = (shared[:,:ntraining,:ntraining].contiguous(),series[:,:current_n,:current_n].contiguous(),
                  positions,pairing,qc,qo)
    steps = []
    for _ in range(115):
        torch.cuda.synchronize()
        start = time.monotonic()
        opt.zero_grad(set_to_none=True)
        loss = fn(*params,*train_args,y,paired,weights/weights.mean(),prior,seq_prior)
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        steps.append(time.monotonic()-start)
    with torch.no_grad():
        cov = joint_covariance(*params,*train_args)
        dual = torch.cholesky_solve(y[:,None],torch.linalg.cholesky(cov))
        k = torch.einsum("g,gij->ij",shared_mass.exp(),shared[:,ntraining:,:ntraining])+4
        seq = torch.einsum("g,gij->ij",seq_mass.exp(),series[:,current_n:,:current_n])+4
        test_cross = torch.cat((rho.square()*k[:,:current_n]+seq,rho*k[:,positions]),1)
        estimate = (test_cross@dual).flatten()*target.std()+target.mean()
    state = {k:v.detach().cpu() for k,v in zip(("shared","sequence","rho","noise","correlation","dual"),[*params,dual],strict=True)}
    return estimate.cpu().numpy(),state,[steps[0],float(np.median(steps[1:]))]


def evaluate(args):
    raw = (args.source/"all_meta.json").read_bytes()
    rows = json.loads(raw)["records"]
    ids = np.array([r["record"] for r in rows])
    times = np.array([r["times"] for r in rows])
    digest = hashlib.sha256(raw).hexdigest()
    with np.load(args.polar) as f:
        if str(f["source_sha256"])!=digest or not np.array_equal(ids,f["record"]) or not np.array_equal(times,f["time"],equal_nan=True):
            raise ValueError("Polar feature source differs")
        groups = {k:f[k] for k in KEYS}
    archive = json.loads((args.extra/"index.json").read_bytes())
    if archive["source_sha256"] != hashlib.sha256(args.historical.read_bytes()).hexdigest():
        raise ValueError("Historical label version differs from acquisition")
    with args.historical.open(encoding="utf-8-sig") as stream:
        original = list(csv.DictReader(stream))
    lookup = unique_lookup(original)
    catalog = json.loads(args.catalog.read_bytes())["records"]
    matched = np.array([lookup.get(record_key(catalog[i]),-1) for i in ids])
    extra,more = auxiliary(args.extra)
    if set(matched)&{r["index"] for r in more}:
        raise ValueError("An external snapshot duplicates a current observation")
    for entry in more:
        if float(original[entry["index"]]["dP MOTIS"])!=entry["target"]:
            raise ValueError("Incorrect external snapshot target")
    valid = np.isfinite(groups["eye"]).any(-1)
    static = {k:summarize_time(groups[k],times,valid)["latest"] for k in STATIC}
    args.output.mkdir(exist_ok=False)
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR",str(args.output.parent/"inductor"))
    os.environ.setdefault("TRITON_CACHE_DIR",str(args.output.parent/"triton"))
    torch.manual_seed(1)
    torch._dynamo.config.suppress_errors = False
    folds = record_folds(len(ids),5,1)
    y = np.array([r["dP"] for r in rows])
    quality = np.array([r["quality"] for r in rows])
    pred = np.full(len(y),np.nan)
    trace = []
    with (args.output/"predictions.csv").open("x",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["seed","fold","record","quality","target","prediction"])
        for k in range(5):
            train,held = np.flatnonzero(folds!=k),np.flatnonzero(folds==k)
            local = np.flatnonzero(matched[train]>=0)
            train_old = [original[i] for i in matched[train[local]]]
            oy = np.array([float(r["dP MOTIS"]) for r in train_old]+[r["target"] for r in more])
            oq = [r["Eye clarity"] for r in train_old]+[r["quality"] for r in more]
            ntraining = len(train)+len(more)
            old_pos = np.r_[local,np.arange(len(train),ntraining)]
            series,names = bank(groups,times,train,held)
            shared,shared_names = snapshot_bank(static,extra,train,held)
            print(f"fold {k} current={len(train)} old={len(oy)} extra={len(more)}",flush=True)
            pred[held],state,timing = fit(shared,series,names,y[train],oy,quality[train],oq,old_pos,ntraining)
            trace.append(timing)
            torch.save({**state,"train_ids":ids[train].tolist(),"matched_old_ids":matched[train[local]].tolist(),
                        "extra_old_ids":[r["index"] for r in more],"names":names,"shared_names":shared_names},args.output/f"fold{k}.pt")
            writer.writerows((1,k,ids[i],quality[i],y[i],pred[i]) for i in held)
            stream.flush()
    result = {"seed":1,"metrics":scores(y,pred,quality),"first_and_steady_s":trace,"source_sha256":digest,
              "extra_scans":len(more),"paired_old":int((matched>=0).sum()),"device":torch.cuda.get_device_name(),"torch":torch.__version__}
    (args.output/"metrics.json").write_text(json.dumps(result,indent=2))
    print(result,flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for key in ("source","polar","extra","historical","output"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--catalog",type=Path,default=Path("MOTIS_cyclone_records.json"))
    evaluate(p.parse_args())
