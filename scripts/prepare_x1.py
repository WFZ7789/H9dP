"""Fetch extra single-scan Himawari training images for historical MOTIS observations."""

import argparse
import bz2
import csv
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen
from xml.etree import ElementTree

import numpy as np

from h9dp.hsd import _parse_header, brightness_temperature, crop_window, earth_mask, lonlat_to_pixel, read_header

BANDS = (8,9,10,16)


def candidates(source):
    with source.open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    result = []
    for index,r in enumerate(rows):
        target = datetime.strptime(r["Year"]+r["Date"].zfill(4)+r["Time"].zfill(4),"%Y%m%d%H%M")
        lat,lon = float(r["Lat"]),float(r["Lon"])
        # Acquire 2015 western Pacific plus southern-hemisphere observations in the AHI disk.
        if target < datetime(2015,7,7) or abs((lon-140.7+180)%360-180)>60:
            continue
        if not ((target.year==2015 and r["Region"]=="WP") or (lat<0 and r["Region"] in ("AU","SP","SI"))):
            continue
        result.append({"index":index,"name":r["Name"],"time":target.isoformat(),"quality":r["Eye clarity"],
                       "target":float(r["dP MOTIS"]),"latitude":lat,"longitude":lon,"region":r["Region"]})
    return result


def fetch(url):
    for i in range(3):
        try:
            with urlopen(url,timeout=50) as stream:
                return stream.read()
        except HTTPError as exc:
            if exc.code in (403,404) or i==2:
                raise
        except OSError:
            if i==2:
                raise
        time.sleep(i+1)


def listing(base,prefix):
    xml = ElementTree.fromstring(fetch(base+"/?"+urlencode({"list-type":2,"prefix":prefix})))
    return [k.text for k in xml.findall(".//{*}Key")]


def extract(row,root,ref):
    out = root/f"{row['index']:04d}.npz"
    if out.exists():
        with np.load(out) as data:
            return json.loads(str(data["manifest"]))
    t = datetime.fromisoformat(row["time"])
    nominal = t.replace(minute=(t.minute//10)*10)
    satellite = 8 if nominal < datetime(2022,12,13,5) else 9
    base = f"https://noaa-himawari{satellite}.s3.amazonaws.com"
    prefix = f"AHI-L1b-FLDK/{nominal:%Y/%m/%d/%H%M}/"
    keys = listing(base,prefix)
    x0,y0 = crop_window(ref,row["latitude"],row["longitude"],500)
    cx,cy = lonlat_to_pixel(ref,row["latitude"],row["longitude"])
    on_earth = earth_mask(ref,(x0,y0),500)
    image = np.zeros((4,500,500),np.int16)
    valid = np.zeros_like(image,dtype=bool)
    used,volume = [],0
    for b,band in enumerate(BANDS):
        segments = []
        pattern = re.compile(rf"_B{band:02d}_FLDK_R20_S(\d{{2}})(\d{{2}})\.DAT\.bz2$")
        for key in keys:
            match = pattern.search(key)
            if match:
                seg,total = map(int,match.groups())
                segment_rows = ref.height//total
                if (seg-1)*segment_rows<y0+500 and seg*segment_rows>y0:
                    segments.append(key)
        if not segments:
            raise ValueError(f"No B{band:02d} segment at {prefix}")
        filled = np.zeros(500,dtype=bool)
        for key in segments:
            packed = fetch(base+"/"+key)
            volume += len(packed)
            raw = bz2.decompress(packed)
            h = _parse_header(raw)
            if (h.width,h.band,h.projection,h.compression)!=(ref.width,band,ref.projection,0):
                raise ValueError(f"Unexpected calibration grid in {key}")
            ybase = h.segment_first_line-1
            lo,hi = max(y0,ybase),min(y0+500,ybase+h.height)
            counts = np.frombuffer(raw,dtype=h.endian+"u2",offset=h.header_bytes).reshape(h.height,h.width)
            section = slice(lo-y0,hi-y0)
            bt,ok = brightness_temperature(counts[lo-ybase:hi-ybase,x0:x0+500],h,on_earth[section])
            image[b,section] = np.rint((bt-150)*100).clip(-32768,32767).astype(np.int16)
            valid[b,section] = ok
            filled[section] = True
            used.append(key)
        if not filled.all():
            raise ValueError(f"Incomplete B{band} window")
    mask = valid.all(0)
    if mask.sum()<500**2*0.5:
        raise ValueError("Insufficient jointly calibrated pixels")
    record = {**row,"nominal":nominal.isoformat(),"satellite":satellite,"bucket":base,
              "keys":used,"bytes":volume,"window":[x0,y0],"center":[cx-x0,cy-y0],"valid_pixels":int(mask.sum())}
    temp = out.with_suffix(".partial")
    with temp.open("wb") as stream:
        np.savez_compressed(stream,image=image,mask=mask,manifest=json.dumps(record))
    temp.replace(out)
    return record


def prepare(source,output,workers):
    proxy_names = [k for k in os.environ if "proxy" in k.lower()]
    for k in proxy_names:
        del os.environ[k]
    output.mkdir(exist_ok=True,parents=True)
    reference = read_header(next(Path("HIMAWARI_data/FLDK").rglob("*B08*DAT*")))
    ref = replace(reference,height=5500,segment_first_line=1)
    rows = candidates(source)
    started = time.monotonic()
    result,errors = [],[]
    print(f"Candidate records {len(rows)}; proxy variables unset for every HTTP request",flush=True)
    with ThreadPoolExecutor(workers) as pool:
        handles = {pool.submit(extract,r,output,ref):r for r in rows}
        for i,f in enumerate(as_completed(handles),1):
            try:
                result.append(f.result())
            except (OSError,ValueError,EOFError) as exc:
                row = handles[f]
                errors.append({"index":row["index"],"error":str(exc)})
                print(f"missing row={row['index']} {exc}",flush=True)
            if i%20==0:
                print(f"{i}/{len(rows)}; saved {len(result)}; GB={sum(r['bytes'] for r in result)/1e9:.2f}",flush=True)
    report = {"source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "proxy_policy":"all variables containing PROXY case-insensitively unset before requests",
              "candidates":len(rows),"seconds":time.monotonic()-started,"records":sorted(result,key=lambda r:r["index"]),
              "errors":errors}
    with (output/"index.tmp").open("w") as stream:
        json.dump(report,stream,indent=2)
    (output/"index.tmp").replace(output/"index.json")
    print(f"saved {len(result)}/{len(rows)}, {len(errors)} missing",flush=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--workers",type=int,default=16)
    a=p.parse_args()
    prepare(a.source,a.output,a.workers)
