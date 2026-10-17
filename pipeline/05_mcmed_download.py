"""Downloads the selected MC-MED Pleth records from PhysioNet (credentialed access required).

Without --inspect or --download only the plan is printed. The password is prompted and not stored.
Usage: python 05_mcmed_download.py [--csns FILE] --download
"""
import argparse
import getpass
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import sys
import time
from urllib.parse import unquote, urljoin, urlsplit
from urllib3.exceptions import HTTPError as TransportError
import pandas as pd
import requests

ROOT = Path(os.environ.get("PPG_ROOT", "."))
BASE = 'https://physionet.org/files/mc-med/1.0.1/data/waveforms/'

class DownloadError(RuntimeError):
    pass

class AccessError(DownloadError):
    pass

class BudgetError(DownloadError):
    pass

class RetryableError(DownloadError):
    pass

def retry(operation, attempts=3):
    for attempt in range(attempts):
        try:
            return operation()
        except (requests.RequestException, TransportError, RetryableError):
            if attempt+1 == attempts:
                raise
            print(f' Network retry {attempt+1}/{attempts-1}', flush=True)
            time.sleep(min(2**attempt, 8))

class Budget:
    def __init__(self, gib):
        self.limit, self.used = int(gib*1024**3), 0

class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'a':
            self.hrefs.extend(v for k,v in attrs if k.lower()=='href' and v)

def file_links(html, directory):
    parser = Links()
    parser.feed(html)
    base = urlsplit(directory)
    found = {}
    for href in parser.hrefs:
        p = urlsplit(urljoin(directory, href))
        raw_path = p.path.replace('/content/mc-med/', '/files/mc-med/', 1)
        if (p.scheme,p.netloc)!=(base.scheme,base.netloc) or not raw_path.startswith(base.path):
            continue
        name = unquote(raw_path[len(base.path):])
        if not re.fullmatch(r'[A-Za-z0-9_-]+\.(?:hea|dat)', name):
            continue
        found[name] = directory+name
    return sorted(found.items())

def chunks(response, budget, cap=None):
    read = 0
    length = response.headers.get('Content-Length')
    if length and (int(length)>budget.limit-budget.used or (cap and int(length)>cap)):
        raise BudgetError('Response exceeds byte budget or directory size limit.')
    while True:
        remaining = budget.limit-budget.used
        if cap is not None:
            remaining = min(remaining,cap-read)
        if remaining <= 0:
            if length and read == int(length):
                return
            raise BudgetError('Byte budget or directory size limit reached.')
        chunk = response.raw.read(min(1024**2,remaining))
        if not chunk:
            if length and read != int(length):
                raise RetryableError('Incomplete HTTP response.')
            return
        budget.used += len(chunk)
        read += len(chunk)
        yield chunk

def check_response(response, file=False):
    if response.status_code in (401,403):
        raise AccessError(f'HTTP {response.status_code}: check PhysioNet credentials and MC-MED access/DUA.')
    if response.status_code in (408,429,500,502,503,504):
        raise RetryableError(f'HTTP {response.status_code}')
    if response.status_code != 200:
        raise DownloadError(f'HTTP {response.status_code}; check account access, DUA, and directory URL.')
    if response.headers.get('Content-Encoding','identity') != 'identity':
        raise DownloadError('Unexpected Content-Encoding.')
    if file and 'text/html' in response.headers.get('Content-Type','').lower():
        raise AccessError('HTML returned instead of waveform file, possibly a login page.')

def list_visit(session, directory, budget):
    with session.get(directory,stream=True,allow_redirects=False,timeout=(15,60)) as r:
        fallback = r.status_code in (301,302,303,307,308,404)
        if not fallback:
            check_response(r)
            html=b''.join(chunks(r,budget,cap=16*1024**2)).decode('utf-8',errors='replace')
    if fallback:
        page=directory.replace('/files/mc-med/','/content/mc-med/',1)
        with session.get(page,stream=True,allow_redirects=False,timeout=(15,60)) as r:
            check_response(r)
            html=b''.join(chunks(r,budget,cap=16*1024**2)).decode('utf-8',errors='replace')
    files=file_links(html,directory)
    if not files:
        raise AccessError('No .hea/.dat links found; check directory layout and access before continuing.')
    names={name for name,_ in files}
    for name in names:
        other=name.rsplit('.',1)[0]+('.dat' if name.endswith('.hea') else '.hea')
        if other not in names:
            raise DownloadError('Missing .hea/.dat partner; inspect the visit directory.')
    def order(item):
        name=item[0]
        match=re.search(r'_(\d+)\.',name)
        return (0 if name.endswith('.hea') else 1, int(match[1]) if match else 0, name)
    return sorted(files,key=order)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):
            h.update(b)
    return h.hexdigest()

def download_file(session,url,dest,budget):
    stamp=dest.with_name(dest.name+'.download.json')
    if dest.is_file() and stamp.is_file():
        try:
            old=json.loads(stamp.read_text(encoding='utf-8'))
            if old['url']==url and old['size']==dest.stat().st_size and old['sha256']==digest(dest):
                return 'existing'
        except (ValueError,KeyError):
            pass
    dest.parent.mkdir(parents=True,exist_ok=True)
    partial=dest.with_name(dest.name+'.part')
    h=hashlib.sha256()
    size=0
    with session.get(url,stream=True,allow_redirects=False,timeout=(15,60)) as r:
        check_response(r,file=True)
        with partial.open('wb') as f:
            for b in chunks(r,budget):
                if size==0 and b.lstrip().lower().startswith((b'<!doctype html',b'<html')):
                    raise AccessError('HTML returned in place of a waveform file.')
                f.write(b)
                h.update(b)
                size+=len(b)
    if size==0:
        raise DownloadError('Empty waveform/header response.')
    partial.replace(dest)
    stamp.write_text(json.dumps({'url':url,'size':size,'sha256':h.hexdigest()}),encoding='utf-8')
    return 'downloaded'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cohort',type=Path,default=ROOT/'derived/mcmed_cohort_visits.csv')
    p.add_argument('--groups',nargs='+',default=['all'],help='Default: every group selected by 04, deduplicated')
    choose=p.add_mutually_exclusive_group()
    choose.add_argument('--csns',type=Path,help='CSN text file; overrides cohort/groups')
    choose.add_argument('--patients',nargs='+',help='MRNs: all their visits with positive Pleth duration')
    p.add_argument('--clinical-dir',type=Path,default=Path(os.environ.get("MCMED_DIR", "data/mc-med")))
    p.add_argument('--out',type=Path,default=Path(os.environ.get("MCMED_DIR", "data/mc-med"))/"data")
    p.add_argument('--limit',type=int)
    p.add_argument('--max-network-gib',type=float,default=25)
    p.add_argument('--retries',type=int,default=3,help='Maximum attempts per directory/file')
    p.add_argument('--save-plan',type=Path,help='Save the exact deduplicated CSNs locally, no network needed')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--inspect',action='store_true')
    mode.add_argument('--download',action='store_true')
    args=p.parse_args()
    if args.max_network_gib<=0 or args.retries<1 or (args.limit is not None and args.limit<1):
        p.error('Budget, retries and limit must be positive.')
    hours=None
    if args.csns:
        csns=list(dict.fromkeys(x.strip() for x in args.csns.read_text().splitlines() if x.strip()))
    else:
        if args.patients:
            v=pd.read_csv(args.clinical_dir/'visits.csv',usecols=['MRN','CSN'],dtype=str)
            missing=set(args.patients)-set(v.MRN)
            if missing:
                p.error(f'{len(missing)} requested MRNs not found.')
            w=pd.read_csv(args.clinical_dir/'waveform_summary.csv',dtype={'CSN':str})
            w=w[w.Type.eq('Pleth')].groupby('CSN',as_index=False).Duration.sum().rename(columns={'Duration':'pleth_s'})
            frame=v[v.MRN.isin(args.patients)].merge(w,on='CSN')
            print('Patient mode: all Pleth-positive visits; no index-visit selection.')
        else:
            frame=pd.read_csv(args.cohort,dtype={'CSN':str})
            if args.groups!=['all']:
                if set(args.groups)-set(frame.grp):
                    p.error('Unknown group name.')
                frame=frame[frame.grp.isin(args.groups)]
        if 'pleth_s' in frame:
            frame=frame[pd.to_numeric(frame.pleth_s,errors='coerce')>0]
        frame=frame.drop_duplicates('CSN')
        if args.limit:
            frame=frame.head(args.limit)
        csns=frame.CSN.tolist()
        hours=float(frame.pleth_s.sum()/3600) if 'pleth_s' in frame else None
    if args.limit:
        csns=csns[:args.limit]
    if not csns or any(not re.fullmatch(r'\d{3,}',x) for x in csns):
        p.error('No valid CSNs selected.')
    if args.save_plan:
        args.save_plan.parent.mkdir(parents=True,exist_ok=True)
        args.save_plan.write_text('\n'.join(csns)+'\n',encoding='utf-8')
    plan={'layout':'data/waveforms/{CSN_last3}/{CSN}/Pleth/', 'visits':len(csns),
          'total_hours':hours,'raw_GiB_if_125Hz_int16':hours*3600*250/1024**3 if hours is not None else None}
    print(json.dumps(plan,indent=2),flush=True)
    if not args.inspect and not args.download:
        print('Offline plan only. Add --inspect or --download.')
        return
    username=os.environ.get('PHYSIONET_USERNAME') or input('PhysioNet username: ').strip()
    password=getpass.getpass('PhysioNet password (not saved): ')
    budget=Budget(args.max_network_gib)
    args.out.mkdir(parents=True,exist_ok=True)
    report=args.out/('download_report.jsonl' if args.download else 'inspect_report.jsonl')
    failed=0
    with requests.Session() as session, report.open('a',encoding='utf-8') as log:
        session.auth=(username,password)
        session.headers.update({'Accept-Encoding':'identity','User-Agent':'MC-MED-selective-Pleth/2.0'})
        for i,csn in enumerate(csns,1):
            relative=f'{csn[-3:]}/{csn}/Pleth/'
            print(f'Visit {i}/{len(csns)}: reading selected Pleth directory',flush=True)
            try:
                files=retry(lambda:list_visit(session,BASE+relative,budget),args.retries)
            except (AccessError,BudgetError):
                raise
            except (DownloadError,requests.RequestException,TransportError) as exc:
                failed+=1
                log.write(json.dumps({'CSN':csn,'status':'directory_failed','error':str(exc)})+'\n')
                log.flush()
                print(f' Visit directory failed after attempts: {exc}',flush=True)
                continue
            statuses={'downloaded':0,'existing':0}
            errors=[]
            for name,url in files:
                if args.download:
                    try:
                        result=retry(lambda:download_file(session,url,args.out/relative/name,budget),args.retries)
                        statuses[result]+=1
                    except (AccessError,BudgetError):
                        raise
                    except (DownloadError,requests.RequestException,TransportError) as exc:
                        failed+=1
                        errors.append({'file':name,'error':str(exc)})
                        print(f' File failed: {name}: {exc}',flush=True)
                    print('.',end='',flush=True)
            log.write(json.dumps({'CSN':csn,'files':[name for name,_ in files],**statuses,
                                  'errors':errors,'network_bytes_this_run':budget.used})+'\n')
            log.flush()
            print(f' {len(files)} files; HTTP {budget.used/1024**3:.3f} GiB',flush=True)
    print(f'Finished: {args.out}')
    if failed:
        raise DownloadError(f'{failed} failed directories/files. See {report}; rerun to retry, completed files will be skipped.')

if __name__=='__main__':
    try:
        main()
    except (DownloadError,requests.RequestException,TransportError,OSError) as exc:
        print(f'STOPPED: {exc}',file=sys.stderr)
        sys.exit(1)
