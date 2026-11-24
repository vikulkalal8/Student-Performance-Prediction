"""Download the official UCI archive or import a downloaded ZIP/CSV."""
import argparse, io, urllib.request, zipfile
from pathlib import Path
import pandas as pd
from model import ROOT, UCI
URL='https://archive.ics.uci.edu/static/public/320/student+performance.zip'

def extract(raw, depth=0):
    if depth>3: raise ValueError('Unexpected nested archive.')
    if not zipfile.is_zipfile(io.BytesIO(raw)): return raw
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            if name.split('/')[-1]=='student-mat.csv': return z.read(name)
        for name in z.namelist():
            if name.split('/')[-1]=='student.zip': return extract(z.read(name),depth+1)
    raise ValueError('student-mat.csv not found in archive.')

def install(raw):
    data=extract(raw); df=pd.read_csv(io.BytesIO(data),sep=';')
    if len(df)!=395 or not set(UCI+['G3']).issubset(df.columns): raise ValueError('Expected official 395-row UCI mathematics CSV.')
    dest=ROOT/'dataset'/'student-mat.csv'; dest.write_bytes(data); return dest
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('file',nargs='?',help='Downloaded UCI ZIP or student-mat.csv'); a=p.parse_args()
    try:
        if a.file: raw=Path(a.file).read_bytes()
        else:
            with urllib.request.urlopen(URL,timeout=30) as response: raw=response.read()
        print('Installed:',install(raw))
    except Exception as e: p.exit(1,f'Import failed: {e}\nDownload from https://archive.ics.uci.edu/dataset/320/student+performance and run python fetch_dataset.py PATH_TO_ZIP\n')
