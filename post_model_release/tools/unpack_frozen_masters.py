import argparse, hashlib, shutil, zipfile
from pathlib import Path
p=argparse.ArgumentParser(description="Extract verified frozen masters for post-model replay; no training")
p.add_argument('--archive',required=True,type=Path)
a=p.parse_args()
h=hashlib.sha256()
with a.archive.open('rb') as f:
    for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
if h.hexdigest()!='7ca27c87166ef33ba1c5ee74f6136ea125474a96543bf31d753d921c38194fd4':
    raise ValueError('Frozen archive checksum mismatch')
target=Path(__file__).resolve().parents[1]/'data/modeling'
if target.exists(): raise FileExistsError('Modeling directory already exists; nothing replaced')
names=[assay+'/'+split+'/5_all_features_2d_3d_morgan_with_metadata.csv'
       for assay in ['single_protein','cell_based'] for split in ['development','external']]
names+=['representation_columns.json']
with zipfile.ZipFile(a.archive) as z:
    for name in names: z.getinfo(name)
    for name in names:
        dest=target/name; dest.parent.mkdir(parents=True,exist_ok=True)
        with z.open(name) as src, dest.open('wb') as dst: shutil.copyfileobj(src,dst)
print('Four frozen masters extracted; no analyses executed.')
