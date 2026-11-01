# -*- coding: utf-8 -*-
"""r3oa 02: WIPO IPC 2026.01 영문 제목표 → 메인그룹 제목 (공개 정의문)"""
import zipfile, re, os, pandas as pd
OUT=os.path.dirname(os.path.abspath(__file__))
z=zipfile.ZipFile(os.path.join(OUT,'..','r3oa_raw','EN_ipc_title_list_20260101.zip'))
T={}
for n in z.namelist():
    for l in z.read(n).decode('utf-8','replace').splitlines():
        if '\t' in l:
            c,t=l.split('\t',1); T[c.strip()]=t.strip()
def key(mg):
    sc,g=mg.split(' '); return sc+g.zfill(4)+'000000'
M=pd.read_csv(os.path.join(OUT,'r3oa_mg_rank.csv'))
M['sc_title']=M.mg.str[:4].map(T)
M['mg_title']=M.mg.map(lambda m:T.get(key(m)))
M.to_csv(os.path.join(OUT,'r3oa_mg_rank_titles.csv'),index=False)
print('titles missing', M.mg_title.isna().sum(), 'of', len(M), '; top450 missing', M.mg_title.head(450).isna().sum())
