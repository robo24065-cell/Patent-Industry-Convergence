# -*- coding: utf-8 -*-
import pathlib, json, sqlite3
import pandas as pd, numpy as np
HERE = pathlib.Path(__file__).resolve().parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
O = {}
ref = pd.read_csv(HERE.parent.parent/'analog_dropout.csv')
k = ref[ref.dim == 'KSIC']
O['ref_ksic_n'] = int(len(k)); O['ref_ksic_min'] = k.loc[k.dropout.idxmin()][['key','n_A','dropout']].astype(str).tolist(); O['ref_ksic_max'] = k.loc[k.dropout.idxmax()][['key','n_A','dropout']].astype(str).tolist()
O['ref_ksic_nA_min'] = int(k.n_A.min())
# 제공 KR 원본 17번째 열(산업분류코드)로 독립 KSIC 재집계 + 별칭 없는 생존
kr = pd.read_parquet(B/'공모전데이터/작업데이터/kr_registered.parquet'); c = list(kr.columns)
kr = kr[[c[3], c[4], c[2], c[17]]]; kr.columns = ['rn', 'rd', 'ay', 'ksic']
O['col17_sample'] = kr.ksic.astype(str).head(3).tolist()
kr['rn'] = kr.rn.astype(str); kr = kr[kr.rn.str[:2] == '10']
O['kr_dup_rn_rows'] = int(kr.rn.duplicated().sum()); O['kr_multi_ksic_rn'] = int((kr.groupby('rn').ksic.nunique() > 1).sum())
br = sqlite3.connect((B/'수집파이프라인/business_rights.sqlite').as_uri()+'?mode=ro', uri=True)
rg = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri()+'?mode=ro', uri=True)
OK = set(pd.read_sql("select applicant_cd from applicant_query where status='OK'", br).applicant_cd)
ap = pd.read_sql("select distinct rgst_no, applicantCd from applicant where rgst_no like '10%'", rg)
ap = ap[ap.applicantCd.isin(OK)]
rl = pd.read_sql("select distinct applicant_cd, rgst_no from right_list where rgst_no like '10%' and json_extract(raw,'$.pertDiv')='권리자란'", br)
s = set(zip(rl.applicant_cd, rl.rgst_no))
kd = kr.drop_duplicates('rn')
A = ap.merge(kd, left_on='rgst_no', right_on='rn')
A = A[(pd.to_numeric(A.ay, errors='coerce') >= 2000) & (A.rd.astype(str).str[:4].astype(int) <= 2019)].copy()
A['s'] = [(a, b) in s for a, b in zip(A.applicantCd, A.rgst_no)]
g = A.groupby('ksic').agg(n=('s', 'size'), d=('s', lambda x: 1 - x.mean()))
O['n'] = int(len(A)); O['ksic_n'] = int(len(g))
O['ksic_min'] = [str(g.d.idxmin()), int(g.loc[g.d.idxmin(), 'n']), float(g.d.min())]
O['ksic_max'] = [str(g.d.idxmax()), int(g.loc[g.d.idxmax(), 'n']), float(g.d.max())]
g5 = g[g.n >= 1000]; O['ksic_n1000_range'] = [float(g5.d.min()), float(g5.d.max()), int(len(g5))]
json.dump(O, open(HERE/'r2_ksic_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
