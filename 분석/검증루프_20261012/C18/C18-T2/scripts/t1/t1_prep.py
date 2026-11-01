# -*- coding: utf-8 -*-
r"""C18-T2 T1(R3E29 동결) 민감도 준비 — EP 이중 수록(V1E)이 닿는 ep_* 5피처만 교체한 prep 사본.
 공식 R3/cache_final/o2019 의 prep_tr·prep_te·P_design·P_um(+index)·baseline_scores 를 이 폴더 cache/t1/<VAR>/FC 로 복사(읽기만),
 ep_* 는 final_run.py cut_features 의 EP 블록과 같은 식으로 ep_firm.parquet 에서 다시 계산:
   P0 = 공식 ep_firm 그대로(재계산값이 저장값과 float32에서 같아야 함 = 재현 조건), V1E = ep_firm 에서 EP 짧은 형식(9자·8자) 이중 수록 행 제거.
 사용: python t1_prep.py P0|V1E"""
import os, sys, shutil, hashlib, json, time
sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lib18 import *
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq
VAR = sys.argv[1]; t = time.time()
T1R3 = MC + r'\T1_산업대기실\R3'; OFC = T1R3 + r'\cache_final\o2019'
FC = os.path.join(C, 't1', VAR, 'FC'); os.makedirs(FC, exist_ok=True)
h = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()
for f in ['P_design.npy', 'P_design_index.parquet', 'P_um.npy', 'P_um_index.parquet', 'baseline_scores.parquet']:
    shutil.copyfile(os.path.join(OFC, f), os.path.join(FC, f)); assert h(os.path.join(OFC, f)) == h(os.path.join(FC, f))
ep = pd.read_parquet(os.path.join(OFC, 'ep_firm.parquet'))
rep = {'ep_firm_rows': int(len(ep))}
if VAR == 'V1E':
    pr = pd.read_parquet(os.path.join(C, 'other_dup_pairs.parquet'))
    pr = pr[pr.cc == 'EP']; short = set(np.where(pr.L1 < pr.L2, pr.n1, pr.n2))
    hit = ep.appl_no.isin(short); rep.update(ep_rows_removed=int(hit.sum()), firms_hit=int(ep.loc[hit, 'applicant_cd'].nunique()),
                                            removed_appl_year=ep.loc[hit, 'appl_year'].value_counts().sort_index().to_dict())
    ep = ep[~hit].reset_index(drop=True)
K = sorted(ep.ksic.astype(str).unique()) if False else None
# KSIC 순서: T1 lib.load()['K'] = 범위안 50 KSIC 정렬 (forecast_future 의 대체 경로와 같음)
K = sorted(pd.read_parquet(MC + r'\T1_산업대기실\cache\ins_pat.parquet', columns=['ksic']).ksic.astype(str).unique()); assert len(K) == 50
kidx = {k: i for i, k in enumerate(K)}
EPF = ['ep_k', 'ep_n', 'ep_share', 'ep_k_rec', 'ep_krprio']


def ep_cols(d):
    out = {c: np.zeros(len(d)) for c in EPF}
    for cut in sorted(d.cut.unique()):
        ii = np.nonzero(d.cut.values == cut)[0]; firm = d.firm.values[ii]; k = d.k.values[ii]
        e = ep.copy(); e['k'] = e.ksic.astype(str).map(kidx); e = e.dropna(subset=['k']); e['k'] = e.k.astype(int)
        e = e[(e.pub_date.astype(str) <= f'{cut}1231') & (e.appl_year >= 2000) & e.pub_date.notna()].drop_duplicates(['applicant_cd', 'appl_no', 'k'])
        def pairc(sub):
            s = sub.groupby(['applicant_cd', 'k']).size()
            return s.reindex(pd.MultiIndex.from_arrays([firm, k])).fillna(0).values
        ck = pairc(e); cr = pairc(e[e.appl_year > cut - 3]); ckr = pairc(e[e.prio_cc.fillna('').str.contains('KR')])
        n = pd.Series(firm).map(e.drop_duplicates(['applicant_cd', 'appl_no']).groupby('applicant_cd').size()).fillna(0).values
        out['ep_k'][ii] = np.log1p(ck); out['ep_n'][ii] = np.log1p(n); out['ep_share'][ii] = ck / np.maximum(n, 1)
        out['ep_k_rec'][ii] = np.log1p(cr); out['ep_krprio'][ii] = np.log1p(ckr)
    return out


for f in ['prep_tr.parquet', 'prep_te.parquet']:
    tb = pq.read_table(os.path.join(OFC, f))
    d = tb.select(['firm', 'k', 'cut']).to_pandas()
    new = ep_cols(d); r = {}
    for c in EPF:
        old = tb.column(c).to_numpy(zero_copy_only=False)
        nv = new[c].astype(old.dtype)
        r[c] = dict(maxabs_vs_stored=float(np.nanmax(np.abs(nv.astype(np.float64) - old.astype(np.float64)))), n_changed=int((nv != old).sum()),
                    rows_pos=int((old > 0).sum()))
        if VAR != 'P0':
            tb = tb.set_column(tb.schema.get_field_index(c), c, pa.array(nv))
    rep[f] = r
    if VAR == 'P0':
        assert all(v['maxabs_vs_stored'] == 0 for v in r.values()), r   # 재현 조건
        shutil.copyfile(os.path.join(OFC, f), os.path.join(FC, f))
    else:
        pq.write_table(tb, os.path.join(FC, f), compression='zstd')
    del tb, d; out(f, json.dumps(r), round(time.time() - t))
json.dump(rep, open(os.path.join(C, 't1', f'prep_{VAR}.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
out(json.dumps(rep, ensure_ascii=False, default=str))
