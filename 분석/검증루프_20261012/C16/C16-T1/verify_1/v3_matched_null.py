# -*- coding: utf-8 -*-
"""verify_1 v3: 법인 등록원부 규모(문서 수 구간) 맞춤 무작위 귀무. 플래그 220쌍과 같은 규모 분포로 플래그 안 된 P 양성 220쌍을 뽑아 제거했을 때 Δskill.
 맞춤 귀무도 +0.01 근처면 +0.011은 오매칭이 아니라 '작은 법인 양성은 낮게 매겨진다'는 선택 효과."""
import sys, json, pathlib, sqlite3
import numpy as np, pandas as pd
from scipy.stats import rankdata
sys.stdout.reconfigure(encoding='utf-8')
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실/R3'; FC = T1/'cache_final'/'o2019'; RD = T1/'runs'/'FROZEN_final'
HERE = B/'분석/검증루프_20261012/C16/C16-T1'; CC = HERE/'cache'; OUT = HERE/'verify_1'
BK = ['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'prior_any']); fz = pd.read_parquet(RD/'o2019.parquet', columns=['px', 'pl'])
bo = pd.read_parquet(FC/'baseline_scores.parquet'); L = pd.read_parquet(CC/'labels.parquet'); L = L[L.src == 'te'].reset_index(drop=True)
kept = ~te.prior_any.values; yk = L.y_u.values[kept].astype(np.int8); neg = yk == 0
firm_k = te.firm.astype(str).values[kept]; kk = te.k.values[kept]
s = 0.9 * pd.Series(fz.px.values[kept]).rank(pct=True).values + 0.1 * pd.Series(fz.pl.values[kept]).rank(pct=True).values
def contrib(v):
    sn = np.sort(v[neg]); sp = v[~neg]; lo = np.searchsorted(sn, sp, 'left'); hi = np.searchsorted(sn, sp, 'right'); return (lo + 0.5 * (hi - lo)) / neg.sum()
Cm = contrib(s); Cb = np.vstack([contrib(bo[c].values[kept]) for c in BK])
def sk(drop):
    keep = ~drop; a = Cm[keep].mean(); M0 = Cb[:, keep].mean(1).max(); return (a - M0) / (1 - M0)
pf = firm_k[~neg]; pk = kk[~neg]
Pm = (L.y_pub.values[kept] == 1) & (L.y_reg.values[kept] == 0); isP = Pm[~neg]
pP = pd.read_parquet(CC/'alias_pairs_P.parquet'); fl = set(zip(pP.firm.astype(str)[pP.all_wrong], pP.k[pP.all_wrong]))
isF = np.array([(f, k) in fl for f, k in zip(pf, pk)])
con = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri() + '?mode=ro', uri=True)
ra = pd.read_sql('select rgst_no, applicantCd from applicant', con); con.close(); ra['applicantCd'] = ra.applicantCd.astype(str)
nd = ra.groupby('applicantCd').rgst_no.nunique()
pnd = pd.Series(pf).map(nd).fillna(0).values
bins = [-1, 0, 1, 2, 5, 10, 30, 100, 10**9]; b = np.digitize(pnd, bins[1:], right=True)
base = sk(np.zeros(len(pf), bool)); flag = sk(isF)
rng = np.random.RandomState(20261018); res = []
need = pd.Series(b[isF]).value_counts()
pool = {g: np.where(isP & ~isF & (b == g))[0] for g in need.index}
for _ in range(2000):
    d = np.zeros(len(pf), bool)
    for g, n in need.items(): d[rng.choice(pool[g], n, replace=len(pool[g]) < n)] = True
    res.append(sk(d) - base)
res = np.array(res)
# 법인 규모 단독 효과: P 양성 중 작은 법인(nd<=10) vs 큰 법인 점수 백분위
pct = pd.Series(s).rank(pct=True).values[~neg]
out = dict(base=base, flag_delta=flag - base, matched_null=dict(mean=float(res.mean()), p2_5=float(np.percentile(res, 2.5)), p97_5=float(np.percentile(res, 97.5)),
           share_ge_flag=float((res >= flag - base).mean())), flag_bins=need.sort_index().to_dict(), P_bins=pd.Series(b[isP]).value_counts().sort_index().to_dict(),
           median_nd=dict(flag=float(np.median(pnd[isF])), P_unflag=float(np.median(pnd[isP & ~isF])), reg_pos=float(np.median(pnd[~isP]))),
           score_pct_by_bin={int(g): dict(flag=float(np.median(pct[isF & (b == g)])) if (isF & (b == g)).any() else None, P_unflag=float(np.median(pct[isP & ~isF & (b == g)])) if (isP & ~isF & (b == g)).any() else None) for g in sorted(set(b[isP]))})
json.dump(out, open(OUT/'v3_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
