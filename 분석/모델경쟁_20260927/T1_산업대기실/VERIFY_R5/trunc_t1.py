# T1 레드팀 — 레이블 창 끝(2023) PCT 국내단계 진입 지연 절단 민감도: 진입창을 2020–22(완결 관측)로 줄인 레이블로 같은 동결 점수 채점. 산출 trunc_t1.json
import json, pathlib, numpy as np, pandas as pd
from scipy.stats import rankdata
V = pathlib.Path(__file__).resolve().parent; T1 = V.parent; R3 = T1/'R3'; CACHE = T1/'cache'; END = '20250822'
def auc(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
sk = lambda a, b: (a - b) / (1 - b)
o = pd.read_parquet(R3/'runs'/'FROZEN_final'/'o2019.parquet', columns=['firm', 'k', 'px', 'pl', 'prior_any'])
K = sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique()); kidx = {k: i for i, k in enumerate(K)}
a = pd.read_parquet(CACHE/'app_pub.parquet', columns=['applicant_cd', 'ay', 'pub_date', 'rgst_date', 'ksic', 'appl_no']); a = a[a.ay >= 2000].copy()
a['k'] = a.ksic.astype(str).map(kidx); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int)
pdt = a.pub_date.fillna('').astype(str); rdt = a.rgst_date.fillna('').astype(str); a = a[((pdt.str.len() == 8) & (pdt <= END)) | ((rdt.str.len() == 8) & (rdt <= END))]
ins = pd.read_parquet(CACHE/'ins_firm.parquet', columns=['applicant_cd', 'ksic', 'ay']); ins['k'] = ins.ksic.map(kidx)
key = pd.MultiIndex.from_arrays([o.firm.values, o.k.values]); keep = ~o.prior_any.values; n = keep.sum()
s = 0.9 * rankdata(o.px.values[keep]) / n + 0.1 * rankdata(o.pl.values[keep]) / n
bs = pd.read_parquet(R3/'cache_final'/'o2019'/'baseline_scores.parquet').loc[keep]
fm = pd.read_parquet(CACHE/'firm_meta.parquet', columns=['applicant_cd', 'tp']).set_index('applicant_cd').tp; tp = o.firm.map(fm).values[keep]
out = {}
for lo, hi in ((2020, 2023), (2020, 2022)):
    ps = pd.MultiIndex.from_frame(a[(a.ay >= lo) & (a.ay <= hi)][['applicant_cd', 'k']].drop_duplicates())
    rs = pd.MultiIndex.from_frame(ins[(ins.ay >= lo) & (ins.ay <= hi)][['applicant_cd', 'k']].dropna().astype({'k': int}).drop_duplicates())
    y = np.maximum(key.isin(ps), key.isin(rs)).astype(int)[keep]
    r = {}
    for nm, m in (('전체', np.ones(n, bool)), ('국내법인', tp == '국내법인'), ('외국법인', tp == '외국법인')):
        am = auc(y[m], s[m]); b = max(auc(y[m], bs[c].values[m]) for c in bs.columns); r[nm] = dict(pos=int(y[m].sum()), auc=am, auc_strongest=b, skill=sk(am, b))
    out[f'{lo}-{hi}'] = r
pct = a[(a.ay >= 2019)].assign(pct=a.appl_no.astype(str).str.replace('-', '').str[6] == '7').groupby('ay').pct.mean()
out['matched_apps_pct_share_by_ay'] = {int(k_): float(v_) for k_, v_ in pct.items()}
json.dump(out, open(V/'trunc_t1.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print(json.dumps(out, ensure_ascii=False, indent=0))
