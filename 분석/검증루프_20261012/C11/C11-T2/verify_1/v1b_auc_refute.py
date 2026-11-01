# -*- coding: utf-8 -*-
# C11-T2 verify_1 (b): T3NC-MG TXT AUC 독립 재계산 (작업 스크립트 s5 미사용, 자체 Mann-Whitney 구현 + sklearn 교차)
import sys, json, os, hashlib
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
T3 = BASE + "/분석/모델경쟁_20260927/T3_기술셀부상예측"
WD = BASE + "/분석/검증루프_20261012/C11/C11-T2"
OUT = WD + "/verify_1/v1b_auc.json"
ORIG = list(range(2010, 2019))

def mw_auc(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    pos = p[y == 1]; neg = p[y == 0]
    # 정렬 기반 동률 0.5 처리
    allp = np.concatenate([pos, neg]); order = np.argsort(allp, kind='mergesort')
    s = allp[order]; ranks = np.empty(len(allp)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    n1 = len(pos); n0 = len(neg)
    return (ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def sha(f):
    return hashlib.sha1(open(f, 'rb').read()).hexdigest()[:12]

files = {
    'stored_R9': T3 + "/TXT_R9/results/mg_ctx-txt_preds.parquet",
    'r8': WD + "/res/runs/mg_ctx-txt__r8_preds.parquet",
    'reg': WD + "/res/runs/mg_ctx-txt__reg_preds.parquet",
    'reg_rep2': WD + "/res/runs/mg_ctx-txt__reg__rep2_preds.parquet",
    'pub': WD + "/res/runs/mg_ctx-txt__pub_preds.parquet",
    'pure': WD + "/res/runs/mg_ctx-txt__pure_preds.parquet",
    'prov_R8': T3 + "/NONCIRC_R8/results/mg_ctx-prov_preds.parquet",
    'conc_R8': T3 + "/NONCIRC_R8/results/mg_ctx-conc_preds.parquet",
    'none_R8': T3 + "/NONCIRC_R8/results/mg_ctx-none_preds.parquet",
}
R = {}
ref = pd.read_parquet(files['stored_R9'])
for k, f in files.items():
    d = pd.read_parquet(f)
    per = []; per_sk = []
    for v in ORIG:
        x = d[d.t0 == v].sort_values('cell')
        r = ref[ref.t0 == v].sort_values('cell')
        same_cells = bool(len(x) == len(r) and (x.cell.values == r.cell.values).all() and (x.yb.values == r.yb.values).all())
        per.append(mw_auc(x.yb.values, x.p.values)); per_sk.append(roc_auc_score(x.yb.values, x.p.values))
        if not same_cells: R.setdefault('cell_mismatch', []).append((k, v))
    maxdp = float(np.abs(d.sort_values(['t0','cell']).p.values - ref.sort_values(['t0','cell']).p.values).max()) if len(d)==len(ref) else None
    R[k] = dict(sha1=sha(f), n=len(d), mean_auc=float(np.mean(per)), mean_auc_sk=float(np.mean(per_sk)),
                r4=round(float(np.mean(per)), 4), per_origin=[round(a, 5) for a in per], maxdp_vs_stored=maxdp)
    if k == 'none_R8':
        bb = [mw_auc(d[d.t0 == v].yb.values, d[d.t0 == v].p_base.values) for v in ORIG]
        R['baseline'] = float(np.mean(bb))
    print(k, R[k]['sha1'], R[k]['n'], f"{R[k]['mean_auc']:.5f}", f"{R[k]['mean_auc_sk']:.5f}", R[k]['r4'], maxdp, flush=True)
print('baseline', R.get('baseline'))
print('cell_mismatch', R.get('cell_mismatch'))
# 원본 R9 요약의 txt 값
j = json.load(open(T3 + "/TXT_R9/summary_mg_r9.json", encoding='utf-8'))
R['summary_mg_r9_models'] = j.get('models')
print(json.dumps(j.get('models'), ensure_ascii=False)[:1500])
# 실행 설정 비교: reg/pub/pure json vs 저장 json (info 의 txt 외 설정)
def L(f): return json.load(open(f, encoding='utf-8'))
S = L(T3 + "/TXT_R9/results/mg_ctx-txt.json")
cfg = {}
for k in ['reg', 'pub', 'pure', 'r8']:
    J = L(WD + f"/res/runs/mg_ctx-txt__{k}.json")
    diffs = {}
    for kk in set(S.get('info', {})) | set(J.get('info', {})):
        a = S['info'].get(kk); b = J['info'].get(kk)
        if a != b: diffs[kk] = [str(a)[:120], str(b)[:120]]
    cfg[k] = dict(info_diff_keys=sorted(diffs), diffs=diffs, summary_auc=J['summary'].get('mean_auc'))
R['cfg'] = cfg
print(json.dumps({k: v['info_diff_keys'] for k, v in cfg.items()}, ensure_ascii=False))
json.dump(R, open(OUT, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
