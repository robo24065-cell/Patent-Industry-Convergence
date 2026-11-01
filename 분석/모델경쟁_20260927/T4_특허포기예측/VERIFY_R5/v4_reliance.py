# -*- coding: utf-8 -*-
"""누수 영향 추정(T4): 동결 저장 모형(runs_final/F22/models, 읽기만)으로 최종시험 행을 다시 예측하고,
시점 정합이 약한 피처 묶음을 망가뜨렸을 때 앙상블 AUC·skill이 얼마나 줄어드는지 본다(그 묶음에 기댄 정도 = 누수였다면 영향 상한).
- mac(KOSIS 광공업지수, 최신 개정·재기준 계열 사용): KSIC 사이에서 값을 통째로 뒤섞기(5시드) / 결측 처리
- fam(해외 패밀리, family 필드 스냅샷 — 공개연도 컷 적용됨): 같은 KSIC×경과연수 안에서 섞기(3시드)
- claims(최종 청구항 스냅샷): 같은 KSIC×경과연수 안에서 섞기(3시드) — 기준선도 같은 열을 쓰므로 기준선도 함께 다시 예측"""
import os, sys, json, pickle, time
sys.dont_write_bytecode = True
os.environ['T4R2_MODE'] = 'final'
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(TR, 'R2'))
import numpy as np, pandas as pd
import xgboost as xgb
import r2lib as R
from sklearn.metrics import roc_auc_score
FD = os.path.join(TR, 'R2', 'runs_final', 'F22'); MD = os.path.join(FD, 'models')
SPEC = json.load(open(os.path.join(TR, 'R2', 'final_spec_r2.json'), encoding='utf-8'))['spec']
CAT = R.CAT
t = time.time()
ev = pd.read_parquet(os.path.join(FD, 'ev.parquet'))
eh = pd.read_parquet(os.path.join(FD, 'ev_haz.parquet'))
y = ev.y.values.astype(int)
sk = lambda a, b: (a - b) / (1 - b)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
BOOST = {}
for comp in ['cls', 'haz']:
    key = R.comp_key(SPEC, comp)
    BOOST[comp] = []
    for k in range(3):
        b = xgb.Booster(); b.load_model(os.path.join(MD, f'{key}_b{k}.json')); b.set_param({'nthread': 2})
        BOOST[comp].append((b, pickle.load(open(os.path.join(MD, f'{key}_b{k}.pkl'), 'rb'))))
LOG = pickle.load(open(os.path.join(MD, f"{R.comp_key(SPEC, 'logit')}.pkl"), 'rb'))
HGB = pickle.load(open(os.path.join(MD, f"{R.comp_key(SPEC, 'hgb')}.pkl"), 'rb'))
BASE = pickle.load(open(os.path.join(MD, [f for f in os.listdir(MD) if f.startswith('base_')][0]), 'rb'))


def xgb_pred(comp, e):
    zs = []
    for b, meta in BOOST[comp]:
        X = e[meta['num'] + CAT].copy()
        for c in meta['num']:
            X[c] = X[c].astype('float32')
        X, _ = R._prep_cat(X, meta['cmap'])
        p = np.clip(b.predict(xgb.DMatrix(X, enable_categorical=True)), 1e-6, 1 - 1e-6)
        zs.append(np.log(p / (1 - p)))
    return 1 / (1 + np.exp(-np.mean(zs, 0)))


def ens(e, h):
    pc_ = xgb_pred('cls', e)
    ph = R.HZ.combine(pd.DataFrame({'row': h.row.values}), xgb_pred('haz', h), len(e))
    cfg = SPEC['components']['logit']
    pl = LOG['m'].predict_proba(LOG['prep'].transform(e))[:, 1]
    cfh = SPEC['components']['hgb']
    ce = HGB['oe'].transform(e[CAT].astype(str)).astype('float32'); ce[ce < 0] = np.nan
    Xe = np.c_[e[cfh['num']].to_numpy(dtype='float32', na_value=np.nan), ce]; Xe[:, HGB['allnan']] = 0
    phg = HGB['m'].predict_proba(Xe)[:, 1]
    ws = SPEC['ensemble']['weights']; P = dict(cls=pc_, haz=ph, logit=pl, hgb=phg)
    z = sum(w * lg(P[k]) for k, w in ws.items()) / sum(ws.values())
    pb = BASE['m'].predict_proba(R.base_X(e, BASE['med'], SPEC['baseline']))[:, 1]
    return 1 / (1 + np.exp(-z)), pb


part = sys.argv[1]
f0 = os.path.join(HERE, 'cache', 'saved_model_pred.npz')
if os.path.exists(f0):
    _z = np.load(f0); p0, b0 = _z['p'], _z['b']
else:
    p0, b0 = ens(ev, eh); np.savez(f0, p=p0, b=b0)
frz = json.load(open(os.path.join(TR, 'R2', 'final_result.json'), encoding='utf-8'))
a0, ab0 = roc_auc_score(y, p0), roc_auc_score(y, b0)
res = dict(saved_model_repro=dict(auc_model=a0, auc_base=ab0, skill=sk(a0, ab0), frozen_auc=frz['model']['auc'], frozen_skill=frz['skill_auc']))
print(res, round(time.time() - t), flush=True)
MAC = ['mac_prod_yoy', 'mac_ship_yoy', 'mac_inv_yoy', 'mac_prod_rel', 'mac_prod_3y']
FAM = ['fam_n', 'fam_cc', 'fam_us', 'fam_ep', 'fam_jp', 'fam_cn', 'fam_wo']
ks = ev.ksic.astype(str).values
grp = ks + '_' + pd.cut(ev.age, [-1, 3, 6, 9, 12, 99], labels=False).astype(str).values
pid2row = pd.Series(np.arange(len(ev)), index=ev.pid.values)


def run(label, fe, fh):
    p, b = ens(fe, fh)
    a, ab = roc_auc_score(y, p), roc_auc_score(y, b)
    return dict(label=label, auc_model=a, auc_base=ab, skill=sk(a, ab), d_skill=sk(a, ab) - sk(a0, ab0))


out = []
# mac: KSIC 간 통째 섞기
uk = np.unique(ks)
for s in (range(5) if part == 'mac' else []):
    rng = np.random.default_rng(s); perm = dict(zip(uk, rng.permutation(uk)))
    src = ev.groupby(ev.ksic.astype(str))[MAC].first()   # 같은 L이라 KSIC 안에서 mac 값은 같다(3자리 연결)
    fe = ev.copy(); fe[MAC] = src.reindex([perm[k] for k in ks]).values
    fh = eh.copy(); hk = fh.ksic.astype(str).values; fh[MAC] = src.reindex([perm.get(k, k) for k in hk]).values
    out.append(run(f'mac KSIC간 섞기 s{s}', fe, fh)); print(out[-1], round(time.time() - t), flush=True)
if part == 'mac':
    fe = ev.copy(); fe[MAC] = np.nan; fh = eh.copy(); fh[MAC] = np.nan
    out.append(run('mac 결측', fe, fh)); print(out[-1], flush=True)
# fam, claims: KSIC×경과연수 안에서 섞기(평가행 기준, 위험모형 결정행에는 같은 특허 값 전달)
for cols, nm in [(c_, n_) for c_, n_ in [(FAM, 'fam'), (['claims'], 'claims')] if n_ == part]:
    for s in range(3):
        rng = np.random.default_rng(100 + s)
        order = np.arange(len(ev))
        for g, ix in pd.Series(np.arange(len(ev))).groupby(grp).apply(np.array).items():
            order[ix] = rng.permutation(ix)
        fe = ev.copy(); fe[cols] = ev[cols].values[order]
        fh = eh.copy(); r = fh.row.values.astype(int)
        fh[cols] = fe[cols].values[r]
        out.append(run(f'{nm} KSIC×경과연수 안 섞기 s{s}', fe, fh)); print(out[-1], round(time.time() - t), flush=True)
fo = os.path.join(HERE, 'v4_reliance.json')
allr = json.load(open(fo, encoding='utf-8')) if os.path.exists(fo) else {}
allr.update(res); allr[part] = out
json.dump(allr, open(fo, 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
