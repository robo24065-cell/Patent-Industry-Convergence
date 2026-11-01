# -*- coding: utf-8 -*-
r"""C18-T2 T1 채점(공식 score_final 의 주 모형 지표 경로 사본, 쓰기는 이 폴더만) + 짝지은 법인 군집 부트스트랩(B=1000, seed 0).
 사용: python t1_score.py point            → P0·V1E 점추정(cache/t1/score_point.json), 평가 레이블·기준선은 공식 규칙 그대로
       python t1_score.py boot <chunk>     → 100회씩(chunk 0..9) cache/t1/boot_<chunk>.npy
       python t1_score.py summarize        → cache/t1/score_boot.json
 skill = (AUC_모형 − AUC_최강기준선)/(1 − AUC_최강기준선), 최강기준선 = 고정 기준선 6변형 중 최대(부트스트랩 재표본마다 다시)."""
import os, sys, json, time, importlib.util, pathlib
sys.dont_write_bytecode = True
W = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C18\C18-T2'
R3 = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T1_산업대기실\R3'
T = os.path.join(W, 'cache', 't1'); mode = sys.argv[1]; t0 = time.time()
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
LAB = os.path.join(T, 'eval_labels.parquet')


def load_scores():
    lab = pd.read_parquet(LAB)
    S = {}
    for V in ('P0', 'V1E'):
        o = pd.read_parquet(os.path.join(T, V, 'RD', 'o2019.parquet'))
        assert (o.firm.values == lab.firm.values).all() and (o.k.values == lab.k.values).all()
        S[V] = o
    return lab, S


def rank(x):
    from scipy.stats import rankdata
    return rankdata(x) / len(x)


if mode == 'point':
    sys.argv = ['final_run_r3.py']
    spec = importlib.util.spec_from_file_location('frr3', os.path.join(R3, 'final_run_r3.py')); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    M = m.MODES['final']
    te = pd.read_parquet(os.path.join(T, 'P0', 'FC', 'prep_te.parquet'), columns=['firm', 'k', 'cut', 'newc', 'prior_any'])
    if not os.path.exists(LAB):
        m.r2.light_load()
        with m.guard(M['guard_eval']):
            te['y_reg'] = m.r3.reg_lab(te, None); te['y_pub'] = m.r3.pub_lab(te, m.END)
        te['y_u'] = np.maximum(te.y_reg.values, te.y_pub.values); te.to_parquet(LAB, index=False)
    lab, S = load_scores()
    keep = ~lab.prior_any.values; t = lab[keep].reset_index(drop=True)
    bs = pd.read_parquet(os.path.join(T, 'P0', 'FC', 'baseline_scores.parquet')).loc[keep].reset_index(drop=True)
    y = t.y_u.values; nc = t.newc.values.astype(bool)
    bm = {kk: m.mets(y, bs[kk].values, nc) for kk in bs.columns}
    strongest = {q: max(v[q] for v in bm.values()) for q in ('auc_all', 'auc_new')}
    res = dict(n_rows=int(len(t)), pos=int(y.sum()), baseline_strongest=strongest)
    for V, o in S.items():
        oo = o[keep].reset_index(drop=True); s = m.blend(oo); mm = m.mets(y, s, nc)
        res[V] = dict(auc_all=mm['auc_all'], auc_new=mm['auc_new'], skill=m.skill(mm['auc_all'], strongest['auc_all']), skill_new=m.skill(mm['auc_new'], strongest['auc_new']))
        np.save(os.path.join(T, f'score_{V}.npy'), np.asarray(s, dtype=np.float64))
    off = json.load(open(os.path.join(R3, 'final_result_r3.json'), encoding='utf-8'))
    res['official_skill'] = off['skill']; res['P0_minus_official'] = res['P0']['skill'] - off['skill']['auc_all']
    res['delta_V1E'] = res['V1E']['skill'] - res['P0']['skill']
    bs[['logit_stack_y', 'logit_single_y', 'xgb_y', 'logit_stack_y_u', 'logit_single_y_u', 'xgb_y_u']].to_parquet(os.path.join(T, 'bs_keep.parquet'))
    pd.DataFrame(dict(firm=t.firm.values, y=y)).to_parquet(os.path.join(T, 'yfirm_keep.parquet'))
    json.dump(res, open(os.path.join(T, 'score_point.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False), round(time.time() - t0))
elif mode == 'boot':
    ch = int(sys.argv[2]); B_CH = 100
    yf = pd.read_parquet(os.path.join(T, 'yfirm_keep.parquet')); y = yf.y.values.astype(bool)
    bs = pd.read_parquet(os.path.join(T, 'bs_keep.parquet'))
    S = [np.load(os.path.join(T, 'score_P0.npy')), np.load(os.path.join(T, 'score_V1E.npy'))] + [bs[c].values.astype(np.float64) for c in bs.columns]
    fr_, inv = np.unique(yf.firm.values, return_inverse=True); nF = len(fr_)
    # 점수별 정렬 순서·동률 묶음 미리 계산 → 가중 AUC(가중 = 재표본 법인 중복 수) O(n)
    prep = []
    for s in S:
        o = np.argsort(s, kind='mergesort'); ss = s[o]; g = np.r_[0, np.cumsum(ss[1:] != ss[:-1])]
        prep.append((o, g, y[o], inv[o]))
    def wauc(o, g, yo, io, cnt):
        w = cnt[io]; wp = np.where(yo, w, 0.0); wn = np.where(yo, 0.0, w)
        G = g[-1] + 1; gp = np.bincount(g, wp, G); gn = np.bincount(g, wn, G)
        below = np.cumsum(gn) - gn
        return float((gp * (below + 0.5 * gn)).sum() / (gp.sum() * gn.sum()))
    rng = np.random.default_rng([0, ch]); A = np.zeros((B_CH, len(S)))
    for r in range(B_CH):
        cnt = np.bincount(rng.integers(0, nF, nF), minlength=nF).astype(np.float64)
        A[r] = [wauc(*p, cnt) for p in prep]
    np.save(os.path.join(T, f'boot_{ch}.npy'), A); print('chunk', ch, round(time.time() - t0))
else:
    A = np.concatenate([np.load(os.path.join(T, f'boot_{c}.npy')) for c in range(10)])
    b = A[:, 2:].max(1); sk = (A[:, :2] - b[:, None]) / (1 - b[:, None]); d = sk[:, 1] - sk[:, 0]
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    # 점검: 가중 AUC 코드 = sklearn (재표본 없이 가중치 1)
    res = dict(B=len(A), skill_P0_ci=q(sk[:, 0]), skill_V1E_ci=q(sk[:, 1]), delta_ci=q(d), delta_mean=float(d.mean()))
    json.dump(res, open(os.path.join(T, 'score_boot.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1); print(json.dumps(res))
