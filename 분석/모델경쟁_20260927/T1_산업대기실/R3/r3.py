# T1 R3 공용 라이브러리 — 4년 진입창(h=4) 전진 검증 + 챔피언/도전자
# PROTOCOL_R3 §3: 🔒 최종시험 c=2019 → 진입 2020–2023(F'=2020). 개발·학습 레이블 진입창은 ≤2019에 끝나야 한다.
# R3 개발 정의(사전 고정, 2026-09-29):
#   - 진입창 h=4 (최종시험 2020–23과 같은 길이)
#   - 개발 원점 c = 2008–2012 (진입창 끝 ≤2016 < 2020). 보조 원점 c=2015(진입 2016–19)는 채택 판정에 쓰지 않는다(외부데이터 결측 설계 확인용).
#   - 원점 c의 학습 원점 = c−7..c−4 (학습 진입창 끝 ≤ c), 학습 레이블·분류기·IPW·공개 보강 cap = c-12-31
#   - 최종시험(B단계, 이 파일에서 실행하지 않음): c=2019, 학습 원점 2012–2015, cap 2019-12-31
# R1·R2 파일은 import·읽기만 한다. R2 캐시(원점별 분류기·패널)는 읽기 전용으로 재사용하고, 새 산출물은 R3/ 아래에만 쓴다.
import os, sys, json, time, pathlib
R3 = pathlib.Path(__file__).resolve().parent; T1 = R3.parent; R2 = T1/'R2'
sys.path.insert(0, str(T1)); sys.path.insert(0, str(R2)); sys.path.insert(0, str(R3))
import numpy as np, pandas as pd
import r2                                  # R2 공용(읽기 전용 import) — fr, lib, app_pub, prior_app, mets, skill, peak_gb, f32 등
from r2 import fr, lib, app_pub, prior_app, mets, skill, peak_gb, f32, BASE, SPEC1, FEATS1, CACHE, B
from sklearn.metrics import roc_auc_score

H = 4
ORIGINS = [2008, 2009, 2010, 2011, 2012]
AUX = [2015]
END = '20250822'
GUARD_DEV = 2019                           # 🔒 R3 최종시험 첫 결과연도 2020 미만
KEEP = r2.KEEP
T0 = time.time(); BUDGET = 520


def elapsed(): return time.time() - T0
def need(sec, what):
    if elapsed() + sec > BUDGET:
        print(f'[시간 예산] {what} 전에 중단(경과 {elapsed():.0f}s) — 같은 명령 재실행 시 이어짐', flush=True); sys.exit(3)


def train_cuts(c): return tuple(range(c - 7, c - 3))


def cfg_of(c):
    assert c + H <= GUARD_DEV, '🔒 R3 개발 원점은 진입창 끝 ≤2019'
    return dict(eval_cut=c, train_cuts=train_cuts(c), cap=f'{c}1231')


def fc_r3(c): p = R3/'cache'/f'o{c}'; p.mkdir(parents=True, exist_ok=True); return p
def fc_r2(c): return R2/'cache'/f'o{c}'


def feat_path(c, t, kind='feat'):
    """원점 c의 패널 t 캐시 경로: R3에 있으면 R3, 없으면 R2(읽기 전용)"""
    p3 = fc_r3(c)/f'{kind}_c{t}.parquet'
    if p3.exists(): return p3
    p2 = fc_r2(c)/f'{kind}_c{t}.parquet'
    if p2.exists(): return p2
    return None


# ---------------- 4년창 레이블 ----------------
def reg_lab(d, cap, h=H):
    """등록 기준: (cut, cut+h] 출원 범위안 특허가 cap까지 등록(원출원인 코드)되어 그 KSIC에 있으면 1 (build_panel과 같은 정의, cap=None이면 데이터 끝)"""
    ins = lib.load()['ins'] if 'ins' in lib._cache else None
    if ins is None: r2.light_load(); ins = lib._cache['ins']
    kidx = lib._cache['kidx']
    if cap is not None: ins = ins[ins.rgst_date <= cap]
    y = np.zeros(len(d), np.int8)
    for cut in sorted(d.cut.unique()):
        assert cut + h <= GUARD_DEV, '🔒 결과창 가드(R3)'
        e = ins[(ins.ay > cut) & (ins.ay <= cut + h)][['applicant_cd', 'ksic']].drop_duplicates()
        e = e.assign(k=e.ksic.map(kidx).astype(int), z=1)[['applicant_cd', 'k', 'z']]
        idx = np.nonzero(d.cut.values == cut)[0]
        x = d.iloc[idx][['firm', 'k']].merge(e, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left')
        y[idx] = x.z.fillna(0).values.astype(np.int8)
    return y


def pub_lab(d, cap, h=H):
    for cut in d.cut.unique(): assert cut + h <= GUARD_DEV, '🔒 결과창 가드(R3)'
    old = lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR; lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = GUARD_DEV
    try: return r2.pub_lab(d, cap, h)
    finally: lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = old


def ipw4(tr, cfg, y):
    """fr.ipw와 같은 식, 진입창만 h=4"""
    D = lib._cache; ins = D['ins']; kidx = D['kidx']; tp = D['fm'].set_index('applicant_cd').tp
    cdf = fr.lag_cdf(cfg); cap = cfg['cap']; w = np.ones(len(tr)); capd = pd.Timestamp(cap)
    for cut in sorted(tr.cut.unique()):
        fut = ins[(ins.ay > cut) & (ins.ay <= cut + H) & (ins.rgst_date <= cap)]
        fst = fut.groupby(['applicant_cd', 'ksic']).appl_date.min().reset_index(); fst['k'] = fst.ksic.map(kidx)
        idx = np.nonzero((tr.cut.values == cut) & (y == 1))[0]
        x = tr.iloc[idx][['firm', 'k']].merge(fst, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left')
        days = (capd - pd.to_datetime(x.appl_date)).dt.days.values; frn = x.firm.map(tp).eq('외국법인').values
        pr = np.array([np.searchsorted(cdf[g], dd, side='right') / len(cdf[g]) for dd, g in zip(days, frn)])
        w[idx] = np.minimum(1 / np.maximum(pr, 1e-3), SPEC1['ipw_wmax'])
    return w


# ---------------- 원점별 준비 ----------------
def prep(c):
    FC = fc_r3(c); ptr, pte = FC/'prep_tr.parquet', FC/'prep_te.parquet'
    if ptr.exists() and pte.exists(): return
    cf = cfg_of(c); r2.light_load()
    miss = [t for t in cf['train_cuts'] if feat_path(c, t) is None] + ([c] if feat_path(c, c) is None else [])
    assert not miss, f'패널 없음 {miss} — build_r3.py --c {c} 먼저'
    tr = pd.concat([pd.read_parquet(feat_path(c, t), columns=KEEP) for t in cf['train_cuts']], ignore_index=True)
    tr = f32(tr); tr['y_reg'] = reg_lab(tr, cf['cap']); tr['y'] = tr.y_reg.values
    tr['y_pub'] = pub_lab(tr, cf['cap']); tr['w_ipw'] = ipw4(tr, cf, tr.y_reg.values).astype(np.float32); tr['prior_pub'] = prior_app(tr, cf['cap'])
    tr.to_parquet(ptr, index=False); del tr
    te = f32(pd.read_parquet(feat_path(c, c), columns=KEEP))
    te['y_reg'] = reg_lab(te, None); te['y'] = te.y_reg.values
    te['y_pub'] = pub_lab(te, END); te['y_u'] = np.maximum(te.y_reg.values, te.y_pub.values).astype(np.int8)
    te['prior_any'] = prior_app(te, None)
    te.to_parquet(pte, index=False)
    print('prep saved', c, round(elapsed()), flush=True)


def prep5(c):
    """학습 원점 5개(c−8..c−4) 변형: prep_tr.parquet + c−8 패널(R2 캐시, 같은 cap·같은 4년 레이블)"""
    FC = fc_r3(c); p5 = FC/'prep_tr5.parquet'
    if p5.exists(): return
    prep(c); cf = cfg_of(c); r2.light_load(); t = c - 8
    x = f32(pd.read_parquet(feat_path(c, t), columns=KEEP))
    x['y_reg'] = reg_lab(x, cf['cap']); x['y'] = x.y_reg.values
    x['y_pub'] = pub_lab(x, cf['cap']); x['w_ipw'] = ipw4(x, cf, x.y_reg.values).astype(np.float32); x['prior_pub'] = prior_app(x, cf['cap'])
    tr = pd.read_parquet(FC/'prep_tr.parquet')
    pd.concat([x[tr.columns], tr], ignore_index=True).to_parquet(p5, index=False)
    print('prep5 saved', c, flush=True)


def load_prep(c, cols=None):
    FC = fc_r3(c)
    return pd.read_parquet(FC/'prep_tr.parquet', columns=cols), pd.read_parquet(FC/'prep_te.parquet', columns=cols)


# ---------------- 고정 기준선(4년창) ----------------
def baselines(c):
    FC = fc_r3(c); pj = FC/'baselines_r3.json'
    if pj.exists(): return json.load(open(pj, encoding='utf-8'))
    import xgboost as xgb
    cf = cfg_of(c); r2.light_load()
    btr = pd.concat([pd.read_parquet(feat_path(c, t, 'base_app')) for t in cf['train_cuts']], ignore_index=True)
    bte = pd.read_parquet(feat_path(c, c, 'base_app'))
    te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'y_reg', 'y_u', 'newc', 'prior_any'])
    btr['y'] = reg_lab(btr, cf['cap']); bte['y'] = reg_lab(bte, None)
    assert (bte.firm.values == te.firm.values).all() and (bte.k.values == te.k.values).all() and (bte.y.values == te.y_reg.values).all()
    btr['y_pub'] = pub_lab(btr, cf['cap']); btr['y_u'] = np.maximum(btr.y.values, btr.y_pub.values)
    keep = ~te.prior_any.values; yv = te.y_u.values[keep]; nc = te.newc.values[keep]; yr = te.y_reg.values[keep]
    S = {}
    for lab in ('y', 'y_u'):
        t2 = btr.assign(y=btr[lab].values)
        S[f'logit_stack_{lab}'] = fr.logit_pred(fr.logit_fit(t2, BASE), bte)
        S[f'logit_single_{lab}'] = fr.logit_pred(fr.logit_fit(t2[t2.cut == max(cf['train_cuts'])], BASE), bte)
        xb = xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, tree_method='hist', nthread=2, random_state=0)
        xb.fit(t2[BASE].values.astype(np.float32), t2.y.values); S[f'xgb_{lab}'] = xb.predict_proba(bte[BASE].values.astype(np.float32))[:, 1]
    res = {}
    for kk, s in S.items():
        res[kk] = mets(yv, s[keep], nc); res[kk]['auc_all_reglabel'] = float(roc_auc_score(yr, s[keep]))
    out = dict(origin=c, h=H, train_cuts=list(cf['train_cuts']), variants=res,
               strongest={m: max(v[m] for v in res.values()) for m in ('auc_all', 'pr_all', 'auc_new', 'pr_new', 'auc_all_reglabel')},
               strongest_arg={m: max(res, key=lambda q: res[q][m]) for m in ('auc_all', 'auc_new')})
    pd.DataFrame({kk: v.astype(np.float32) for kk, v in S.items()}).to_parquet(FC/'baseline_scores.parquet', index=False)
    json.dump(out, open(pj, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    return out
