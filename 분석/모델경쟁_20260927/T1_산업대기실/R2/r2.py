# T1 R2 공용 라이브러리 — 전진 검증(개발 원점 2008–2012) + 챔피언/도전자 실험
# 🔒 최종시험(c=2017 → 진입 2018–2022, F=2018)의 결과변수는 이 파일에서 계산하지 않는다. 개발 원점은 c ≤ 2012로 강제한다.
import os, sys, json, time, pathlib, pickle, ctypes, ctypes.wintypes as wt, atexit
R2 = pathlib.Path(__file__).resolve().parent; T1 = R2.parent
sys.path.insert(0, str(T1)); sys.path.insert(0, str(R2))
import numpy as np, pandas as pd
import final_run as fr   # 1라운드 동결 함수(수정하지 않고 import만). T1_STRICT=1
import lib
from common import CACHE, B
from sklearn.metrics import roc_auc_score, average_precision_score

ORIGINS = [2008, 2009, 2010, 2011, 2012]
END = '20250822'                      # 제공 KR 데이터 최종일
BASE = fr.BASE                        # 고정 기준선 피처(트랙 정의)
SPEC1 = fr.SPEC                       # 1라운드 동결 스펙(v38)
DERIVED = ('k_prior', 'k_prior_new', 'f_any')
FEATS1 = [c for c in SPEC1['cols'] if c not in DERIVED]
T0 = time.time(); BUDGET = 520


def elapsed(): return time.time() - T0
def need(sec, what):
    if elapsed() + sec > BUDGET:
        print(f'[시간 예산] {what} 전에 중단(경과 {elapsed():.0f}s) — 같은 명령 재실행 시 이어짐', flush=True); sys.exit(3)


class _PMC(ctypes.Structure):
    _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
def peak_gb():
    c = _PMC(); c.cb = ctypes.sizeof(_PMC); k = ctypes.WinDLL('kernel32'); ps = ctypes.WinDLL('psapi')
    k.GetCurrentProcess.restype = wt.HANDLE; ps.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(_PMC), wt.DWORD]
    ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb); return c.PeakWorkingSetSize / 2**30
def _atexit_mem():
    try: print(f'[mem] peak working set {peak_gb():.2f} GB', flush=True)
    except Exception: pass
atexit.register(_atexit_mem)


def _mode(final, mode): return mode or ('final' if final else 'dev')
GUARD = {'dev': 2017, 'final': 2022, 'future': 2029}


def cfg_of(c, final=False, mode=None):
    mode = _mode(final, mode)
    if mode == 'future':
        # 미래 트렌드: 평가 c=2024(→2025–29), 학습 원점 2014–17(진입창 끝 ≤2022 → 데이터 끝(2025-08)까지 공개 완결), 레이블 cap=데이터 끝
        assert c == 2024; return dict(eval_cut=2024, train_cuts=(2014, 2015, 2016, 2017), cap=END)
    if mode == 'final': assert c == 2017
    else: assert c <= 2012, '🔒 개발 원점은 c ≤ 2012'
    return dict(eval_cut=c, train_cuts=tuple(range(c - 8, c - 4)), cap=f'{c}1231')


def fc_of(c, final=False, mode=None):
    mode = _mode(final, mode)
    return {'final': R2/'cache_final', 'future': R2/'cache_future', 'dev': R2/'cache'}[mode]/f'o{c}'


def f32(d):
    for col in d.columns:
        if d[col].dtype == np.float64: d[col] = d[col].astype(np.float32)
    return d


# ---------------- 레이블·위험집합 보조 ----------------
_APP = {}
def app_pub():
    if 'a' in _APP: return _APP['a']
    a = pd.read_parquet(CACHE/'app_pub.parquet'); K = lib.load()['K']; kidx = {k: i for i, k in enumerate(K)}
    a['k'] = a.ksic.astype(str).map(kidx); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int)
    a['pub_date'] = a.pub_date.fillna('').astype(str); a['rgst_date'] = a.rgst_date.fillna('').astype(str)
    a = a[a.ay >= 2000]; _APP['a'] = a; return a


def pub_lab(d, cap, h=5):
    """(cut, cut+h] 출원 범위안 공개·등록 출원(법인 별칭 매칭)이 그 KSIC에 있으면 1. cap까지 공개 또는 등록된 것만."""
    a = app_pub(); vis = ((a.pub_date.str.len() == 8) & (a.pub_date <= cap)) | ((a.rgst_date.str.len() == 8) & (a.rgst_date <= cap)); a = a[vis]
    y = np.zeros(len(d), np.int8)
    for cut in sorted(d.cut.unique()):
        assert cut + h <= lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR, '🔒 결과창 가드'
        e = a[(a.ay > cut) & (a.ay <= cut + h)].drop_duplicates(['applicant_cd', 'k'])[['applicant_cd', 'k']].assign(z=1)
        idx = np.nonzero(d.cut.values == cut)[0]
        x = d.iloc[idx][['firm', 'k']].merge(e, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left')
        y[idx] = x.z.fillna(0).values.astype(np.int8)
    return y


def prior_app(d, cap):
    """쌍(법인,KSIC)에 기준시점(cut) 이하 출원된 범위안 공개·등록 출원이 이미 있으면 True (cap까지 가시; cap=None이면 데이터 끝까지)."""
    a = app_pub()
    if cap is not None:
        a = a[((a.pub_date.str.len() == 8) & (a.pub_date <= cap)) | ((a.rgst_date.str.len() == 8) & (a.rgst_date <= cap))]
    out = np.zeros(len(d), bool)
    for cut in sorted(d.cut.unique()):
        e = a[a.ay <= cut].drop_duplicates(['applicant_cd', 'k'])[['applicant_cd', 'k']].assign(z=1)
        idx = np.nonzero(d.cut.values == cut)[0]
        x = d.iloc[idx][['firm', 'k']].merge(e, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left')
        out[idx] = x.z.fillna(0).values > 0
    return out


# ---------------- 원점별 준비 캐시 ----------------
KEEP = list(dict.fromkeys(['firm', 'k', 'y', 'newc', 'cut'] + BASE + FEATS1))


def prep(c, final=False, mode=None):
    mode = _mode(final, mode); final = mode != 'dev'
    FC = fc_of(c, mode=mode); ptr, pte = FC/'prep_tr.parquet', FC/'prep_te.parquet'
    if ptr.exists() and pte.exists(): return
    cf = cfg_of(c, mode=mode)
    lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = GUARD[mode]
    tr = pd.concat([pd.read_parquet(FC/f'feat_c{t}.parquet', columns=KEEP) for t in cf['train_cuts']], ignore_index=True)
    tr = f32(tr); tr['y_reg'] = tr.y.astype(np.int8)
    tr['y_pub'] = pub_lab(tr, cf['cap']); tr['w_ipw'] = fr.ipw(tr, cf).astype(np.float32); tr['prior_pub'] = prior_app(tr, cf['cap'])
    tr.to_parquet(ptr, index=False); del tr
    te = f32(pd.read_parquet(FC/f'feat_c{c}.parquet', columns=KEEP))
    te['y_reg'] = te.y.astype(np.int8)
    if final:
        te['y_pub'] = 0; te['y_u'] = te.y_reg   # 🔒 final 모드에서는 평가 레이블을 final_run_r2.py에서만 산출
    else:
        te['y_pub'] = pub_lab(te, END); te['y_u'] = np.maximum(te.y_reg.values, te.y_pub.values).astype(np.int8)
    te['prior_any'] = prior_app(te, None)
    te.to_parquet(pte, index=False)
    print('prep saved', c, round(elapsed()), flush=True)


def load_prep(c, final=False, cols=None, mode=None):
    FC = fc_of(c, final, mode)
    tr = pd.read_parquet(FC/'prep_tr.parquet', columns=cols); te = pd.read_parquet(FC/'prep_te.parquet', columns=cols)
    return tr, te


# ---------------- 지표 ----------------
def mets(y, s, nc):
    y = np.asarray(y); s = np.asarray(s); nc = np.asarray(nc, bool)
    return dict(auc_all=float(roc_auc_score(y, s)), pr_all=float(average_precision_score(y, s)),
                auc_new=float(roc_auc_score(y[nc], s[nc])), pr_new=float(average_precision_score(y[nc], s[nc])))


def skill(m, b): return (m - b) / (1 - b)


# ---------------- 고정 기준선(원점별 1회, 캐시) ----------------
def baselines(c, final=False):
    FC = fc_of(c, final); pj = FC/'baselines_r2.json'
    if pj.exists(): return json.load(open(pj, encoding='utf-8'))
    import xgboost as xgb
    cf = cfg_of(c, final)
    btr = pd.concat([pd.read_parquet(FC/f'base_app_c{t}.parquet') for t in cf['train_cuts']], ignore_index=True)
    bte = pd.read_parquet(FC/f'base_app_c{c}.parquet')
    te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'y_reg', 'y_u', 'newc', 'prior_any', 'lsize'])
    assert (bte.firm.values == te.firm.values).all() and (bte.k.values == te.k.values).all() and (bte.y.values == te.y_reg.values).all()
    # 주의: 1라운드도 기준선 점수는 출원 기준 lsize로 산출(가시 lsize는 vis3 보조지표 부분집합 정의에만 사용)
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
        if final: res[kk] = None; continue
        res[kk] = mets(yv, s[keep], nc); res[kk]['auc_all_reglabel'] = float(roc_auc_score(yr, s[keep])); res[kk]['auc_all_R1def'] = float(roc_auc_score(te.y_reg.values, s))
    out = dict(origin=c, variants=res)
    if not final:
        out['strongest'] = {m: max(v[m] for v in res.values()) for m in ('auc_all', 'pr_all', 'auc_new', 'pr_new', 'auc_all_reglabel')}
        out['strongest_arg'] = {m: max(res, key=lambda q: res[q][m]) for m in ('auc_all', 'auc_new')}
    pd.DataFrame({kk: v.astype(np.float32) for kk, v in S.items()}).to_parquet(FC/'baseline_scores.parquet', index=False)
    json.dump(out, open(pj, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    return out


def light_load():
    """lib.load()와 같은 ins·K·fm을 채우되 범위밖 제목 표(oos, 약 150만 행 문자열)는 싣지 않는다(학습 단계 메모리 절약). T1_VIS='rgst' 전제."""
    if 'ins' in lib._cache: return
    assert lib.VIS == 'rgst'
    ins = pd.read_parquet(CACHE/'ins_firm.parquet'); ins = ins[ins.ay >= 2000]
    K = sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique())
    lib._cache.update(ins=ins.assign(vis=ins.rgst_date.values), oos=None, K=K, kidx={k: i for i, k in enumerate(K)}, fm=pd.read_parquet(CACHE/'firm_meta.parquet'))
