# C18-T4 verify_0 r3: K1 경량 LR P0(b) 실패(제공 ≤ SCAGG)가 학습 사양에 견고한지 독립 반박(정보용; 사전 등록 사양은 바꾸지 않음).
# 변형 V1: 음성 상한 없음(전체 v-1 행, 가중 w), C=1 / V2: 상한 150,000, C=0.05(강한 규제) / V3: 상한 없음, 가중 없음.
# C12 캐시 u_{CN,EP}_{Y}.parquet 읽기만. 쓰기: verify_0\r3_out.json
import sys, json, time
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings('ignore')
t0 = time.time()
C12C = sys.argv[1]; OUT = sys.argv[2]; VS = [int(x) for x in sys.argv[3:]]
PUBF = ['f_log3', 'f_log1', 'f_gr', 'f_nsc', 'f_age', 'f_missc', 'is_cn', 's_n5', 's_grow', 's_nfirm', 's_newrate',
        'p_sec', 'p_relmean', 'p_relmax', 'p_cls', 'p_sect', 'p_cf', 'pi_rfirm', 'pi_rsc']
CTX = ['dot', 'cos', 'adopt', 'entry', 'qmain', 'mom_c', 'sind_c', 'mom_kr', 'sind_kr', 'mom_5', 'sind_5']
ARMS = ['prov', 'free', 'hsc', 'scagg']
cols = ['y', 'w'] + PUBF + [f'{a}_{x}' for a in ARMS for x in CTX]


def load(Y):
    return pd.concat([pd.read_parquet(f'{C12C}\\u_{cc}_{Y}.parquet', columns=cols) for cc in ['CN', 'EP']], ignore_index=True)


R = {}
for v in VS:
    tr0 = load(v - 1); ev = load(v)
    for vn, cap, Cr, usew in [('V1_nocap_C1', None, 1.0, True), ('V2_cap150k_C0.05', 150000, 0.05, True), ('V3_nocap_unweighted', None, 1.0, False)]:
        tr = tr0
        wtr = tr.w.values.astype(float)
        if cap is not None:
            rng = np.random.default_rng(99 + v)
            neg = np.flatnonzero(tr.y.values == 0)
            if len(neg) > 2 * cap:
                keep = np.ones(len(tr), bool); keep[rng.choice(neg, len(neg) - 2 * cap, replace=False)] = False
                fac = len(neg) / (2 * cap)
                wtr = np.where(tr.y.values == 0, wtr * fac, wtr)[keep]; tr = tr[keep]
        if not usew:
            wtr = np.ones(len(tr))
        auc = {}
        for arm in ['base'] + ARMS:
            cs = PUBF + ([f'{arm}_{x}' for x in CTX] if arm != 'base' else [])
            A = tr[cs].values.astype(float); Bm = ev[cs].values.astype(float)
            mu = np.nanmean(A, 0); A = np.where(np.isnan(A), mu, A); Bm = np.where(np.isnan(Bm), mu, Bm)
            sc = StandardScaler().fit(A)
            m = LogisticRegression(C=Cr, solver='lbfgs', max_iter=300).fit(sc.transform(A), tr.y.values, sample_weight=wtr)
            auc[arm] = roc_auc_score(ev.y.values, m.decision_function(sc.transform(Bm)), sample_weight=ev.w.values)
        R[f'{v}_{vn}'] = {'auc': auc, 'd_prov_scagg': auc['prov'] - auc['scagg'], 'd_prov_free': auc['prov'] - auc['free'], 'd_prov_hsc': auc['prov'] - auc['hsc']}
        print(v, vn, {k: round(x, 5) for k, x in auc.items()}, 'prov-scagg %.5f' % (auc['prov'] - auc['scagg']), round(time.time() - t0, 1), flush=True)
        if time.time() - t0 > 480:
            break
json.dump(R, open(OUT, 'w'), indent=1)
print('done', round(time.time() - t0, 1))
