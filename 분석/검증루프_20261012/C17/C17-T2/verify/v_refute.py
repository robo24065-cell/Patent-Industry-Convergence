# -*- coding: utf-8 -*-
"""C17-T2 verify: 반박 시도(PREREG §10).
 R1 '보정하면 Δ가 0.002 아래' — 가장 유리한 가정 결합: 이전형=정확, 2순위=정확, NF 오매칭=코드 확정만(권리목록 보정), f를 칸별 부트스트랩 97.5% 상한으로(거짓 불일치 최대),
    그리고 F 정확 비율을 E1·E2 중 큰 값으로. → Δ(i) 기대값·구간
 R2 위약(placebo): F와 같은 칸 분포로 NF 쌍을 무작위 '가짜 F'로 지정해 같은 E2 절차를 돌리면 가짜 F 오매칭이 0 근처여야 한다(추정법 자체가 오류를 만들어내지 않는지). 100회.
 R3 출원 2020–21 쌍만(등록까지 시간이 길어 가시율 높음)으로 F 정확 비율 재추정.
 R4 반대 방향: Δ ≥ +0.0112(C16 상한형)일 확률.
 채점은 이 파일 안에서 따로 구현(양성 기여 searchsorted). s2의 estimate()는 소스를 읽어 그대로 실행(같은 추정법의 반박이므로).
 산출: verify/v_refute_out.json"""
import sys, pathlib, json, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent/'scripts'))
import guard17 as G
from c17common import *
t0 = time.time(); out = {}
src = open(pathlib.Path(__file__).resolve().parent.parent/'scripts'/'s2_estimate.py', encoding='utf-8').read()
ns = {'np': np, 'pd': pd, 'LogisticRegression': __import__('sklearn.linear_model', fromlist=['LogisticRegression']).LogisticRegression}
exec(src[src.index('CELLS = '):src.index('# ---- 주 추정 ----')], ns)
estimate, e1_wf, CELLS = ns['estimate'], ns['e1_wf'], ns['CELLS']
PJ = pd.read_parquet(CC/'pair_judg.parquet'); P = PJ[PJ.grp == 'P'].reset_index(drop=True)
Z = np.load(CC/'eval_base.npz', allow_pickle=True); s = Z['s']; y = Z['y']; base = Z['base']
key = pd.Series(np.arange(len(y)), index=pd.MultiIndex.from_arrays([Z['firm'], Z['k']]))
rows = key.reindex(pd.MultiIndex.from_arrays([P.firm.values, P.k.values])).values.astype(int)
pos = np.where(y == 1)[0]; neg = y == 0; n0 = neg.sum(); pm = -np.ones(len(y), int); pm[pos] = np.arange(len(pos)); prow = pm[rows]
def contrib(v):
    sn = np.sort(v[neg]); sp = v[pos]; lo = np.searchsorted(sn, sp, 'left'); hi = np.searchsorted(sn, sp, 'right'); return (lo + 0.5 * (hi - lo)) / n0
Cm = contrib(s); Cb = np.vstack([contrib(b) for b in base])
def sk(drop):
    k = np.ones(len(pos), bool); k[drop] = False; a = Cm[k].mean(); M0 = Cb[:, k].mean(1).max(); return (a - M0) / (1 - M0)
OFF = sk(np.array([], int))
def delta_dist(p, n=1000, seed=11):
    r = np.random.RandomState(seed); d = [sk(prow[r.rand(len(p)) < p]) - OFF for _ in range(n)]
    return dict(mean=float(np.mean(d)), p2_5=float(np.percentile(d, 2.5)), p97_5=float(np.percentile(d, 97.5)), share_le_0_002=float(np.mean(np.array(d) <= 0.002)))
out['OFF'] = OFF
# ---- R1 ----
rs = np.random.RandomState(5); fb = {c: [] for c in CELLS}
cid = (P.F.astype(int) * 100 + P.foreign.astype(int) * 10 + P['size'].map({'S': 0, 'M': 1, 'L': 2})).values; groups = [np.where(cid == g)[0] for g in np.unique(cid)]
for b in range(500):
    Xb = P.iloc[np.concatenate([g[rs.randint(0, len(g), len(g))] for g in groups])].reset_index(drop=True)
    r, _ = estimate(Xb, ret_p=False)
    for (cf, cs) in CELLS: fb[(cf, cs)].append(r['cells'][f'{"외국" if cf else "국내"}-{cs}']['f'])
fhi = {c: float(np.percentile(v, 97.5)) for c, v in fb.items()}
# f 상한을 넣은 E2(칸별), NF는 코드 확정만, 이전형=정확
Pv = P.copy(); Pv.loc[(Pv.pv == 'wrong') & (Pv.ptype == '권리이전형'), 'pv'] = 'correct'
WF = {}; pvec = np.where(Pv.pv == 'wrong', 1.0, 0.0)
e1res, _ = e1_wf(Pv)
for (cf, cs) in CELLS:
    c = (Pv.foreign.values == cf) & (Pv['size'].values == cs); F = Pv.F.values
    nF = (c & F).sum(); nNF = (c & ~F).sum(); kwN = (c & ~F & (Pv.pv.values == 'wrong')).sum(); kwF = (c & F & (Pv.pv.values == 'wrong')).sum()
    f = fhi[(cf, cs)]; C = (nNF - kwN) / (1 - f); wf_e2 = nF - min(nF, f * C); wf = max(kwF, min(wf_e2, e1res[(cf, cs)]))
    WF[f'{"외국" if cf else "국내"}-{cs}'] = dict(f_hi=f, WF_e2_fhi=float(wf_e2), WF_e1=float(e1res[(cf, cs)]), WF_used=float(wf))
    u = c & F & (Pv.pv.values == 'undet'); pvec[u] = np.clip((wf - kwF) / max(u.sum(), 1), 0, 1)
out['R1_favorable'] = dict(cells=WF, expected_removed=float(pvec.sum()), e=float(pvec.sum() / len(P)), delta=delta_dist(pvec))
# ---- R2 위약 ----
nfidx = {c: np.where((~P.F.values) & (P.foreign.values == c[0]) & (P['size'].values == c[1]))[0] for c in CELLS}
nFc = {c: int((P.F.values & (P.foreign.values == c[0]) & (P['size'].values == c[1])).sum()) for c in CELLS}
rp = np.random.RandomState(9); plc = []; plc_known = []
for it in range(100):
    X = P[~P.F].copy(); X['F'] = False
    for c in CELLS:
        ch = rp.choice(nfidx[c], min(nFc[c], len(nfidx[c])), replace=False); X.loc[ch, 'F'] = True
    r, pv_ = estimate(X.reset_index(drop=True))
    fake = X.reset_index(drop=True).F.values
    plc.append(float(r['W_F'])); plc_known.append(int((fake & (X.reset_index(drop=True).pv.values == 'wrong')).sum()))
out['R2_placebo'] = dict(fake_F_n=int(sum(nFc.values())), W_F_mean=float(np.mean(plc)), W_F_p97_5=float(np.percentile(plc, 97.5)), known_wrong_in_fake_mean=float(np.mean(plc_known)),
                         real_W_F=float(estimate(P, ret_p=False)[0]['W_F']))
# R2b 위약 편향 보정: 가짜 F에서 생긴 초과 오매칭(평균 W_F − 가짜 F 안 코드 확정 수)을 실제 W_F에서 빼고, 판정 불가 F 쌍의 p를 비례 축소
bias = out['R2_placebo']['W_F_mean'] - out['R2_placebo']['known_wrong_in_fake_mean']
_, pmain = estimate(P); uF = (P.pv.values == 'undet') & P.F.values
scale = max(0.0, (pmain[uF].sum() - bias) / pmain[uF].sum()); pb2 = pmain.copy(); pb2[uF] *= scale
out['R2b_bias_corrected'] = dict(bias=float(bias), scale=float(scale), expected_removed=float(pb2.sum()), e=float(pb2.sum() / len(P)), delta=delta_dist(pb2))
# ---- R3 2020–21 ----
X = P[P.ay_min <= 2021].reset_index(drop=True); r3, p3 = estimate(X)
rF = r3['W_F'] / max(int(X.F.sum()), 1); rmain = estimate(P, ret_p=False)[0]
out['R3_2020_21'] = dict(n=int(len(X)), nF=int(X.F.sum()), W_F=r3['W_F'], wrong_share_of_F=float(rF), main_wrong_share_of_F=float(rmain['W_F'] / P.F.sum()), e=r3['e'])
# ---- R4 ----
s3 = json.load(open(CC/'s3_out.json', encoding='utf-8'))
PB = np.load(CC/'p_boot_main.npy'); rr = np.random.RandomState(20261017)
d = np.array([sk(prow[rr.rand(len(P)) < PB[b]]) - OFF for b in range(PB.shape[0])])
out['R4_ge_c16'] = dict(share_ge_0_0112=float(np.mean(d >= 0.0112)), max=float(d.max()), mean_recomputed=float(d.mean()), s3_mean=s3['main']['excl']['mean'])
out['secs'] = round(time.time() - t0); out['peak'] = G.rss_peak()
json.dump(out, open(pathlib.Path(__file__).resolve().parent/'v_refute_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
