# -*- coding: utf-8 -*-
"""P0 사전 점검(결과 창 사건은 세지 않음): (a) 예측 재현 (c) 누수 (d) 시차·창.
- 창 안 실효(E > 2025-07-01) 사건은 이 스크립트에서 집계하지 않는다. 시차 분석은 E <= 2024-12-31만.
출력: cache/s02_p0.json"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import c08lib as C
t0 = time.time()
D = C.load()
pt, pay, rt = D['pt'], D['pay'], D['rt']
R = dict(n_pt=len(pt), n_pay=len(pay), n_rt=len(rt), n_patent_rows=int(pt.n_all_rows.iloc[0]))
L = C.L0
# ---------------- (a) 위험집합 독립 재구성 대조
rp, pthL, lvL = C.risk_set(D, L)
pr = pd.read_parquet(os.path.join(C.FUT, "pred_20250701.parquet"))
ev = pd.read_parquet(os.path.join(C.FUT, "ev.parquet"), columns=["pid", "ksic", "atype", "acd", "paid_year", "paid_ahead", "n_renew", "n_late",
                                                                    "yrs_since_pay", "n_prev_lapse", "n_recov", "ksic_lapse_1y", "a_lapse_rate2", "age", "claims", "rgst_year"])
s_ind, s_fz = set(rp.tolist()), set(pr.pid.tolist())
R['risk_set'] = dict(n_indep=len(rp), n_frozen=len(pr), only_indep=len(s_ind - s_fz), only_frozen=len(s_fz - s_ind),
                     jaccard=len(s_ind & s_fz) / len(s_ind | s_fz))
m = pt.set_index("pid").loc[pr.pid.values]
R['risk_set']['ksic_mismatch'] = int((m.ksic.astype(str).values != pr.ksic.astype(str).values).sum())
R['risk_set']['acd_mismatch'] = int((m.acd.astype(str).values != pr.acd.astype(str).values).sum())
R['risk_set']['atype_mismatch'] = int((m.atype.astype(str).values != pr.atype.astype(str).values).sum())
R['pid_order_pred_eq_ev'] = bool((pr.pid.values == ev.pid.values).all())
# ---------------- (a) 동결 예측 = 학습 직후 구성요소 npy로 다시 합성
FS = json.load(open(os.path.join(C.R2, "final_spec_r2.json"), encoding="utf-8"))
ws = FS['spec']['ensemble']['weights']; how = FS['spec']['ensemble'].get('how', 'logit')
P = os.path.join(C.FUT, "pred")
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
sig = lambda z: 1 / (1 + np.exp(-z))
comp = {}
comp['cls'] = sig(np.mean([np.load(os.path.join(P, f"cls_99f270b3cd_b{k}.npy")) for k in range(3)], 0))
zh = np.mean([np.load(os.path.join(P, f"haz_0568d4297b_b{k}.npy")) for k in range(3)], 0)
hrow = pd.read_parquet(os.path.join(C.FUT, "ev_haz.parquet"), columns=["row", "k", "pid", "lead"])
h = sig(zh)
lsum = pd.Series(np.log1p(-np.clip(h, 1e-6, 1 - 1e-6))).groupby(hrow.row.values).sum()
ph = np.zeros(len(ev)); ph[lsum.index.values] = 1 - np.exp(lsum.values)
comp['haz'] = ph
comp['logit'] = np.load(os.path.join(P, "logit_3008b8b195.npy"))
comp['hgb'] = np.load(os.path.join(P, "hgb_b4bd0fa46a.npy"))
base = np.load(os.path.join(P, "base_46526c56d8.npy"))
z = sum(w * lg(comp[k]) for k, w in ws.items() if w != 0) / sum(w for w in ws.values() if w != 0)
pens = sig(z)
R['ensemble_weights'] = ws; R['ensemble_how'] = how
R['recompose_maxdiff'] = dict(ens=float(np.abs(pens - pr.p.values).max()), base=float(np.abs(base - pr.p_base.values).max()),
                              **{k: float(np.abs(comp[k] - pr['p_' + k].values).max()) for k in comp})
R['recompose_corr_ens'] = float(np.corrcoef(pens, pr.p.values)[0, 1])
R['pred_mean'] = dict(p=float(pr.p.mean()), p_base=float(pr.p_base.mean()))
# 위험모형 결정행 = 창 안 납부기한(k년차 기한 = 등록일 + (k-1)년) — 독립 재계산 대조
rdv = pt.rd.values[hrow.pid.values]
dd = C.plus_years(rdv, hrow.k.values.astype(int) - 1)
lead_ind = (dd - np.datetime64(L)).astype('timedelta64[D]').astype(float) / 365.25
R['haz_rows'] = dict(n=len(hrow), lead_maxdiff=float(np.nanmax(np.abs(lead_ind - hrow.lead.values))), d_min=str(dd.min()), d_max=str(dd.max()),
                     row_pid_consistent=bool((ev.pid.values[hrow.row.values] == hrow.pid.values).all()))
# ---------------- (c) L 피처가 L 이전 정보만 쓰는가: 핵심 피처 독립 재계산 (L 필터) + 필터 없는 '누수판'과 비교
pids = ev.pid.values
fx = pd.DataFrame({'pid': pids})
fx['paid_year'] = lvL[pids]
fx['paid_ahead'] = (pthL[pids] - np.datetime64(L)).astype('timedelta64[D]').astype(float) / 365.25
p_le = pay[(pay.pd <= L) & (pay.seq > 0)].copy()
due = C.plus_years(pt.rd.values[p_le.pid.values], p_le.s.values.astype(int) - 1)
p_le['late'] = ((due - p_le.pd.values).astype('timedelta64[D]').astype(float) < 0).astype(int)
p_le = p_le.sort_values(['pid', 'pd', 'seq'])
g = p_le.groupby('pid')
fx['n_renew'] = fx.pid.map(g.size()).fillna(0).values
fx['n_late'] = fx.pid.map(g.late.sum()).fillna(0).values
lastpd = g.pd.max()
fx['yrs_since_pay'] = (np.datetime64(L) - fx.pid.map(lastpd).values.astype('datetime64[D]')).astype('timedelta64[D]').astype(float) / 365.25
lap = C.lapse_events(D)
fx['n_prev_lapse'] = fx.pid.map(lap[lap.cd <= L].groupby('pid').size()).fillna(0).values
rec = rt[rt.name == "회복등록"]
fx['n_recov'] = fx.pid.map(rec[rec.cd <= L].groupby('pid').size()).fillna(0).values
cmp = {}
for c in ['paid_year', 'paid_ahead', 'n_renew', 'n_late', 'yrs_since_pay', 'n_prev_lapse', 'n_recov']:
    a = fx[c].values.astype(float); b = ev[c].values.astype(float)
    both = np.isfinite(a) & np.isfinite(b)
    cmp[c] = dict(maxabs=float(np.abs(a[both] - b[both]).max()) if both.any() else None, n_nan_mismatch=int((np.isfinite(a) != np.isfinite(b)).sum()),
                  share_exact=float((np.abs(a[both] - b[both]) < 1e-3).mean()))
R['feature_recompute_le_L'] = cmp
# [재개 시 수정] 필터 없는 2026-08 전체 자료와 비교하는 '누수판' 블록은 L 이후 납부·소멸 건수(=결과 근사)를 드러내므로 삭제(결과 미열람 원칙).
# ksic_lapse_1y 독립 재계산: (L-1y, L] 소멸등록 / (소멸등록 + 납부) — t4lib 정의
L1 = C.plus_years(np.array([np.datetime64(L)]), -1)[0]
lp = lap[(lap.cd <= L) & (lap.cd > L1)]
pa = pay[(pay.pd <= L) & (pay.pd > L1) & (pay.seq > 0)]
kl = pd.Series(pt.ksic.values[lp.pid.values]).value_counts(); kr = pd.Series(pt.ksic.values[pa.pid.values]).value_counts()
kk = (kl / (kl + kr)).fillna(0)
a = pd.Series(ev.ksic.values).map(kk).values.astype(float); b = ev.ksic_lapse_1y.values.astype(float)
R['feature_recompute_le_L']['ksic_lapse_1y'] = dict(maxabs=float(np.nanmax(np.abs(a - b))))
# ---------------- (c) 학습 레이블 컷 C=2025-08-22 확인
Cc = np.datetime64('2025-08-22')
fc = pd.read_parquet(os.path.join(C.FUT, "frame_cls.parquet"), columns=["pid", "L", "y"])
fh = pd.read_parquet(os.path.join(C.FUT, "frame_haz.parquet"), columns=["pid", "L", "y", "d", "k"])
fb = pd.read_parquet(os.path.join(C.FUT, "frame_base.parquet"), columns=["pid", "L", "y"])
R['train_frames'] = dict(cls_rows=len(fc), cls_L_max=str(fc.L.max()), cls_L_min=str(fc.L.min()), cls_pos=float(fc.y.mean()),
                         base_rows=len(fb), base_L_max=str(fb.L.max()),
                         haz_rows=len(fh), haz_d_max=str(pd.to_datetime(fh.d).max().date()), haz_L_max=str(fh.L.max()),
                         haz_rule_d_le=str(Cc - np.timedelta64(450, 'D')))
# 분류형 레이블 독립 재계산 (label_asof 정의, 컷 C) + 컷 없는 버전과 비교
def label_asof(Lstr, pids_, Ccut):
    Ld = np.datetime64(Lstr); end = C.plus_years(np.array([Ld]), 3)[0]
    if Ccut is None:
        e = lap[(lap.E > Ld) & (lap.E <= end)].pid.unique(); Cq = np.datetime64('2026-08-15')
    else:
        e = lap[(lap.E > Ld) & (lap.E <= end) & (lap.cd <= Ccut)].pid.unique(); Cq = Ccut
    y = np.full(len(pids_), np.nan)
    pthC = PTH[str(Cq)]
    exv = pt.exp.values[pids_]
    zero = (pthC[pids_] > end) | ((exv <= end) & (exv <= Cq) & (pthC[pids_] >= exv))
    ml = rt[rt.name.isin(C.MALSO)]
    mz = ml[(ml.cd > Ld) & (ml.cd <= min(end, Cq))].pid.unique()
    zero = zero | np.isin(pids_, mz)
    y[zero] = 0; y[np.isin(pids_, e)] = 1
    return y
pthAll = C.paid_thru(pt, pay, np.datetime64('2026-08-15'))[0]  # L<=2021-07 창(끝<=2024-07) 레이블 대조에만 사용
PTH = {str(Cc): C.paid_thru(pt, pay, Cc)[0], '2026-08-15': pthAll}
mism_c, mism_full, n_chk, lab_only_after_C = 0, 0, 0, 0
for Lstr, gg in fc.groupby('L'):
    yc = label_asof(Lstr, gg.pid.values, Cc)
    mism_c += int((yc != gg.y.values).sum()); n_chk += len(gg)
    if Lstr >= '2020-07-01':
        yf = label_asof(Lstr, gg.pid.values, None)
        mism_full += int((yf != gg.y.values).sum())
R['cls_label_check'] = dict(n=n_chk, mismatch_vs_cut_C=mism_c, mismatch_vs_nocut_L_ge_2020_07=mism_full)
# 위험모형 y=1 결정: 같은 (pid, E=d) 소멸등록이 cd <= C 인가
h1 = fh[fh.y == 1]
key = pd.DataFrame({'pid': lap.pid.values, 'Ei': lap.E.values.astype('datetime64[D]').astype('int64'), 'cd': lap.cd.values})
mh = pd.DataFrame({'pid': h1.pid.values, 'd': pd.to_datetime(h1.d).values.astype('datetime64[D]').astype('int64')}).merge(key, left_on=['pid', 'd'], right_on=['pid', 'Ei'], how='left')
R['haz_pos_check'] = dict(n_pos=len(h1), matched=int(mh.cd.notna().groupby([mh.pid, mh.d]).any().sum()),
                          cd_max=str(mh.cd.max()), any_cd_gt_C=int((mh.groupby(['pid', 'd']).cd.min() > Cc).sum()))
# ---------------- (d) 소멸등록 시차(E -> cd), E <= 2024-12-31 코호트만 (공식 시험 구간 이전·내부, 새 창 아님)
lg_ = lap[(lap.E >= np.datetime64('2021-01-01')) & (lap.E <= np.datetime64('2024-12-31'))].copy()
lg_['lag'] = (lg_.cd - lg_.E).dt.days.astype(float)
lg_['h'] = pd.DatetimeIndex(lg_.E).year.astype(str) + 'H' + np.where(pd.DatetimeIndex(lg_.E).month <= 6, '1', '2')
qs = [0.5, 0.9, 0.97, 0.99, 0.995, 0.999]
R['lag_by_half'] = {h_: dict(n=len(v), **{str(q): float(v.lag.quantile(q)) for q in qs}, share_gt_287=float((v.lag > 287).mean()),
                             share_gt_300=float((v.lag > 300).mean()), share_neg=float((v.lag < 0).mean())) for h_, v in lg_.groupby('h')}
# 창 끝 결정: 스냅샷(right 최대일) 2026-08-15 - 287일(P97, 2016-10 이후 제도) / P99.9
snap = np.datetime64('2026-08-15')
R['window_candidates'] = {f'lag{d}': str(snap - np.timedelta64(d, 'D')) for d in (287, 290, 300, 332)}
# 예상 사건 수(결과 미사용): 위험집합 × 개발 기저율(공식 3년 18.77%, 단기 환산)
R['expected_events'] = {}
for wend in ('2025-10-31', '2025-09-30'):
    days = (np.datetime64(wend) - np.datetime64(L)).astype(int)
    r3 = 0.1877
    rw = 1 - (1 - r3) ** (days / 1096)
    due = (pthL[pids] > np.datetime64(L)) & (pthL[pids] <= np.datetime64(wend))
    R['expected_events'][wend] = dict(days=int(days), risk_set=len(pids), rate_scaled=rw, exp_events_scaled=rw * len(pids),
                                      n_due_in_window=int(due.sum()),
                                      exp_events_due_x_frame_haz_pos=float(due.sum() * float(fh.y.mean())))
import pyarrow.parquet as pq
TC = os.path.join(C.TR, 'cache')
def _mx(f, col):
    t = pq.read_table(os.path.join(TC, f), columns=[col]).column(0).to_pandas().astype(str)
    t = t[t.str.match(r'^[12][0-9]{7}$')]
    return dict(n=int(len(t)), max=str(t.max()), n_gt_20250822=int((t > '20250822').sum()))
R['t4_cache_dates'] = dict(pay_payDate=_mx('pay.parquet', 'payDate'), right_rgstCsDate=_mx('right.parquet', 'rgstCsDate'),
                           owner_od=_mx('owner.parquet', 'ownerRgstCsDate'), patent_rgstDate=_mx('patent.parquet', 'rgstDate'))
R['secs'] = round(time.time() - t0, 1); R['peak_gb'] = C.peak_gb()
C.jdump(R, os.path.join(C.CACHE, "s02_p0.json"))
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
