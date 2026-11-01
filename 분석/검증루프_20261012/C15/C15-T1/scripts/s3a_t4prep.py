# -*- coding: utf-8 -*-
"""C15-T1 s3a: T4 공식 최종시험 — (e) 실용신안 혼입 수, 독립 레이블(label_np 재현·S3·S3b), KSIC 내부 포착 r_int_k, S1–S4 점 추정, 4.2 포기율 재계산.
t4lib/r2lib import 없음. 입력은 읽기만: registry_full.sqlite(mode=ro), T4 cache/patent.parquet, R2 runs_final/F22(ev.parquet, frame_*.parquet, pred/*.npy), final_spec_r2.json, _shared/ext/ntis_patent_flag.parquet
출력: cache/s3a_out.json, cache/s3a_arrays.npz, cache/s3a_ksic.csv, cache/s3a_cohort42.csv"""
import os, sys, json, time, sqlite3
import numpy as np, pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
t0 = time.time()
sys.stdout.reconfigure(encoding="utf-8")
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
DB = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
MC = os.path.join(BASE, "분석", "모델경쟁_20260927")
TR = os.path.join(MC, "T4_특허포기예측"); FD = os.path.join(TR, r"R2\runs_final\F22")
OUT = {}
L = pd.Timestamp("2022-01-01"); END = pd.Timestamp("2025-01-01"); DE = pd.Timestamp("2025-08-22"); GR = pd.Timedelta(days=213)


def pdt(s):
    return pd.to_datetime(pd.Series(s, dtype="object").astype(str).str.strip().str[:8], format="%Y%m%d", errors="coerce")


def addy(d, k):
    d = pd.DatetimeIndex(d); k = np.asarray(k, float)
    y = d.year.values + k; mo = d.month.values; dd = np.where((mo == 2) & (d.day.values == 29), 28, d.day.values)
    ok = np.isfinite(y) & ~pd.isna(d)
    out = pd.Series(pd.NaT, index=range(len(d)), dtype="datetime64[ns]")
    if ok.any():
        out[ok] = pd.to_datetime(pd.DataFrame({"year": y[ok].astype(int), "month": mo[ok], "day": dd[ok]}), errors="coerce").values
    return out.values


# ---------------- (e) pid → 등록번호 (t4lib 규칙 재현: T4 cache patent.parquet, rd 있음, '10'만, 순서 유지)
tp = pd.read_parquet(os.path.join(TR, "cache", "patent.parquet"), columns=["rgst_no", "ksic", "rgstDate", "applDate", "cndrtExptnDate"])
OUT["t4cache_patent_rows"] = int(len(tp)); OUT["t4cache_prefix"] = tp.rgst_no.str[:2].value_counts().to_dict()
tp["rd"] = pdt(tp.rgstDate)
keep = tp.rd.notna() & (tp.rgst_no.str[:2] == "10")
M = tp[keep].reset_index(drop=True); M["pid"] = np.arange(len(M))
OUT["t4_pid_universe"] = int(len(M))
ev = pd.read_parquet(os.path.join(FD, "ev.parquet"), columns=["pid", "acd", "atype", "ksic", "y", "L"])
OUT["ev_rows"] = int(len(ev)); OUT["ev_pid_max"] = int(ev.pid.max())
evr = M.rgst_no.values[ev.pid.values]
OUT["ev_ksic_match_via_mapping"] = float((M.ksic.values[ev.pid.values].astype(str) == ev.ksic.astype(str).values).mean())
OUT["e_final_pop_non10"] = int((pd.Series(evr).str[:2] != "10").sum())
OUT["e_final_pos_non10"] = int(((pd.Series(evr).str[:2] != "10").values & (ev.y.values == 1)).sum())
for fn in ["frame_cls.parquet", "frame_haz.parquet", "ev_haz.parquet"]:
    pids = pq.read_table(os.path.join(FD, fn), columns=["pid"]).column(0).to_numpy()
    r = M.rgst_no.values[pids]
    OUT[f"e_{fn}"] = dict(rows=int(len(pids)), pid_max=int(pids.max()), non10=int((pd.Series(r).str[:2] != "10").sum()))
# 원부 sqlite 순서 대조(C04 방식: rowid 순) — 같은 pid 체계인지
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
pt = pd.read_sql("select rowid as rid, rgst_no, ksic, applDate, rgstDate, cndrtExptnDate from patent order by rowid", con)
pt["rd"] = pdt(pt.rgstDate); pt["ad"] = pdt(pt.applDate); pt["ex"] = pdt(pt.cndrtExptnDate)
pt = pt[pt.rd.notna() & pt.rgst_no.str.startswith("10")].reset_index(drop=True)
OUT["sqlite_vs_t4cache_order_equal"] = bool((pt.rgst_no.values == M.rgst_no.values).all()) if len(pt) == len(M) else False
m = pt.ex.isna(); pt.loc[m, "ex"] = addy(pt.loc[m, "ad"].values, np.full(m.sum(), 20))
idx = pd.Series(np.arange(len(pt)), index=pt.rgst_no.values)
pay = pd.read_sql("select rgst_no, lastAnnl, payDate from pay where substr(rgst_no,1,2)='10'", con)
pay["pid"] = pay.rgst_no.map(idx); pay["l"] = pd.to_numeric(pay.lastAnnl, errors="coerce"); pay["pd"] = pdt(pay.payDate)
pay = pay[pay.pid.notna() & pay.l.notna() & pay.pd.notna()].copy(); pay["pid"] = pay.pid.astype(int)
rt = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName, rgstCsReason from right where substr(rgst_no,1,2)='10'", con)
rt["pid"] = rt.rgst_no.map(idx); rt = rt[rt.pid.notna()].copy(); rt["pid"] = rt.pid.astype(int); rt["cd"] = pdt(rt.rgstCsDate)
PAYEND = pay.pd.max(); OUT["pay_max"] = str(PAYEND.date())
rd = pt.rd.values; ex = pt.ex.values


def paid_thru(asof):
    p = pay[pay.pd <= asof]; lm = p.groupby("pid").l.max()
    lv = np.zeros(len(pt)); lv[lm.index.values] = lm.values
    out = np.full(len(pt), np.datetime64("NaT"), dtype="datetime64[ns]"); ok = lv > 0
    out[ok] = addy(rd[ok], lv[ok].astype(int))
    return out


malso = rt[rt.rgstCsName.isin(["권리의 말소등록", "말소등록", "직권말소등록"])]
pthL = paid_thru(L)
risk = (rd < np.datetime64(L)) & (pthL > np.datetime64(L)) & (ex > np.datetime64(L))
risk[malso[malso.cd <= L].pid.unique()] = False
rp = np.where(risk)[0]
mz = np.isin(rp, malso[(malso.cd > L) & (malso.cd <= END)].pid.unique())


def label_np(de):
    pthE = paid_thru(de)[rp]; exr = ex[rp]
    zero = (pthE > np.datetime64(END)) | ((exr <= np.datetime64(END)) & (pthE >= exr)) | mz
    one = (pthE > np.datetime64(L)) & (pthE <= np.datetime64(END)) & (pthE < exr) & (pthE + np.timedelta64(213, "D") <= np.datetime64(de)) & ~mz
    y = np.full(len(rp), np.nan); y[zero] = 0; y[one & ~zero] = 1
    return y


y0 = label_np(DE); y3 = label_np(PAYEND)
lap = rt[(rt.rgstCsName == "소멸등록") & rt.rgstCsReason.isin(["등록료불납", "연차료불납"]) & rt.cd.notna()][["pid", "cd"]].reset_index(drop=True)
mm = lap.reset_index().merge(pay[["pid", "pd", "l"]], on="pid"); mm = mm[mm.pd < mm.cd]
lap["lmax"] = lap.index.map(mm.groupby("index").l.max()); lap = lap[lap.lmax.notna()].copy()
lap["E"] = addy(rd[lap.pid.values], lap.lmax.values.astype(int))
lapw = lap[(lap.E > L) & (lap.E <= END)]
y3b = np.isin(rp, lapw.pid.unique()).astype(float)
y3b_asof = np.isin(rp, lapw[lapw.cd <= DE].pid.unique()).astype(float)
lab = pd.DataFrame({"pid": rp, "y0": y0, "y3": y3, "y3b": y3b, "y3b_asof": y3b_asof})
e2 = ev.merge(lab, on="pid", how="left")
OUT["indep_pop"] = int(len(rp)); OUT["indep_undetermined_y0"] = int(np.isnan(y0).sum()); OUT["indep_undetermined_y3"] = int(np.isnan(y3).sum())
OUT["label_reproduced"] = bool((e2.y0.values == e2.y.values).all()) and len(rp) == len(ev)
OUT["pos_official"] = int(ev.y.sum()); OUT["pos_y3"] = int(np.nansum(e2.y3)); OUT["pos_y3b"] = int(e2.y3b.sum()); OUT["pos_y3b_asof"] = int(e2.y3b_asof.sum())
OUT["S3_flips"] = dict(np1_to0=int(((e2.y == 1) & (e2.y3 == 0)).sum()), np0_to1=int(((e2.y == 0) & (e2.y3 == 1)).sum()))
OUT["S3b_vs_official"] = dict(off1_lap0=int(((e2.y == 1) & (e2.y3b == 0)).sum()), off0_lap1=int(((e2.y == 0) & (e2.y3b == 1)).sum()),
                              lap_regs_after_DE_in_window=int(lapw[lapw.cd > DE].pid.nunique()))
# ---------------- 예측 합성(공식 spec)
FS = json.load(open(os.path.join(TR, r"R2\final_spec_r2.json"), encoding="utf-8"))
ws = FS["spec"]["ensemble"]["weights"]
files = {"cls": "cls_99f270b3cd.npy", "haz": "haz_0568d4297b.npy", "logit": "logit_3008b8b195.npy", "hgb": "hgb_b4bd0fa46a.npy", "base": "base_46526c56d8.npy"}
Pp = {k: np.load(os.path.join(FD, "pred", f)) for k, f in files.items()}
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
z = sum(w * lg(Pp[k]) for k, w in ws.items() if w != 0) / sum(w for w in ws.values() if w != 0)
pm = 1 / (1 + np.exp(-z)); pb = Pp["base"]
y = ev.y.values.astype(int)
sk = lambda a, b: (a - b) / (1 - b)
A = lambda yy, pp, w=None: float(roc_auc_score(yy, pp, sample_weight=w))
OUT["S0"] = dict(auc=A(y, pm), auc_base=A(y, pb), skill=sk(A(y, pm), A(y, pb)))
# ---------------- KSIC 내부 포착 r_int_k = 소멸 사건 기준 양성 / 공식 양성
K = e2.groupby("ksic").agg(n=("y", "size"), pos_np=("y", "sum"), pos_lap=("y3b", "sum"), pos_y3=("y3", "sum"))
K["r_int"] = K.pos_lap / K.pos_np
K = K.sort_values("r_int")
K.to_csv(os.path.join(W, "cache", "s3a_ksic.csv"), encoding="utf-8-sig")
out_k = K[(K.r_int < 0.97) | (K.r_int > 1.03)].index.tolist()
rule = "r_int 밖 [0.97,1.03]"
if not out_k:
    out_k = (K.r_int - 1).abs().sort_values(ascending=False).index[:3].tolist(); rule = "대체 규칙: |r_int-1| 상위 3"
OUT["S2_rule"] = rule; OUT["S2_excluded_ksic"] = out_k
OUT["r_int_overall"] = float(K.pos_lap.sum() / K.pos_np.sum()); OUT["r_int_range"] = [float(K.r_int.min()), float(K.r_int.max())]
keep2 = ~ev.ksic.isin(out_k).values
OUT["S2"] = dict(n=int(keep2.sum()), pos=int(y[keep2].sum()), auc=A(y[keep2], pm[keep2]), auc_base=A(y[keep2], pb[keep2]))
OUT["S2"]["skill"] = sk(OUT["S2"]["auc"], OUT["S2"]["auc_base"])
y3i = e2.y3.values; ok3 = ~np.isnan(y3i)
OUT["S3"] = dict(n=int(ok3.sum()), pos=int(np.nansum(y3i)), auc=A(y3i[ok3].astype(int), pm[ok3]), auc_base=A(y3i[ok3].astype(int), pb[ok3]))
OUT["S3"]["skill"] = sk(OUT["S3"]["auc"], OUT["S3"]["auc_base"])
y3bi = e2.y3b.values.astype(int)
OUT["S3b"] = dict(pos=int(y3bi.sum()), auc=A(y3bi, pm), auc_base=A(y3bi, pb)); OUT["S3b"]["skill"] = sk(OUT["S3b"]["auc"], OUT["S3b"]["auc_base"])
# S4 국적 2분 가중 (공식 2021 존속 특허: 내국 882,651 / 외국 270,669 — 4.1.11)
fr = ev.atype.isin(["fcorp", "findiv"]).values
off_for = 270669 / (882651 + 270669); prov_for = fr.mean()
w4 = np.where(fr, off_for / prov_for, (1 - off_for) / (1 - prov_for))
OUT["S4"] = dict(prov_for=float(prov_for), off_for=float(off_for), w_for=float(off_for / prov_for), w_dom=float((1 - off_for) / (1 - prov_for)),
                 auc=A(y, pm, w4), auc_base=A(y, pb, w4))
OUT["S4"]["skill"] = sk(OUT["S4"]["auc"], OUT["S4"]["auc_base"])
s2o = json.load(open(os.path.join(W, "cache", "s2_out.json"), encoding="utf-8"))
w4b = w4 / np.where(fr, s2o["r_F_for"], s2o["r_F_dom"]); w4b = w4b / w4b.mean()
OUT["S4b"] = dict(auc=A(y, pm, w4b), auc_base=A(y, pb, w4b)); OUT["S4b"]["skill"] = sk(OUT["S4b"]["auc"], OUT["S4b"]["auc_base"])
# 유형별 양성률(보고)
OUT["pos_rate_by_nat"] = dict(dom=float(y[~fr].mean()), foreign=float(y[fr].mean()))
np.savez_compressed(os.path.join(W, "cache", "s3a_arrays.npz"), pm=pm, pb=pb, y=y, y3=np.nan_to_num(y3i, nan=-1).astype(int), y3b=y3bi, keep2=keep2, w4=w4, w4b=w4b,
                    acd=ev.acd.values.astype(str), pid=ev.pid.values)
print("T4 part", round(time.time() - t0), json.dumps({k: OUT[k] for k in ["S0", "S2", "S3", "S3b", "S4", "S4b"]}, ensure_ascii=False), flush=True)

# ---------------- 4.2 포기율(표 3) 재계산 — build_ntis_tables.py 논리 재현(import 없음)
pa = pd.read_sql("select rgst_no, rgstDate, ksic from patent", con)
lapq = pd.read_sql("select rgst_no, min(rgstCsDate) d from right where rgstCsName='소멸등록' and rgstCsReason like '%불납%' group by rgst_no", con)
recq = pd.read_sql("select rgst_no, max(rgstCsDate) d from right where rgstCsName like '회복등록%' group by rgst_no", con)
lastnp = pd.read_sql("select rgst_no, max(rgstCsDate) d from right where rgstCsName='소멸등록' and rgstCsReason like '%불납%' group by rgst_no", con)
con.close()
pa["rd"] = pdt(pa.rgstDate); pa = pa[pa.rd.notna() & (pa.rd.dt.year >= 2007) & (pa.rd.dt.year <= 2016)].drop_duplicates("rgst_no")
pa["lap"] = pa.rgst_no.map(lapq.set_index("rgst_no").d).pipe(pdt)
pa["rec"] = pa.rgst_no.map(recq.set_index("rgst_no").d).pipe(pdt)
pa["lastnp"] = pa.rgst_no.map(lastnp.set_index("rgst_no").d).pipe(pdt)
pa["rd5"] = pa.rd + pd.DateOffset(years=5)
pa["lap5"] = (pa.lap.notna() & (pa.lap <= pa.rd5)).astype(float)
# 실효일 E 기준: lap 표에서 E(첫 불납 소멸) — 위 lap(E) 표는 '10'만이므로 실용은 NaN
firstE = lap.sort_values("E").drop_duplicates("pid")
Emap = pd.Series(firstE.E.values, index=pt.rgst_no.values[firstE.pid.values])
pa["E"] = pa.rgst_no.map(Emap)
pa["lapE5"] = (pa.E.notna() & (pa.E <= pa.rd5)).astype(float)
# 회복된 소멸 제외(마지막 불납 소멸 뒤 같은 날·이후 회복이 있으면 소멸 아님)
pa["lap5_norec"] = (pa.lap5 == 1) & ~(pa.rec.notna() & (pa.rec >= pa.lastnp))
pa["lap5_norec"] = pa.lap5_norec.astype(float)
fl = pd.read_parquet(os.path.join(MC, "_shared", "ext", "ntis_patent_flag.parquet"), columns=["rgst_no", "gov_rnd"]).dropna(subset=["rgst_no"])
fl["rgst_no"] = fl.rgst_no.astype(str); fl = fl.groupby("rgst_no").gov_rnd.max()
pa["gov"] = pa.rgst_no.map(fl).fillna(False).astype(bool)
pa["pre"] = pa.rgst_no.str[:2]
rows = []
for var, sub in [("S1_특허만(원 코드와 같음)", pa[pa.pre == "10"]), ("반사실_실용 포함", pa)]:
    for col, lab_ in [("lap5", "소멸등록일 ≤ 등록+5년(원 정의)"), ("lapE5", "실효일 E ≤ 등록+5년"), ("lap5_norec", "원 정의 − 회복 제외")]:
        g1 = sub[sub.gov][col].mean(); g0 = sub[~sub.gov][col].mean()
        rows.append(dict(variant=var, definition=lab_, n=len(sub), n_gov=int(sub.gov.sum()), rate_gov=g1, rate_nongov=g0, gap=g0 - g1))
C42 = pd.DataFrame(rows)
C42.to_csv(os.path.join(W, "cache", "s3a_cohort42.csv"), index=False, encoding="utf-8-sig")
OUT["cohort42"] = C42.round(4).to_dict("records")
OUT["cohort42_utility_rows_in_window"] = int((pa.pre == "20").sum())
OUT["secs"] = round(time.time() - t0, 1)
json.dump(OUT, open(os.path.join(W, "cache", "s3a_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in OUT.items() if k not in ("S0", "S2", "S3", "S3b", "S4", "S4b")}, ensure_ascii=False, indent=1, default=str))
