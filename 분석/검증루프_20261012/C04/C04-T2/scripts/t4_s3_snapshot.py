# -*- coding: utf-8 -*-
"""C04-T2 T4 시점 정합 점검(정보): 등록원부 납부·권리 테이블은 2026-08까지 들어 있다(특허 목록은 2025-08-22까지).
공식 레이블(납부 asof 2025-08-22)을 후속 스냅샷(납부 asof 2026-08-14)으로 다시 판정하면 몇 건이 바뀌고 성적이 얼마나 움직이는지.
공식 성적은 바꾸지 않는다. 출력 cache/t4_s3_out.json"""
import os, json, sqlite3, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import importlib.util
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
TR = os.path.join(BASE, r"분석\모델경쟁_20260927\T4_특허포기예측"); FD = os.path.join(TR, r"R2\runs_final\F22")
con = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'registry_full.sqlite')}?mode=ro", uri=True)
lab = pd.read_parquet(os.path.join(W, "cache", "t4_labels_indep.parquet"))
pt = pd.read_sql("select rgst_no, rgstDate, cndrtExptnDate, applDate from patent", con).set_index("rgst_no").loc[lab.rgst_no]
pay = pd.read_sql("select rgst_no, lastAnnl, payDate from pay", con)
rt = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName from right where rgstCsName in ('권리의 말소등록','말소등록','직권말소등록')", con); con.close()
pdt = lambda s: pd.to_datetime(pd.Series(s).astype(str), format="%Y%m%d", errors="coerce").values.astype("datetime64[D]")
rd = pdt(pt.rgstDate.values); ex = pdt(pt.cndrtExptnDate.values)
pay = pay[pay.rgst_no.isin(set(lab.rgst_no))].copy(); pay["l"] = pd.to_numeric(pay.lastAnnl, errors="coerce"); pay["pd"] = pdt(pay.payDate.values)
pay = pay[pay.l.notna() & ~pd.isna(pay.pd)]
L = np.datetime64("2022-01-01"); END = np.datetime64("2025-01-01")


def addy(d, k):
    di = pd.DatetimeIndex(d); y = di.year.values + k; m = di.month.values; dd = np.where((m == 2) & (di.day.values == 29), 28, di.day.values)
    return pd.to_datetime(pd.DataFrame({"year": y, "month": m, "day": dd})).values.astype("datetime64[D]")


rt["cd"] = pdt(rt.rgstCsDate.values)
mz = lab.rgst_no.isin(set(rt[(rt.cd > L) & (rt.cd <= END)].rgst_no)).values
out = {}
for tag, de in (("official_2025-08-22", "2025-08-22"), ("later_2026-08-14", "2026-08-14")):
    de = np.datetime64(de)
    lm = pay[pay.pd <= de].groupby("rgst_no").l.max().reindex(lab.rgst_no).fillna(0).values.astype(int)
    p = np.where(lm > 0, addy(rd, lm), np.datetime64("NaT"))
    zero = (p > END) | ((ex <= END) & (p >= ex)) | mz
    one = (p > L) & (p <= END) & (p < ex) & (p + np.timedelta64(213, "D") <= de) & ~mz
    y = np.full(len(lab), np.nan); y[zero] = 0; y[one & ~zero] = 1
    out[tag] = y
y0, y1 = out["official_2025-08-22"], out["later_2026-08-14"]
assert (y0 == lab.y_indep.values).all()
ev = pd.read_parquet(os.path.join(FD, "ev.parquet"), columns=["pid"])
ws = json.load(open(os.path.join(TR, r"R2\final_spec_r2.json"), encoding="utf-8"))["spec"]["ensemble"]["weights"]
P = {k: np.load(os.path.join(FD, "pred", f)) for k, f in [("cls", "cls_99f270b3cd.npy"), ("haz", "haz_0568d4297b.npy"), ("logit", "logit_3008b8b195.npy"), ("hgb", "hgb_b4bd0fa46a.npy"), ("base", "base_46526c56d8.npy")]}
lg = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
pm = 1 / (1 + np.exp(-sum(w * lg(P[k]) for k, w in ws.items()) / sum(ws.values())))
ps = np.load(os.path.join(TR, r"VERIFY_R5\cache\strong_full.npy"))
order = pd.Series(np.arange(len(lab)), index=lab.pid.values).reindex(ev.pid).values
yl = y1[order]; ok = ~np.isnan(yl)
A = lambda yy, q: float(roc_auc_score(yy, q))
am, ab, as_ = A(yl[ok], pm[ok]), A(yl[ok], P["base"][ok]), A(yl[ok], ps[ok])
R = dict(n=int(len(y0)), changed_1_to_0=int(((y0 == 1) & (y1 == 0)).sum()), changed_0_to_1=int(((y0 == 0) & (y1 == 1)).sum()), pos_later=int(np.nansum(y1)),
         auc_model=am, auc_base=ab, skill=(am - ab) / (1 - ab), skill_vs_strong=(am - as_) / (1 - as_), secs=round(time.time() - t0, 1),
         note="1→0은 2025-08-22 이후 납부(회복 등)로 살아난 특허. 공식 레이블 규칙은 그대로, 납부 기준일만 늦춤")
json.dump(R, open(os.path.join(W, "cache", "t4_s3_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(R)
