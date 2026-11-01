# -*- coding: utf-8 -*-
"""R5 레드팀(T4) 1단계: 최종시험 위험집합·레이블을 원본 registry_full.sqlite(읽기전용)에서 캐시 없이 직접 다시 만든다.
t4lib/r2lib를 import하지 않는다(정의만 같게 독립 구현). 결과를 동결 ev.parquet와 대조한다.
위험집합(L=2022-01-01): 등록일 < L, L까지의 납부로 본 납부완료일 > L, 존속만료 > L, L까지 말소 없음
레이블(label_np): 2025-08-22까지의 납부로 본 납부완료일 pth ∈ (L, L+3y], pth < 존속만료, pth+213일 ≤ 2025-08-22, 창 안 말소 없음 → 1
                  pth > L+3y, 또는 존속만료까지 납부, 또는 창 안 말소 → 0"""
import os, sys, json, sqlite3, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE)
DB = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
L = np.datetime64("2022-01-01"); END = np.datetime64("2025-01-01"); DE = np.datetime64("2025-08-22"); G = np.timedelta64(213, "D")
t = time.time()
c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
pt = pd.read_sql("select rgst_no, ksic, applDate, claimCount, cndrtExptnDate, rgstDate from patent", c)
pay = pd.read_sql("select rgst_no, lastAnnl, payDate from pay", c)
ml = pd.read_sql("select rgst_no, rgstCsDate from right where rgstCsName in ('권리의 말소등록','말소등록','직권말소등록')", c)
c.close()
D = lambda s: pd.to_datetime(pd.Series(s, dtype="object").astype(str).str.strip(), format="%Y%m%d", errors="coerce")


def addy(d, k):
    d = pd.DatetimeIndex(d); k = np.asarray(k)
    day = np.where((d.month == 2) & (d.day == 29), 28, d.day)
    ok = ~pd.isna(d) & ~np.isnan(np.asarray(k, float))
    out = pd.Series(pd.NaT, index=range(len(d)), dtype="datetime64[ns]")
    out[ok] = pd.to_datetime(pd.DataFrame({"year": (d.year.values[ok] + k[ok]).astype(int), "month": d.month.values[ok], "day": day[ok]}), errors="coerce").values
    return out.values


pt["rd"] = D(pt.rgstDate); pt["ad"] = D(pt.applDate); pt["exp"] = D(pt.cndrtExptnDate)
pt = pt[pt.rd.notna() & (pt.rgst_no.str[:2] == "10")].reset_index(drop=True)
pt["pid"] = np.arange(len(pt))
m = pt.exp.isna(); pt.loc[m, "exp"] = addy(pt.loc[m, "ad"], np.full(m.sum(), 20))
pid = pd.Series(pt.pid.values, index=pt.rgst_no.values)
pay["pid"] = pay.rgst_no.map(pid); pay["l"] = pd.to_numeric(pay.lastAnnl, errors="coerce"); pay["pd"] = D(pay.payDate)
pay = pay[pay.pid.notna() & pay.l.notna() & pay.pd.notna()]
ml["pid"] = ml.rgst_no.map(pid); ml["cd"] = D(ml.rgstCsDate); ml = ml[ml.pid.notna() & ml.cd.notna()]


def pth(asof):
    lm = pay[pay.pd <= asof].groupby("pid").l.max()
    l = np.zeros(len(pt)); l[lm.index.values.astype(int)] = lm.values
    out = np.full(len(pt), np.datetime64("NaT"), dtype="datetime64[ns]")
    ok = l > 0
    out[ok] = addy(pt.rd.values[ok], l[ok].astype(int))
    return out


pL = pth(L)
ok = (pt.rd.values < L) & (pL > L) & (pt.exp.values > L)
ok[ml[ml.cd <= L].pid.values.astype(int)] = False
rs = np.where(ok)[0]
pE = pth(DE)[rs]; ex = pt.exp.values[rs]
mz = np.isin(rs, ml[(ml.cd > L) & (ml.cd <= END)].pid.values.astype(int))
zero = (pE > END) | ((ex <= END) & (pE >= ex)) | mz
one = (pE > L) & (pE <= END) & (pE < ex) & (pE + G <= DE) & ~mz
y = np.full(len(rs), np.nan); y[zero] = 0; y[one & ~zero] = 1
mine = pd.DataFrame({"pid": rs, "y_indep": y, "ksic_indep": pt.ksic.values[rs].astype(str), "rgst_no": pt.rgst_no.values[rs]})
ev = pd.read_parquet(os.path.join(TR, "R2", "runs_final", "F22", "ev.parquet"), columns=["pid", "ksic", "y", "acd", "atype"])
mm = ev.merge(mine, on="pid", how="outer", indicator=True)
both = mm[mm._merge == "both"]
res = dict(n_frozen=len(ev), n_indep=len(mine), only_frozen=int((mm._merge == "left_only").sum()), only_indep=int((mm._merge == "right_only").sum()),
           undetermined_indep=int(np.isnan(y).sum()), pos_frozen=int(ev.y.sum()), pos_indep=int(np.nansum(y)),
           label_mismatch=int((both.y != both.y_indep).sum()), ksic_mismatch=int((both.ksic.astype(str) != both.ksic_indep).sum()),
           base_rate_indep=float(np.nanmean(y)), secs=round(time.time() - t))
print(json.dumps(res, ensure_ascii=False, indent=1))
json.dump(res, open(os.path.join(HERE, "v1_labels.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
mine.to_parquet(os.path.join(HERE, "cache", "labels_indep.parquet"), index=False)
