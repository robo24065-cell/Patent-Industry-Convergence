# verify_0 v1: C-A 주 정의(m=2, 다KSIC) 양성 수를 원천 sqlite에서 독립 재계산 (캐시 parquet 미사용)
import sqlite3, time, sys
import pandas as pd, numpy as np
t = time.time()
DB = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C01\C01-T4\verify_0"
con = sqlite3.connect("file:" + DB + "?mode=ro", uri=True)
pat = pd.read_sql("select rgst_no, ksic, rgstDate, cndrtExptnDate from patent", con)
pay = pd.read_sql("select rgst_no, cast(lastAnnl as integer) a, payDate from pay", con)
rt = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName, rgstCsReason from right where rgstCsName like '%소멸%' or rgstCsName like '회복%'", con)
own = pd.read_sql("select rgst_no, min(ownerRgstCsDate) ft from owner where seq>=1 and ownerRgstCsName is not null group by rgst_no", con)
app = pd.read_sql("select distinct rgst_no, applicantCd from applicant where applicantCd like '1%'", con)
con.close()
D = lambda s: pd.to_datetime(s, format="%Y%m%d", errors="coerce")
pat["rg"] = D(pat.rgstDate); pat["ex"] = D(pat.cndrtExptnDate)
pay["pd"] = D(pay.payDate)
rt["d"] = D(rt.rgstCsDate)
# 불납 소멸: 이름에 '소멸' + 사유에 '불납' (공백 변형 포함)
lp = rt[rt.rgstCsName.str.replace(" ", "").str.contains("소멸") & rt.rgstCsReason.fillna("").str.replace(" ", "").str.contains("불납")]
lap = lp.groupby("rgst_no").d.max().rename("lapd")
rec = rt[rt.rgstCsName.str.startswith("회복")].groupby("rgst_no").d.max().rename("recd")
own["ftd"] = D(own.ft)
pat = pat.merge(lap, on="rgst_no", how="left").merge(rec, on="rgst_no", how="left").merge(own[["rgst_no", "ftd"]], on="rgst_no", how="left")
amax = pay.groupby("rgst_no").a.max()
def addy(rg, n):
    n = n.astype("float")
    y = rg.dt.year + n
    day = np.where((rg.dt.month == 2) & (rg.dt.day == 29), 28, rg.dt.day)
    return pd.to_datetime(pd.DataFrame(dict(year=y, month=rg.dt.month, day=day)), errors="coerce")
pat["pthru"] = addy(pat.rg, pat.rgst_no.map(amax))
DELTA = pd.Timedelta(days=335)
res = []
for t0 in range(2012, 2024):
    T = pd.Timestamp(f"{t0}-12-31"); W = pd.Timestamp(f"{t0+1}-08-31"); OBS = W + DELTA
    aT = pay[pay.pd <= T].groupby("rgst_no").a.max()
    d = pat.copy(); d["aT"] = d.rgst_no.map(aT)
    d = d[d.aT.notna()]
    d["pT"] = addy(d.rg, d.aT)
    live = d[(d.rg <= T) & (d.pT > T) & ((d.ex > W) | d.ex.isna()) & ~(d.ftd <= T)].copy()
    restored = live.recd.notna() & (live.recd >= live.lapd) & (live.recd <= OBS)
    live["lap"] = live.lapd.notna() & (live.lapd <= OBS) & (live.pthru > T) & (live.pthru <= W) & ~restored
    u = live[["rgst_no", "ksic", "lap"]].merge(app, on="rgst_no")
    g = u.groupby(["applicantCd", "ksic"]).agg(n=("rgst_no", "size"), nl=("lap", "sum")).reset_index()
    g["nk"] = g.groupby("applicantCd").ksic.transform("size")
    g["y"] = g.nl == g.n
    rec_ = dict(t0=t0, live=len(live))
    for m in (1, 2, 3):
        for multi in (True, False):
            s = g[(g.n >= m) & ((g.nk >= 2) if multi else True)]
            rec_[f"u_m{m}_{'mk' if multi else 'all'}"] = len(s)
            rec_[f"p_m{m}_{'mk' if multi else 'all'}"] = int(s.y.sum())
    # m=1 다KSIC 양성 중 n=1 비율
    s1 = g[(g.nk >= 2) & g.y]
    rec_["m1mk_pos_n1_share"] = round(float((s1.n == 1).mean()), 4)
    # 대안 정의 민감도: 창 12개월 (t0+1)-12-31, 같은 Δ
    W2 = pd.Timestamp(f"{t0+1}-12-31"); OBS2 = W2 + DELTA
    live2 = d[(d.rg <= T) & (d.pT > T) & ((d.ex > W2) | d.ex.isna()) & ~(d.ftd <= T)].copy()
    r2 = live2.recd.notna() & (live2.recd >= live2.lapd) & (live2.recd <= OBS2)
    live2["lap"] = live2.lapd.notna() & (live2.lapd <= OBS2) & (live2.pthru > T) & (live2.pthru <= W2) & ~r2
    u2 = live2[["rgst_no", "ksic", "lap"]].merge(app, on="rgst_no")
    g2 = u2.groupby(["applicantCd", "ksic"]).agg(n=("rgst_no", "size"), nl=("lap", "sum")).reset_index()
    g2["nk"] = g2.groupby("applicantCd").ksic.transform("size")
    rec_["p_m2_mk_win12"] = int(((g2.n >= 2) & (g2.nk >= 2) & (g2.nl == g2.n)).sum())
    res.append(rec_)
    print(t0, rec_["p_m2_mk"], round(time.time() - t, 1), flush=True)
R = pd.DataFrame(res)
R.to_csv(OUT + r"\v1_recount.csv", index=False, encoding="utf-8-sig")
print(R.sum(numeric_only=True).to_string())
print("elapsed", time.time() - t)
