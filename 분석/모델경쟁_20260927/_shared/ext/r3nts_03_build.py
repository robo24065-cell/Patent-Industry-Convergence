# -*- coding: utf-8 -*-
"""국세청 상태조회 결과 정리 + 연계율 + 시점정합 점검 + 개발구간 선행검정(T4형).
입력: r3nts_work/nts_status_raw.jsonl, r3nts_work/applicant_bizno_link.parquet (읽기전용 DB: registry_full.sqlite)
출력: r3nts_status.parquet, r3nts_applicant_status.parquet, r3nts_link_by_ksic.csv, r3nts_link_by_year.csv,
      r3nts_closure_year_ksic.parquet, r3nts_test_lapse.csv, r3nts_lapse_decomp_ksic.csv
🔒 검정 원점 L ∈ {2014,2016,2018}-01-01, 결과 3년(≤ 2020-12-31) → T4 개발 규칙(L ≤ 2019-01-01, 결과 ≤ 2021-12) 안.
"""
import os, json, sqlite3
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

EXT = Path(__file__).resolve().parent
W = EXT / "r3nts_work"
BASE = EXT.parents[3]
def ro(p): return sqlite3.connect(Path(p).as_uri() + "?mode=ro", uri=True)

# ---------- 1. 원자료 정리 ----------
recs = []
for l in open(W / "nts_status_raw.jsonl", encoding="utf-8"):
    try: recs.append(json.loads(l))
    except Exception: pass
st = pd.DataFrame(recs).drop_duplicates("b_no", keep="last").rename(columns={"b_no": "bizno"})
if "missing" not in st: st["missing"] = np.nan
st["missing"] = st["missing"].fillna(False).astype(bool)
for c in ["end_dt", "tax_type_change_dt", "invoice_apply_dt"]:
    st[c] = pd.to_datetime(st[c].replace("", np.nan), format="%Y%m%d", errors="coerce")
st["registered"] = st.b_stt_cd.fillna("").isin(["01", "02", "03"])
keep = ["bizno", "b_stt", "b_stt_cd", "end_dt", "tax_type", "tax_type_cd", "tax_type_change_dt", "utcc_yn",
        "rbf_tax_type", "rbf_tax_type_cd", "invoice_apply_dt", "registered", "missing", "query_date"]
st = st[[c for c in keep if c in st]]
st.to_parquet(EXT / "r3nts_status.parquet", index=False)
tg = pd.read_parquet(W / "targets.parquet")
q = tg.merge(st, on="bizno", how="left")
print("대상", len(tg), "조회완료", q.b_stt_cd.notna().sum() + q.missing.fillna(False).sum(), "| 국세청 등록 확인", int(st.registered.sum()))
print(st.b_stt.fillna("(미등록/빈값)").value_counts().to_string())
print("tier별 상태:\n", q.assign(s=q.b_stt.fillna("미조회/미등록")).groupby(["tier", "s"]).size().unstack(fill_value=0).to_string())
print("폐업자 중 end_dt 보유율", round(st.loc[st.b_stt_cd == "03", "end_dt"].notna().mean(), 4))

# ---------- 2. 출원인코드 단위 ----------
link = pd.read_parquet(W / "applicant_bizno_link.parquet")
ap = link.merge(st, on="bizno", how="left")
ap.to_parquet(EXT / "r3nts_applicant_status.parquet", index=False)

# ---------- 3. 제공범위 KR 등록특허 연계율 ----------
con = ro(BASE / "수집파이프라인" / "registry_full.sqlite")
pat = pd.read_sql("select rgst_no, rgstDate, cndrtExptnDate, claimCount, ksic, applDate from patent", con)
rt = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName, rgstCsReason from right where rgstCsName in ('소멸등록','권리의 말소등록')", con)
app = pd.read_sql("select rgst_no, applicantCd from applicant where seq=0", con)
con.close()
pat = pat.merge(app, on="rgst_no", how="left")
a1 = ap[ap.registered.fillna(False).astype(bool)].sort_values("n_prov_rgst", ascending=False).drop_duplicates("applicantCd")
pat = pat.merge(a1[["applicantCd", "bizno", "b_stt_cd", "end_dt"]], on="applicantCd", how="left")
pat["has_bizno"] = pat.applicantCd.isin(set(link.applicantCd))
pat["linked"] = pat.b_stt_cd.notna()
pat["closed_now"] = pat.b_stt_cd.eq("03")
pat["rgst_year"] = pat.rgstDate.str[:4].astype(int)
pat["ksic"] = pat.ksic.astype(str)
print("제공범위 KR 등록특허", len(pat), "| 사업자번호 보유 출원인", round(pat.has_bizno.mean(), 4),
      "| 국세청 상태 연결", round(pat.linked.mean(), 4), "| 연결 중 현재 폐업", round(pat.loc[pat.linked, "closed_now"].mean(), 4))
bk = pat.groupby("ksic").agg(n_pat=("rgst_no", "size"), has_bizno=("has_bizno", "mean"), linked=("linked", "mean"),
                             n_linked=("linked", "sum"), closed_share_of_linked=("closed_now", "sum")).reset_index()
bk["closed_share_of_linked"] = bk.closed_share_of_linked / bk.n_linked.clip(lower=1)
bk.round(4).to_csv(EXT / "r3nts_link_by_ksic.csv", index=False, encoding="utf-8-sig")
print("KSIC별 연결률: 최소", round(bk.linked.min(), 3), bk.loc[bk.linked.idxmin(), "ksic"], "중앙", round(bk.linked.median(), 3),
      "최대", round(bk.linked.max(), 3), bk.loc[bk.linked.idxmax(), "ksic"], "| KSIC 수", len(bk))
by = pat.groupby("rgst_year").agg(n_pat=("rgst_no", "size"), linked=("linked", "mean")).reset_index()
by.round(4).to_csv(EXT / "r3nts_link_by_year.csv", index=False, encoding="utf-8-sig")

# ---------- 4. 폐업 연도 × KSIC 패널 (특허 가중: 원점 L 기준 재구성 가능한 사건) ----------
# 특허 보유 법인(사업자번호 단위)의 폐업 연도별 수, 그 법인이 폐업 시점에 보유한 제공범위 등록특허 KSIC 가중
fz = pat[pat.closed_now & pat.end_dt.notna()].copy()
fz["end_year"] = fz.end_dt.dt.year
fz = fz[fz.rgstDate <= fz.end_dt.dt.strftime("%Y%m%d")]
w = fz.groupby(["bizno", "ksic"]).size().rename("n").reset_index()
w["w"] = w.n / w.groupby("bizno").n.transform("sum")
w = w.merge(fz.drop_duplicates("bizno")[["bizno", "end_year"]], on="bizno")
cy = w.groupby(["ksic", "end_year"]).agg(n_firm_w=("w", "sum"), n_pat_heldt=("n", "sum")).reset_index()
cy.to_parquet(EXT / "r3nts_closure_year_ksic.parquet", index=False)
print("폐업 법인(특허 보유, end_dt 있음) 연도별:", fz.drop_duplicates("bizno").end_year.value_counts().sort_index().loc[2005:2020].to_dict())

# ---------- 5. 개발구간 선행 검정: 원점 L 이전 폐업 → L 이후 3년 등록료불납 소멸 ----------
lapse = rt[(rt.rgstCsName == "소멸등록") & (rt.rgstCsReason == "등록료불납")].groupby("rgst_no").rgstCsDate.min().rename("lapse_dt")
dead = rt.groupby("rgst_no").rgstCsDate.min().rename("dead_dt")
P = pat.merge(lapse, on="rgst_no", how="left").merge(dead, on="rgst_no", how="left")
P = P[P.has_bizno]

def build(L):
    Ls = f"{L}0101"; Ld = pd.Timestamp(f"{L}-01-01")
    d = P[(P.rgstDate < Ls) & (P.dead_dt.isna() | (P.dead_dt >= Ls)) & (P.cndrtExptnDate > f"{L+3}1231")].copy()
    d["y"] = (d.lapse_dt.notna() & (d.lapse_dt <= f"{L+2}1231")).astype(int)
    d["age"] = L - d.rgstDate.str[:4].astype(int)
    d["claims"] = pd.to_numeric(d.claimCount, errors="coerce").fillna(1).clip(upper=50)
    port = P[P.rgstDate < Ls].groupby("applicantCd").size().rename("port")
    d = d.merge(port, on="applicantCd", how="left"); d["lport"] = np.log1p(d.port)
    # 시점정합: 폐업일(end_dt) < L 인 것만 '원점에 알려진 폐업'. L 이후 폐업은 '계속'으로 취급(미래정보 차단)
    d["closed_L"] = (d.end_dt.notna() & (d.end_dt < Ld)).astype(int)
    d["closed_L_1y"] = (d.end_dt.notna() & (d.end_dt < Ld) & (d.end_dt >= Ld - pd.DateOffset(years=1))).astype(int)
    d["yrs_since_close"] = np.where(d.closed_L == 1, (Ld - d.end_dt).dt.days / 365.25, 0)
    d["nts_linked"] = d.linked.astype(int)
    d["L"] = L
    return d

LS = (2014, 2016, 2018)
D = {L: build(L) for L in LS}
rows = []
for L, d in D.items():
    r_c, r_o = d.loc[d.closed_L == 1, "y"].mean(), d.loc[(d.closed_L == 0) & (d.nts_linked == 1), "y"].mean()
    print(L, "n", len(d), "포기율", round(d.y.mean(), 4), "| 원점 전 폐업 법인 특허", int(d.closed_L.sum()),
          "포기율", round(r_c, 4), "vs 계속(연결)", round(r_o, 4), "| 직전1년 폐업", int(d.closed_L_1y.sum()),
          round(d.loc[d.closed_L_1y == 1, "y"].mean(), 4))
    rows.append(dict(kind="rate", train_L=L, test_L=L, features="closed_L", auc=np.nan, n_test=len(d),
                     n_closed=int(d.closed_L.sum()), lapse_closed=round(r_c, 4), lapse_open=round(r_o, 4), lapse_all=round(d.y.mean(), 4)))
BASEF = ["age", "claims", "lport"]; NTSF = ["nts_linked", "closed_L", "closed_L_1y", "yrs_since_close"]
for tr, tes in ((2014, (2016, 2018)), (2016, (2018,))):
    for name, F in (("base", BASEF), ("base+nts", BASEF + NTSF)):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(D[tr][F], D[tr].y)
        for te in tes:
            auc = roc_auc_score(D[te].y, mdl.predict_proba(D[te][F])[:, 1])
            rows.append(dict(kind="auc", train_L=tr, test_L=te, features=name, auc=round(auc, 4), n_test=len(D[te])))
            print(tr, "->", te, name, round(auc, 4))
R = pd.DataFrame(rows); R.to_csv(EXT / "r3nts_test_lapse.csv", index=False, encoding="utf-8-sig")

# ---------- 6. 해석층: 등록료불납 소멸 중 '기업 사망 동반' 비율 (소멸일 ±1년 안 폐업) ----------
lp = P[P.lapse_dt.notna() & (P.lapse_dt <= "20201231") & (P.lapse_dt >= "20100101")].copy()
ld = pd.to_datetime(lp.lapse_dt, format="%Y%m%d", errors="coerce")
lp["firm_death"] = (lp.end_dt.notna() & ((lp.end_dt - ld).abs() <= pd.Timedelta(days=366))).astype(int)
lp["firm_dead_before"] = (lp.end_dt.notna() & (lp.end_dt < ld - pd.Timedelta(days=366))).astype(int)
dk = lp.groupby("ksic").agg(n_lapse=("rgst_no", "size"), death_pm1y=("firm_death", "mean"), dead_before=("firm_dead_before", "mean")).reset_index()
dk.round(4).to_csv(EXT / "r3nts_lapse_decomp_ksic.csv", index=False, encoding="utf-8-sig")
print("2010-2020 등록료불납 소멸(사업자번호 보유 법인 출원)", len(lp), "| 소멸 ±1년 폐업 동반", round(lp.firm_death.mean(), 4),
      "| 1년 넘게 전 폐업", round(lp.firm_dead_before.mean(), 4),
      "| KSIC별 ±1년 범위", round(dk.death_pm1y.min(), 3), "~", round(dk.death_pm1y.max(), 3))
