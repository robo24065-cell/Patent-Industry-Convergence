# -*- coding: utf-8 -*-
"""C04-T1 s1: 국세청 상태조회(r3nts) 원천 프로파일 — 검증 의무 (a)(b)(d), PREREG P1·P3·P4(좌측절단).
읽기 전용 입력만 쓰고, 쓰기는 C04-T1/cache 안에서만."""
import os, sys, json, re, html, hashlib, sqlite3
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
N4 = os.path.join(MC, "N4_특허행태_기업폐업조기경보")
HERE = os.path.join(BASE, r"분석\검증루프_20261012\C04\C04-T1")
OUT = os.path.join(HERE, "cache"); DL = os.path.join(HERE, "download")
SNAP = pd.Timestamp("2026-09-29")
R = {}

# ---------- 공식표 파싱 (지표누리 2825 '연도별 사업자 현황', 단위 천명) ----------
fp = os.path.join(DL, "stbl_282501_2001_2025.html")
raw = open(fp, "rb").read()
R["official_sha256"] = hashlib.sha256(raw).hexdigest()
t = raw.decode("utf-8", "replace")
s = re.sub(r"<script.*?</script>", "", t, flags=re.S); s = re.sub(r"<style.*?</style>", "", s, flags=re.S)
s = html.unescape(re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s)))
rows = []
for m in re.finditer(r"\b(20\d\d) ((?:[\d,]+|-)(?: (?:[\d,]+|-)){14})", s):
    v = [np.nan if x == "-" else float(x.replace(",", "")) for x in m.group(2).split()]
    rows.append([int(m.group(1))] + v)
cols = ["year"] + [f"{a}_{b}" for a in ["total", "corp", "gen", "simple", "exempt"] for b in ["stock", "new", "close"]]
off = pd.DataFrame(rows, columns=cols).drop_duplicates("year").set_index("year")
off.to_csv(os.path.join(OUT, "official_bizstat_2825.csv"), encoding="utf-8-sig")
R["official_corp_close_k"] = off.corp_close.to_dict()

# ---------- r3nts 원천 ----------
st = pd.read_parquet(os.path.join(EXT, "r3nts_status.parquet"))
aps = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"))
tg = pd.read_parquet(os.path.join(EXT, r"r3nts_work\targets.parquet"))
R["n_bizno"] = len(st); R["n_pairs"] = len(aps)
R["b_stt_cd"] = st.b_stt_cd.fillna("NA").value_counts().to_dict()
R["registered_false"] = int((~st.registered.astype(bool)).sum())
R["missing_true"] = int(st.missing.astype(bool).sum())
R["query_date"] = st.query_date.value_counts().to_dict()
st["corp2"] = pd.to_numeric(st.bizno.str[3:5], errors="coerce")
st["kind"] = np.select([st.corp2.isin([81, 86, 87, 88]), st.corp2 == 85, st.corp2 == 82, st.corp2.between(1, 79), st.corp2 == 80],
                       ["corp_hq", "corp_branch", "nonprofit", "individual", "individual_exempt"], "other")
R["kind_counts"] = st.kind.value_counts().to_dict()
st = st.merge(tg[["bizno", "tier", "n_prov_rgst", "n_appcd"]], on="bizno", how="left")
# 이상값
e = st.end_dt
R["anom"] = dict(
    closed_no_enddt=int(((st.b_stt_cd == "03") & e.isna()).sum()),
    enddt_not_closed=int(((st.b_stt_cd != "03") & e.notna()).sum()),
    enddt_future=int((e > SNAP).sum()),
    enddt_before_1990=int((e < "1990-01-01").sum()),
    enddt_before_2013=int((e < "2013-01-01").sum()),
    closed=int((st.b_stt_cd == "03").sum()))
# 중복
R["dup"] = dict(bizno_dup_rows=int(st.bizno.duplicated().sum()),
                pairs_dup=int(aps.duplicated(["applicantCd", "bizno"]).sum()),
                appcd_multi_bizno=int((aps.groupby("applicantCd").bizno.nunique() > 1).sum()),
                bizno_multi_appcd=int((aps.groupby("bizno").applicantCd.nunique() > 1).sum()),
                appcd_multi_bizno_registered=int((aps[aps.registered].groupby("applicantCd").bizno.nunique() > 1).sum()))
# load_nts 와 같은 drop_duplicates 에서 상태가 엇갈리는 출원인
reg = aps[aps.registered]
mb = reg.groupby("applicantCd").filter(lambda g: g.bizno.nunique() > 1) if R["dup"]["appcd_multi_bizno_registered"] < 5000 else reg.iloc[:0]
if len(mb):
    conf = mb.groupby("applicantCd").agg(n=("bizno", "nunique"), nst=("b_stt_cd", "nunique"), anyclosed=("b_stt_cd", lambda x: (x == "03").any()))
    R["dup"]["appcd_multi_bizno_status_conflict"] = int((conf.nst > 1).sum())
    R["dup"]["appcd_multi_bizno_any_closed"] = int(conf.anyclosed.sum())

# ---------- 연도·월 분포 ----------
cl = st[(st.b_stt_cd == "03") & st.end_dt.notna()].copy()
cl["y"] = cl.end_dt.dt.year; cl["m"] = cl.end_dt.dt.month
yr = cl.groupby(["y", "kind"]).size().unstack(fill_value=0)
yr_t1 = cl[cl.tier == 1].groupby(["y", "kind"]).size().unstack(fill_value=0)
yr.to_csv(os.path.join(OUT, "nts_close_year_kind_all.csv"), encoding="utf-8-sig")
yr_t1.to_csv(os.path.join(OUT, "nts_close_year_kind_tier1.csv"), encoding="utf-8-sig")
mon = cl[cl.y >= 2015].groupby(["y", "m"]).size().unstack(fill_value=0)
mon.to_csv(os.path.join(OUT, "nts_close_month_all.csv"), encoding="utf-8-sig")
mon_c = cl[(cl.y >= 2015) & cl.kind.isin(["corp_hq", "corp_branch"])].groupby(["y", "m"]).size().unstack(fill_value=0)
mon_c.to_csv(os.path.join(OUT, "nts_close_month_corp.csv"), encoding="utf-8-sig")

# ---------- P1 공식 추세 비율 ----------
corp = yr.reindex(columns=["corp_hq", "corp_branch"], fill_value=0).sum(axis=1)
t1c = yr_t1.reindex(columns=["corp_hq", "corp_branch"], fill_value=0).sum(axis=1)
p1 = pd.DataFrame({"r3nts_corp_close": corp, "tier1_corp_close": t1c}).reindex(range(2010, 2027)).fillna(0)
p1["official_corp_close"] = off.corp_close.reindex(p1.index) * 1000
p1["official_corp_stock_prev"] = off.corp_stock.shift(1).reindex(p1.index) * 1000
p1["ratio_all"] = p1.r3nts_corp_close / p1.official_corp_close
p1["ratio_tier1"] = p1.tier1_corp_close / p1.official_corp_close
p1["official_close_rate"] = p1.official_corp_close / p1.official_corp_stock_prev
p1.to_csv(os.path.join(OUT, "p1_trend.csv"), encoding="utf-8-sig")
def p1judge(col):
    b = p1.loc[2016:2023, col]
    cv = float(b.std(ddof=1) / b.mean()); mu = float(b.mean())
    return dict(cv_2016_23=cv, mean_2016_23=mu, r2024=float(p1.loc[2024, col]), r2025=float(p1.loc[2025, col]),
                drop2024=float(1 - p1.loc[2024, col] / mu), drop2025=float(1 - p1.loc[2025, col] / mu),
                deficit_flag=bool(cv > 0.15 or p1.loc[2024, col] < 0.8 * mu or p1.loc[2025, col] < 0.8 * mu))
R["P1"] = {"all": p1judge("ratio_all"), "tier1": p1judge("ratio_tier1")}

# ---------- P3 하반기/상반기 ----------
def hh(m):
    h = {}
    for y in m.index:
        h1 = m.loc[y, 1:6].sum(); h2 = m.loc[y, 7:12].sum() if y < 2026 else np.nan
        h[int(y)] = float(h2 / h1) if h1 > 0 else np.nan
    return h
R["P3_h2h1_all"] = hh(mon); R["P3_h2h1_corp"] = hh(mon_c)
for nm, m in [("all", mon), ("corp", mon_c)]:
    h = R[f"P3_h2h1_{nm}"]; med = float(np.median([h[y] for y in range(2017, 2025)]))
    R[f"P3_deficit_2025H2_{nm}"] = float(1 - h[2025] / med)
    R[f"P3_med_h2h1_2017_24_{nm}"] = med
    # 2026 1~9월 월별 vs 2025 월평균
    R[f"P3_2025_monthly_mean_{nm}"] = float(m.loc[2025].mean())
    R[f"P3_2026_months_{nm}"] = {int(k): int(v) for k, v in m.loc[2026].items()} if 2026 in m.index else {}
# 12월 집중(직권·연말 폐업 신고) 비율
R["dec_share_by_year_corp"] = {int(y): float(mon_c.loc[y, 12] / mon_c.loc[y].sum()) for y in mon_c.index if y < 2026}

# ---------- 원천 캐시 정합 (등록원부) ----------
try:
    c = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'registry_full.sqlite')}?mode=ro", uri=True)
    R["registry_counts"] = dict(pay=c.execute("select count(*) from pay").fetchone()[0],
                                applicant=c.execute("select count(*) from applicant").fetchone()[0],
                                owner_seq_gt0=c.execute("select count(*) from owner where seq>0 and ownerRgstCsName is not null").fetchone()[0],
                                max_pay=c.execute("select max(payDate) from pay").fetchone()[0])
    c.close()
except Exception as ex:
    R["registry_counts"] = str(ex)
R["n4cache_counts"] = dict(reg_py=len(pd.read_parquet(os.path.join(N4, r"cache\reg_py.parquet"), columns=["d"])),
                           reg_own=len(pd.read_parquet(os.path.join(N4, r"cache\reg_own.parquet"), columns=["d"])))
json.dump(R, open(os.path.join(OUT, "s1_profile.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: R[k] for k in R if k not in ["official_corp_close_k"]}, ensure_ascii=False, indent=1, default=str)[:9000])
print(p1.round(4).to_string())
