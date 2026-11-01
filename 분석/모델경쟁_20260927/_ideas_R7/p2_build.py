# -*- coding: utf-8 -*-
"""IPO 레이더 파일럿 2단계: 패널(레이블 + 제공데이터 피처 + 대체 산업배정 피처).
- 기반 패널: N4 VERIFY_R5 별칭 NPS 패널(final_panel_alias.parquet, 읽기만). PROTOCOL_R6 §2(원점 시점 상호 별칭) 충족.
  모집단 = 원점 t0 말까지 제공범위 KR 등록특허 ≥1, 국세청 등록 사업자번호, t0까지 미폐업인 국내법인 출원인코드.
- 추가 제외: t0-12-31까지 어떤 시장이든(코넥스 포함) 상장 이력이 있는 기업 / KIND(1999~)에 상장 기록이 없는 기존 상장사(1999 이전 상장).
- y = (t0, t0+2]에 유가증권·코스닥 '신규상장' 또는 '이전상장'. 창 안에 재상장만 있는 기업은 그 원점에서 제외. 보조 y_konex.
- 봉인: dev 레이블은 t0 ≤ 2021(상장일 ≤ 2023-12-31)만. t0=2023 행은 피처만(최종시험 설계용, 레이블 없음). 2024년 이후 상장 사건은 읽은 뒤 즉시 버린다.
출력: cache/ipo_panel.parquet, cache/p2_stats.json"""
import os, re, json, sys, time
import numpy as np, pandas as pd, duckdb
sys.dont_write_bytecode = True
R7 = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(R7); BASE = os.path.dirname(os.path.dirname(ROOT))
EXT = os.path.join(ROOT, "_shared", "ext"); META = os.path.join(ROOT, "_shared", "meta", "*.parquet").replace("\\", "/")
N4C = os.path.join(ROOT, "N4_특허행태_기업폐업조기경보", "VERIFY_R5", "cache")
SEAL = pd.Timestamp("2023-12-31")
T = time.time()
def log(*a): print(*a, f"[{round(time.time()-T)}s]", flush=True)

# ---------- 상장 사건 ----------
E = pd.read_parquet(os.path.join(R7, "cache", "ipo_events.parquet"))
E["applicantCd"] = E.applicantCd.astype(str)
E = E[E.listing_date <= SEAL].copy()          # 봉인: 2024년 이후 사건 폐기
assert E.listing_date.max() <= SEAL
E["main"] = E["시장"].isin(["유가증권", "코스닥"]) & E["상장유형"].isin(["신규상장", "이전상장"])
first_any = E.groupby("applicantCd").listing_date.min()
first_main = E[E.main].groupby("applicantCd").listing_date.min()
# 1999 이전 상장사(KIND 상장 기록 없음): DART corpcode 종목코드가 있으나 KIND 상장목록(모든 증권·유형)에 없는 법인
import sqlite3
KL = pd.read_csv(os.path.join(EXT, "r7_data_kind_listing.csv"), dtype=str)
DL = pd.read_csv(os.path.join(EXT, "r7_data_kind_delisting.csv"), dtype=str)
cc = pd.read_csv(os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_corpcode.csv"), dtype=str).dropna(subset=["stock_code"])
old = cc[~cc.stock_code.isin(set(KL["종목코드"].str.zfill(6)))]
con = sqlite3.connect(f"file:{os.path.join(BASE, r'공모전데이터/외부데이터/DART/dart_company.sqlite')}?mode=ro", uri=True)
dc = pd.read_sql("select corp_code, jurir_no, bizr_no from company", con)
CA = pd.read_parquet(os.path.join(BASE, r"공모전데이터\외부데이터\DART\corp_applicant.parquet"))
oj = old.merge(dc, on="corp_code")
old_codes = set(CA[CA.jurir.isin(set(oj.jurir_no.dropna())) | CA.bizr.isin(set(oj.bizr_no.dropna()) - {""})]["출원인코드"].astype(str))
log("1999 이전 상장 추정 법인", len(oj), "출원인코드", len(old_codes))

# ---------- 패널 ----------
P = pd.read_parquet(os.path.join(N4C, "final_panel_alias.parquet"))
P["applicantCd"] = P.applicantCd.astype(str)
P = P.drop(columns=["y"])
rows = []
st = {}
for t0, g in P.groupby("t0"):
    end = pd.Timestamp(f"{t0}-12-31")
    pre = g.applicantCd.map(first_any).le(end).fillna(False) | g.applicantCd.isin(old_codes)
    g = g[~pre].copy()
    if t0 <= 2021:
        hi = pd.Timestamp(f"{t0+2}-12-31"); assert hi <= SEAL
        fm = g.applicantCd.map(first_main)
        g["y"] = ((fm > end) & (fm <= hi)).astype(int)
        w = E[(E.listing_date > end) & (E.listing_date <= hi)]
        relist_only = set(w.applicantCd) - set(w[w.main | (w["시장"] == "코넥스")].applicantCd)
        g = g[~(g.applicantCd.isin(relist_only) & (g.y == 0))]
        g["y_konex"] = g.applicantCd.isin(set(w[w["시장"] == "코넥스"].applicantCd)).astype(int)
    else:
        g["y"] = np.nan; g["y_konex"] = np.nan
    st[int(t0)] = {"n": int(len(g)), "excluded_listed": int(pre.sum()), "pos": None if t0 > 2021 else int(g.y.sum()),
                   "rate": None if t0 > 2021 else round(float(g.y.mean()), 5)}
    rows.append(g)
P = pd.concat(rows, ignore_index=True)
log("panel", P.shape, st)
POP = set(P.applicantCd)

# ---------- 기업 특허(제공 KR) + 메인 IPC ----------
A = pd.read_parquet(os.path.join(N4C, "app_firm.parquet"))
A["applicantCd"] = A.applicantCd.astype(str)
A = A[A.applicantCd.str[0] == "1"]
kr = duckdb.sql(f"select appl_no, main_ipc from read_parquet('{META}') where cc='KR'").df()
A = A.merge(kr, on="appl_no", how="left")
A["sub"] = A.main_ipc.fillna("").str.replace(" ", "").str[:4]
A["mg"] = A.main_ipc.fillna("").str.split("/").str[0].str.replace(" ", "")
log("app_firm", A.shape, "IPC 보유", round(A.main_ipc.notna().mean(), 4))

# ---------- 공식 IPC-KSIC 연계표(2013) → 대체 산업 배정 ----------
X = pd.read_excel(os.path.join(BASE, r"공모전데이터\외부데이터\20131002_IPC-KSIC\20131002_IPC-KSIC.xlsx"), sheet_name="전체연계표", dtype=str)
X.columns = ["cls", "ipc", "cnt"]; X["cnt"] = pd.to_numeric(X.cnt)
X["ipc"] = X.ipc.str.replace(" ", "")
X["mg"] = X.ipc.str.split("/").str[0]; X["sub"] = X.ipc.str[:4]
def best(df, key):
    s = df.groupby([key, "cls"]).cnt.sum().reset_index().sort_values("cnt", ascending=False)
    return s.drop_duplicates(key).set_index(key).cls
B_sg, B_mg, B_sub = best(X, "ipc"), best(X, "mg"), best(X, "sub")
def alt_cls(ipc_series):
    ip = ipc_series.fillna("").str.replace(" ", "")
    c = ip.map(B_sg)
    lvl = pd.Series(np.where(c.notna(), "sg", ""), index=ip.index)
    m = c.isna(); c[m] = ip[m].str.split("/").str[0].map(B_mg); lvl[m & c.notna()] = "mg"
    m = c.isna(); c[m] = ip[m].str[:4].map(B_sub); lvl[m & c.notna()] = "sub"
    return c, lvl
A["alt"], A["alt_lvl"] = alt_cls(A.main_ipc)
st["alt_match_level"] = A.alt_lvl.value_counts(normalize=True).round(4).to_dict()
log("대체 배정", st["alt_match_level"])

# ---------- 전역 집계(제공 5개국): KSIC × cc × 연도, 대체분류 × cc × 연도, IPC 메인그룹 × 연도 ----------
G = duckdb.sql(f"""select cc='KR' kr, ksic, main_ipc, yr, count(*) n from (
        select cc, ksic, main_ipc, cast(substr(coalesce(nullif(pub_date,''), nullif(rgst_date,'')),1,4) as int) yr
        from read_parquet('{META}')) where yr between 2000 and 2023 group by 1,2,3,4""").df()
U = pd.Series(G.main_ipc.dropna().unique()); ua, _ = alt_cls(U)
G["alt"] = G.main_ipc.map(dict(zip(U, ua)))
G["mg"] = G.main_ipc.fillna("").str.split("/").str[0].str.replace(" ", "")
CK = G.groupby(["ksic", "yr"]).n.sum(); CKkr = G[G.kr].groupby(["ksic", "yr"]).n.sum()
CA_ = G.groupby(["alt", "yr"]).n.sum(); CAkr = G[G.kr].groupby(["alt", "yr"]).n.sum()
CM = G.groupby(["mg", "yr"]).n.sum(); CMkr = G[G.kr].groupby(["mg", "yr"]).n.sum()
st["global_alt_cover"] = round(float(G.loc[G.alt.notna(), "n"].sum() / G.n.sum()), 4)
del G
log("전역 집계 완료")

def growth(C, t0):
    c = C.unstack(fill_value=0)
    a = c.loc[:, [y for y in c.columns if t0 - 2 <= y <= t0]].sum(1); b = c.loc[:, [y for y in c.columns if t0 - 5 <= y <= t0 - 3]].sum(1)
    return np.log((a + 1) / (b + 1)), a

def heat(t0, col):
    """산업(col) 상장 열기: (t0-3) 말까지 그 산업 특허가 알려진 미상장 법인 중 (t0-3, t0]에 유가·코스닥 첫 상장한 가중 비율(평활)."""
    s = int(f"{t0-3}1231"); e0 = pd.Timestamp(f"{t0-3}-12-31"); e1 = pd.Timestamp(f"{t0}-12-31")
    a = A[A.known_date <= s].dropna(subset=[col])
    fa = a.applicantCd.map(first_any)
    a = a[~(fa <= e0)]
    w = a.groupby(["applicantCd", col]).size(); w = w / w.groupby(level=0).transform("sum")
    w = w.reset_index(name="w")
    fm = w.applicantCd.map(first_main)
    w["hit"] = ((fm > e0) & (fm <= e1)).astype(float) * w.w
    agg = w.groupby(col)[["w", "hit"]].sum()
    prior = agg.hit.sum() / agg.w.sum()
    return (agg.hit + 20 * prior) / (agg.w + 20), prior

def port_feats(t0, col, CC, CCkr, pre):
    s = int(f"{t0}1231")
    a = A[(A.known_date <= s) & A.applicantCd.isin(POP)].dropna(subset=[col])
    w = a.groupby(["applicantCd", col]).size().rename("n").reset_index()
    w["sh"] = w.n / w.groupby("applicantCd").n.transform("sum")
    g, tot = growth(CC, t0); gk, totk = growth(CCkr, t0)
    h, _ = heat(t0, col)
    rec = a[a.known_date > s - 30000].groupby(["applicantCd", col]).size().rename("r").reset_index()
    w = w.merge(rec, on=["applicantCd", col], how="left").fillna({"r": 0})
    w["g"] = w[col].map(g).fillna(0); w["gk"] = w[col].map(gk).fillna(0); w["h"] = w[col].map(h)
    w["pos"] = w.r / w[col].map(totk).replace(0, np.nan)
    o = pd.DataFrame({
        f"{pre}_n": w.groupby("applicantCd").size(),
        f"{pre}_hhi": (w.sh ** 2).groupby(w.applicantCd).sum(),
        f"{pre}_heat": (w.sh * w.h).groupby(w.applicantCd).sum(),
        f"{pre}_g3": (w.sh * w.g).groupby(w.applicantCd).sum(),
        f"{pre}_g3kr": (w.sh * w.gk).groupby(w.applicantCd).sum(),
        f"{pre}_pos": np.log1p(1e4 * (w.pos.fillna(0)).groupby(w.applicantCd).sum()),
    })
    return o

def ipc_feats(t0):
    s = int(f"{t0}1231")
    a = A[(A.known_date <= s) & A.applicantCd.isin(POP) & (A.mg != "")]
    g, _ = growth(CM, t0); gk, _ = growth(CMkr, t0)
    o = pd.DataFrame({"ipc_div": np.log1p(a.groupby("applicantCd")["sub"].nunique()),
                      "cell_hot": a.mg.map(g).groupby(a.applicantCd).mean(),
                      "cell_hot_kr": a.mg.map(gk).groupby(a.applicantCd).mean()})
    rec = a[a.known_date > s - 30000]; old_ = a[a.known_date <= s - 30000]
    seen = set(zip(old_.applicantCd, old_["sub"]))
    rec = rec.assign(new=[(c, x) not in seen for c, x in zip(rec.applicantCd, rec["sub"])])
    o["ipc_new_share"] = rec.groupby("applicantCd").new.mean()
    return o

out = []
for t0, g in P.groupby("t0"):
    fk = port_feats(t0, "ksic", CK, CKkr, "ksic")
    fa = port_feats(t0, "alt", CA_, CAkr, "alt")
    fi = ipc_feats(t0)
    g = g.join(fk, on="applicantCd").join(fa, on="applicantCd").join(fi, on="applicantCd")
    out.append(g); log("t0", t0, g.shape)
P = pd.concat(out, ignore_index=True)
P.to_parquet(os.path.join(R7, "cache", "ipo_panel.parquet"), index=False)
st["cols"] = [c for c in P.columns]
json.dump(st, open(os.path.join(R7, "cache", "p2_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("저장", P.shape)
