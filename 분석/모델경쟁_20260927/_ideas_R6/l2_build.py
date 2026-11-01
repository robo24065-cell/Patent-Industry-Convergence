# -*- coding: utf-8 -*-
"""K1(국내 최초출원 → 해외 US/JP/EP 공개)·K2(US/JP/EP 최초출원 → KR 공개) 패널 빌더.
단위: (최초출원, t0) — t0 = 그 출원이 처음 공개(공개·등록 중 이른 날)된 해. t0 말까지 상대 관청 공개가 없는 것만.
결과: 상대 관청 첫 공개연도 ∈ [t0+1, t0+2]. 🔒 개발 원점 t0 ≤ 2020(결과 ≤ 2022). 최종 t0=2022 → 2023–24 공개(미채점).
피처는 t0 말까지 보이는 정보만(토큰 연도 ≤ t0, 출원인 이력은 공개연도 ≤ t0−1 단위 + 상대 관청 공개 ≤ t0)."""
import os, re, time, numpy as np, pandas as pd, duckdb
from r6_common import CACHE, EXT, BASE

t = time.time()
C = CACHE.replace("\\", "/")
T0_MIN, T0_MAX = 2006, 2020
EU = {"EP", "DE", "FR", "GB", "NL", "SE", "CH", "FI", "IT", "DK", "AT", "BE", "ES", "NO", "IE", "LU", "PL"}


def norm(s):
    s = s.fillna("").str.split("|").str[0].str.lower()
    s = s.str.replace(r"\(주\)|㈜|주식회사|유한회사|\(유\)|유한책임회사|\(사\)|사단법인|재단법인|\(재\)|株式会社|有限公司|co\.?,?\s*ltd\.?|inc\.?|corp(oration)?\.?|limited|gmbh|\bag\b|\bs\.?a\.?\b|llc", "", regex=True)
    return s.str.replace(r"[\s\.,\-·()\[\]]", "", regex=True)


def apptype(s):
    s = s.fillna("").str.lower()
    out = np.full(len(s), 0, dtype=np.int8)  # 0 개인
    comp = s.str.contains(r"주식회사|㈜|\(주\)|inc|corp|co\.|ltd|gmbh|\bag\b|株式会社|有限公司|llc|company|kabushiki|kaisha|s\.a\.|b\.v\.|\bplc\b|컴퍼니|코포레이션|리미티드|인코포레이티드|게엠베하|카부시키", regex=True)
    univ = s.str.contains(r"대학|university|大学|univ|college|산학협력|institute of technology|kaist|포항공과", regex=True)
    res = s.str.contains(r"연구원|연구소|institute|研究所|研究院|fraunhofer|cnrs|research|commissariat|national|國立|国立|산업기술|전자통신", regex=True)
    out[comp.values] = 1; out[res.values] = 2; out[univ.values] = 3
    return out


def load(cc):
    con = duckdb.connect(); con.sql("set threads=2; set memory_limit='3GB'")
    d = con.sql(f"""select least(coalesce(nullif(trim(p.pubd),''),'99999999'), coalesce(nullif(trim(p.rgd),''),'99999999')) vis, p.appl_no, p.ksic, p.nk, p.ad, p.ay, p.pubd, p.rgd, p.pcc, p.ipc, p.ipcs, p.app,
        k.y_us, k.y_jp, k.y_ep, k.y_wo, k.y_kr
      from '{C}/p_{cc}.parquet' p left join '{C}/tok_{cc}.parquet' k using(appl_no)""").df()
    vis = d.vis
    for c in ["y_us", "y_jp", "y_ep", "y_wo", "y_kr", "nk"]:
        d[c] = d[c].astype(float)
    d["va"] = vis.str[:4].astype(float)
    d["vm"] = vis.str[4:6].astype(float)
    d["lag_m"] = (d.va - d.ad.str[:4].astype(float)) * 12 + d.vm - d.ad.str[4:6].astype(float)
    d["reg_first"] = (d.rgd.fillna("").str.strip().str[:8] == d.vis).astype(int)
    d["rgy"] = pd.to_numeric(d.rgd.fillna("").str.strip().str[:4], errors="coerce")
    d["cc"] = cc
    return d


def cum_rate(df, key, ev_n, ev_a, years):
    """key별·t0별: N=#(ev_n ≤ t0−1), A=#(ev_a ≤ t0) — ev_a는 이미 max(va+1, 상대공개연도)."""
    out = []
    for nm, ev in (("N", ev_n), ("A", ev_a)):
        g = df.assign(e=ev).dropna(subset=["e"]).groupby([key, "e"]).size().rename("c").reset_index()
        g = g.sort_values([key, "e"]); g["cs"] = g.groupby(key)["c"].cumsum()
        out.append(g[[key, "e", "cs"]].rename(columns={"cs": nm, "e": "yr"}))
    return out


def asof(units, tab, key, col, newcol, shift):
    """units의 (key, t0)에 tab의 (key, yr ≤ t0+shift) 누적값을 붙인다."""
    u = units[[key, "t0"]].copy(); u["q"] = (u.t0 + shift).astype(float); u["_i"] = np.arange(len(u))
    tab = tab.dropna(subset=[key]).copy(); tab["yr"] = tab.yr.astype(float); tab = tab.sort_values("yr"); u = u.sort_values("q")
    m = pd.merge_asof(u, tab.rename(columns={col: newcol})[[key, "yr", newcol]], left_on="q", right_on="yr", by=key, direction="backward")
    return m.sort_values("_i")[newcol].fillna(0).values


def hist_feats(units, pool, key, pref, other_year):
    """pool: 같은 관청의 전체 단위(공개 ≤2022). other_year: 상대 관청 첫 공개연도 열 이름."""
    ev_n = pool["va"] + 1  # 공개된 다음 해부터 이력으로 셈(t0−1까지)
    ev_a = np.where(pool[other_year].notna(), np.maximum(pool["va"] + 1, pool[other_year]), np.nan)
    tN, tA = cum_rate(pool, key, ev_n, ev_a, None)
    N = asof(units, tN, key, "N", "N", 0); A = asof(units, tA, key, "A", "A", 0)
    units[f"{pref}_n"] = np.log1p(N)
    units[f"{pref}_rate"] = np.where(N > 0, A / np.maximum(N, 1), np.nan)
    units[f"{pref}_rate_sm"] = (A + 0.5) / (N + 10)


def base_feats(d):
    d["ksic_i"] = d.ksic.astype(int); d["ksic2"] = d.ksic.str[:2].astype(int)
    d["ipc_sec"] = d.ipc.fillna("Z").str[0].map(lambda c: ord(c) - 65)
    d["mg"] = d.ipc.fillna("").str.split("/").str[0].str.strip()
    d["n_ipc"] = d.ipcs.fillna("").str.count(r"\|") + 1
    d["n_app"] = d.app.fillna("").str.count(r"\|") + 1
    d["atype"] = apptype(d.app)
    for k, nm in enumerate(["indiv", "comp", "res", "univ"]):
        d[f"at_{nm}"] = (d.atype == k).astype(int)
    d["akey"] = d.cc + ":" + norm(d.app)
    d["akey_ks"] = d.akey + ":" + d.ksic
    return d


def ksic_context():
    """5개국 KSIC×공개연도 건수(캐시 = 공개 ≤2022) → t0 시점 맥락 피처."""
    con = duckdb.connect(); con.sql("set threads=2; set memory_limit='3GB'")
    q = " union all ".join([f"""select '{c}' cc, ksic, cast(substr(least(coalesce(nullif(trim(pubd),''),'99999999'), coalesce(nullif(trim(rgd),''),'99999999')),1,4) as int) va from '{C}/p_{c}.parquet'""" for c in ["KR", "US", "JP", "CN", "EP"]])
    g = con.sql(f"select ksic, cc, va, count(*) n from ({q}) group by all").df()
    p = g.pivot_table(index=["ksic", "va"], columns="cc", values="n", fill_value=0).reset_index()
    return p


def ctx_feats(units, ctx):
    rows = []
    for t0 in sorted(units.t0.unique()):
        a = ctx[(ctx.va >= t0 - 2) & (ctx.va <= t0)].groupby("ksic")[["KR", "US", "JP", "CN", "EP"]].sum()
        b = ctx[(ctx.va >= t0 - 5) & (ctx.va <= t0 - 3)].groupby("ksic")[["KR", "US", "JP", "CN", "EP"]].sum()
        r = pd.DataFrame(index=a.index)
        tot_a = a.sum(axis=1); tot_b = b.reindex(a.index).fillna(0).sum(axis=1)
        r["k_world_g"] = np.log1p(tot_a) - np.log1p(tot_b)
        r["k_kr_share"] = a.KR / tot_a.clip(lower=1)
        r["k_kr_g"] = np.log1p(a.KR) - np.log1p(b.reindex(a.index).fillna(0).KR)
        r["k_cn_g"] = np.log1p(a.CN) - np.log1p(b.reindex(a.index).fillna(0).CN)
        r["k_cn_share"] = a.CN / tot_a.clip(lower=1)
        r["k_us_share"] = a.US / tot_a.clip(lower=1)
        r["t0"] = t0; rows.append(r.reset_index())
    R = pd.concat(rows)
    return units.merge(R, on=["ksic", "t0"], how="left")


def trade_feats(units):
    mp = pd.read_csv(os.path.join(BASE, r"분석\fusion\ksic_hs6_map.csv"), dtype=str, encoding="utf-8-sig")[["k5", "hs6"]].drop_duplicates()
    ex = pd.read_csv(os.path.join(EXT, "R2global_kr_export_hs6.csv"), dtype={"hs6": str}, encoding="utf-8-sig")
    im = pd.read_csv(os.path.join(BASE, r"분석\fusion\hs6_import_yearly.csv"), dtype={"hs6": str}, encoding="utf-8-sig")
    ex = ex.merge(mp, on="hs6").groupby(["k5", "year"]).value.sum().rename("ex").reset_index()
    imw = im[im.partner == "WLD"].merge(mp, on="hs6").groupby(["k5", "year"]).value.sum().rename("imw").reset_index()
    imc = im[im.partner == "CN"].merge(mp, on="hs6").groupby(["k5", "year"]).value.sum().rename("imc").reset_index()
    T = ex.merge(imw, on=["k5", "year"], how="outer").merge(imc, on=["k5", "year"], how="outer").fillna(0)
    T = T[T.year <= 2021]  # 공표 시차: t0 피처에는 t0−1 값만
    rows = []
    for t0 in sorted(units.t0.unique()):
        a = T[T.year == t0 - 1].set_index("k5"); b = T[T.year == t0 - 4].set_index("k5").reindex(a.index).fillna(0)
        r = pd.DataFrame(index=a.index)
        r["x_ex_l"] = np.log1p(a.ex); r["x_ex_g"] = np.log1p(a.ex) - np.log1p(b.ex)
        r["x_im_l"] = np.log1p(a.imw); r["x_im_g"] = np.log1p(a.imw) - np.log1p(b.imw)
        r["x_imcn_sh"] = a.imc / a.imw.clip(lower=1); r["x_imcn_g"] = np.log1p(a.imc) - np.log1p(b.imc)
        r["x_net"] = (a.ex - a.imw) / (a.ex + a.imw).clip(lower=1)
        r["t0"] = t0; rows.append(r.reset_index().rename(columns={"k5": "ksic"}))
    return units.merge(pd.concat(rows), on=["ksic", "t0"], how="left")


ctx = ksic_context(); print("ctx", ctx.shape, round(time.time() - t, 1), flush=True)

# ---------------- K1 ----------------
kr = base_feats(load("KR"))
kr["fy"] = kr[["y_us", "y_jp", "y_ep"]].min(axis=1)
first = kr.pcc.isna() | (kr.pcc.fillna("").str.replace("KR", "").str.replace("|", "") == "")
pool = kr[first].copy()
pool["util"] = pool.appl_no.str.startswith("20").astype(int); pool["dom_prio"] = pool.pcc.notna().astype(int)
U = pool[(pool.va >= T0_MIN) & (pool.va <= T0_MAX)].copy(); U["t0"] = U.va.astype(int)
U = U[~(U.fy <= U.t0)].copy()  # t0 말까지 해외 공개된 것은 제외
U["y"] = ((U.fy >= U.t0 + 1) & (U.fy <= U.t0 + 2)).astype(int)
assert (U.t0 + 2).max() <= 2022 and U.loc[U.y == 1, "fy"].max() <= 2022
U["y_us"] = ((U.y_us >= U.t0 + 1) & (U.y_us <= U.t0 + 2)).astype(int)
U["wo_vis"] = (U.y_wo <= U.t0).astype(int); U["reg_by_t0"] = (U.rgy <= U.t0).astype(int)
hist_feats(U, pool, "akey", "h_app", "fy")
hist_feats(U, pool, "akey_ks", "h_appks", "fy")
pool["mg_k"] = pool.mg; U["mg_k"] = U.mg
hist_feats(U, pool, "mg_k", "h_mg", "fy")
pool["ks_k"] = pool.ksic; U["ks_k"] = U.ksic
hist_feats(U, pool, "ks_k", "h_ks", "fy")
# WO 성향(출원인): 공개 ≤ t0−1 단위 중 WO 공개 ≤ t0 비율
hist_feats(U, pool, "akey", "h_appwo", "y_wo")
U = ctx_feats(U, ctx); U = trade_feats(U)
# NTIS 정부R&D 기원(성과 보고연도 ≤ t0)
nt = duckdb.sql(f"""select appl_no, min(try_cast(Year as int)) ny, max((perform_grp='기업')::int) n_firm, max((perform_grp='대학')::int) n_univ
   from '{EXT.replace(chr(92), '/')}/ntis_patent_link.parquet' where cc='KR' group by 1""").df()
U = U.merge(nt, on="appl_no", how="left")
U["gov_rnd"] = (U.ny <= U.t0).astype(int); U["gov_firm"] = ((U.ny <= U.t0) & (U.n_firm == 1)).astype(int); U["gov_univ"] = ((U.ny <= U.t0) & (U.n_univ == 1)).astype(int)
keep = [c for c in U.columns if c not in ("ipcs", "fam", "pubd", "rgd", "pcc", "ad", "ipc")]
U[keep].to_parquet(os.path.join(CACHE, "K1_panel.parquet"), index=False)
print("K1", U.groupby("t0").agg(n=("y", "size"), pos=("y", "sum"), rate=("y", "mean")).T.round(4).to_string(), round(time.time() - t, 1), flush=True)
del kr, pool, U

# ---------------- K2 ----------------
parts = []
for cc in ["US", "JP", "EP"]:
    d = base_feats(load(cc))
    toks = d.pcc.fillna("").str.split("|")
    if cc == "EP":
        first = toks.map(lambda L: all((x == "" or x in EU) for x in L))
    else:
        first = toks.map(lambda L: all((x == "" or x == cc) for x in L))
    d = d[first].copy()
    d["util"] = d.appl_no.str.endswith("U").astype(int) if cc == "JP" else 0
    d["dom_prio"] = d.pcc.notna().astype(int)
    parts.append(d); print(cc, len(d), round(time.time() - t, 1), flush=True)
pool = pd.concat(parts, ignore_index=True); del parts
pool["ky"] = pool.y_kr
U = pool[(pool.va >= T0_MIN) & (pool.va <= T0_MAX)].copy(); U["t0"] = U.va.astype(int)
U = U[~(U.ky <= U.t0)].copy()
U["y"] = ((U.ky >= U.t0 + 1) & (U.ky <= U.t0 + 2)).astype(int)
assert (U.t0 + 2).max() <= 2022 and U.loc[U.y == 1, "ky"].max() <= 2022
U["wo_vis"] = (U.y_wo <= U.t0).astype(int); U["reg_by_t0"] = (U.rgy <= U.t0).astype(int)
own = {"US": "y_us", "JP": "y_jp", "EP": "y_ep"}
oth = []
for cc in ["US", "JP", "EP"]:
    m = U.cc == cc
    cols = [c for c in ["y_us", "y_jp", "y_ep"] if c != own[cc]]
    oth.append(pd.Series((U.loc[m, cols].le(U.loc[m, "t0"], axis=0)).sum(axis=1), index=U.index[m]))
U["n_oth_vis"] = pd.concat(oth).reindex(U.index).values
U["cc_us"] = (U.cc == "US").astype(int); U["cc_jp"] = (U.cc == "JP").astype(int)
hist_feats(U, pool, "akey", "h_app", "ky")
hist_feats(U, pool, "akey_ks", "h_appks", "ky")
pool["mg_k"] = pool.cc + pool.mg; U["mg_k"] = U.cc + U.mg
hist_feats(U, pool, "mg_k", "h_mg", "ky")
pool["ks_k"] = pool.cc + pool.ksic; U["ks_k"] = U.cc + U.ksic
hist_feats(U, pool, "ks_k", "h_ks", "ky")
hist_feats(U, pool, "akey", "h_appwo", "y_wo")
U = ctx_feats(U, ctx); U = trade_feats(U)
# ETSI 표준특허 선언(첫 선언 ≤ t0)
et = duckdb.sql(f"""select cc, appl_no, min(year(first_decl)) ey from '{EXT.replace(chr(92), '/')}/quality_etsi_sep_link.parquet' group by 1,2""").df()
U = U.merge(et, on=["cc", "appl_no"], how="left"); U["sep_decl"] = (U.ey <= U.t0).astype(int)
keep = [c for c in U.columns if c not in ("ipcs", "fam", "pubd", "rgd", "pcc", "ad", "ipc")]
U[keep].to_parquet(os.path.join(CACHE, "K2_panel.parquet"), index=False)
print("K2", U.groupby("t0").agg(n=("y", "size"), pos=("y", "sum"), rate=("y", "mean")).T.round(4).to_string(), round(time.time() - t, 1), flush=True)
print(U.groupby("cc").y.agg(["size", "mean"]))
