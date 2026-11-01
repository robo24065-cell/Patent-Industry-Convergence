# -*- coding: utf-8 -*-
"""P1 '기술의 산업 간 이동 레이더' 패널 생성.
단위 = (원점 t, IPC 메인그룹 g, 제공 KSIC k). 결과 = g의 KR 문서 중 k 몫이 (t-2..t)→(t+1..t+3)에 10%p 이상 상승(그리고 미래 k 문서 ≥3).
연도는 최초공표연도(min(공개,등록)). 개발 원점 t≤2018(결과 ≤2021). 🔒 2022 이후 공표 레코드는 읽자마자 버린다(assert).
배정 3종: prov(제공 라벨) / cw(연계표 그룹) / txt(연계표 그룹+KSIC 해설서 유사도, KR만 5자리, 해외는 그룹).
"""
import sys
sys.dont_write_bytecode = True
from r8_common import *
import duckdb

PY_MAX = 2021  # 개발 상한(최종시험 결과 구간 2022–2024 봉인)
con = duckdb.connect(); con.sql("SET threads=2; SET memory_limit='3GB'")
R, C, A = p('cache', 'rec.parquet'), p('cache', 'ipc_cw.parquet'), p('cache', 'kr_assign.parquet')
con.sql(f"""CREATE TABLE r AS SELECT r.cc, r.mg, r.pyear, r.ksic AS prov, c.cw,
   CASE WHEN r.cc='KR' THEN a.k_txt ELSE c.cw END AS txt
   FROM '{R}' r LEFT JOIN '{C}' c USING (sg, mg, sc)
   LEFT JOIN '{A}' a ON r.cc='KR' AND a.appl_no=r.appl_no
   WHERE r.mg<>'' AND r.pyear BETWEEN 2000 AND {PY_MAX}""")
assert con.sql("select max(pyear) from r").fetchone()[0] <= PY_MAX
tot = con.sql("SELECT cc, mg, pyear, count(*) n FROM r GROUP BY ALL").df()
cnt = {a: con.sql(f"SELECT cc, mg, {a} AS lab, pyear, count(*) n FROM r WHERE {a} IS NOT NULL GROUP BY ALL").df()
       for a in ("prov", "cw", "txt")}
log("counts", {a: len(v) for a, v in cnt.items()})
CCS = ["KR", "US", "JP", "CN", "EP"]


def wsum(df, keys, y0, y1):
    m = (df.pyear >= y0) & (df.pyear <= y1)
    return df[m].groupby(keys).n.sum()


def build_units(t):
    """제공 데이터로 단위 정의(모든 배정에 공통)."""
    kr_tot = tot[tot.cc == "KR"]
    nP_g = wsum(kr_tot, "mg", t - 2, t)
    gs = nP_g[nP_g >= 30].index
    pc = cnt["prov"]
    near = wsum(pc[pc.mg.isin(gs)], ["mg", "lab"], t - 5, t)
    u = near[near >= 2].reset_index()[["mg", "lab"]].rename(columns={"lab": "k"})
    u["t"] = t
    return u


def outcome(u, t):
    kr = cnt["prov"][cnt["prov"].cc == "KR"]
    kt = tot[tot.cc == "KR"]
    nP = wsum(kr, ["mg", "lab"], t - 2, t); nF = wsum(kr, ["mg", "lab"], t + 1, t + 3)
    gP = wsum(kt, "mg", t - 2, t); gF = wsum(kt, "mg", t + 1, t + 3)
    key = list(zip(u.mg, u.k))
    a = np.array([nP.get(x, 0) for x in key], float); b = np.array([nF.get(x, 0) for x in key], float)
    ga = u.mg.map(gP).fillna(0).values; gb = u.mg.map(gF).fillna(0).values
    shP = a / np.maximum(ga, 1); shF = np.where(gb > 0, b / np.maximum(gb, 1), 0)
    y = ((shF - shP >= 0.10) & (b >= 3)).astype(int)
    # 보조 결과: 추월(과거 g 안 KR 최다 KSIC가 아니었는데 미래 최다)
    top = {}
    for w, (y0, y1) in (("P", (t - 2, t)), ("F", (t + 1, t + 3))):
        s = wsum(kr, ["mg", "lab"], y0, y1).reset_index().sort_values(["mg", "n", "lab"], ascending=[True, False, True])
        top[w] = s.drop_duplicates("mg").set_index("mg").lab
    ov = ((u.mg.map(top["P"]) != u.k) & (u.mg.map(top["F"]) == u.k)).astype(int).values
    return y, ov, shF - shP


def feats(u, t, a):
    """배정 a의 라벨로 피처. 단위의 k는 배정 공간으로 사상."""
    c = cnt[a]
    if a == "prov":
        lab_kr = u.k.values; lab_f = u.k.values
    elif a == "cw":
        lab_kr = u.k.map(K2G).values; lab_f = lab_kr
    else:  # txt: KR은 5자리, 해외는 그룹
        lab_kr = u.k.values; lab_f = u.k.map(K2G).values
    out = {}
    for cc in CCS:
        lab = lab_kr if cc == "KR" else lab_f
        cx = c[c.cc == cc]; tx = tot[tot.cc == cc]
        W = {"P": (t - 2, t), "PP": (t - 5, t - 3), "T": (t, t)}
        sh = {}
        for w, (y0, y1) in W.items():
            n = wsum(cx, ["mg", "lab"], y0, y1); g = wsum(tx, "mg", y0, y1)
            num = np.array([n.get(x, 0) for x in zip(u.mg, lab)], float)
            den = u.mg.map(g).fillna(0).values
            sh[w] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
            if w == "P":
                out[f"{cc}_lnn"] = np.log1p(num); out[f"{cc}_lng"] = np.log1p(den)
            if w == "PP":
                out[f"{cc}_lngPP"] = np.log1p(den)
        out[f"{cc}_shP"] = sh["P"]; out[f"{cc}_dsh"] = sh["P"] - sh["PP"]
        if cc == "KR":
            out["KR_shPP"] = sh["PP"]; out["KR_shT"] = sh["T"]; out["KR_dshT"] = sh["T"] - sh["P"]
        # 산업(라벨) 전체 모멘텀: 모든 g 합계
        nPk = cx[(cx.pyear >= t - 2) & (cx.pyear <= t)].groupby("lab").n.sum()
        nPPk = cx[(cx.pyear >= t - 5) & (cx.pyear <= t - 3)].groupby("lab").n.sum()
        out[f"{cc}_kgrow"] = np.log1p(pd.Series(lab).map(nPk).fillna(0).values) - np.log1p(pd.Series(lab).map(nPPk).fillna(0).values)
    o = pd.DataFrame(out, index=u.index)
    o["g_grow"] = o.KR_lng - o.KR_lngPP
    for cc in CCS[1:]:
        o[f"{cc}_gap"] = o[f"{cc}_shP"] - o["KR_shP"]
    # 라벨 공간에서 g 안에 KR 문서가 있는 라벨 수(형제 경쟁)
    kr = c[(c.cc == "KR") & (c.pyear >= t - 2) & (c.pyear <= t)]
    nl = kr.groupby("mg").lab.nunique()
    o["n_lab_g"] = u.mg.map(nl).fillna(0).values
    return o


rows = []
for t in range(2006, 2022):
    u = build_units(t)
    if t <= 2018:
        y, ov, dsh = outcome(u, t)
        assert t + 3 <= PY_MAX
    else:  # 최종시험 설계용 단위(결과 없음)
        y = ov = dsh = np.full(len(u), -1)
    u["y"] = y; u["ov"] = ov; u["dshF"] = dsh
    cwg = con.sql("SELECT mg, cw FROM (SELECT mg, cw, count(*) n FROM r WHERE cc='KR' AND cw IS NOT NULL GROUP BY ALL) QUALIFY row_number() OVER (PARTITION BY mg ORDER BY n DESC, cw)=1").df().set_index("mg").cw
    u["cw_g"] = u.mg.map(cwg)
    u["W"] = ((u.k.map(K2G) == u.cw_g) & u.k.isin(MULTI_K)).astype(int)
    for a in ("prov", "cw", "txt"):
        f = feats(u, t, a)
        f.columns = [f"{a}__{x}" for x in f.columns]
        u = pd.concat([u, f], axis=1)
    rows.append(u)
    log(t, len(u), "pos", int((u.y == 1).sum()), "W", int(u.W.sum()))
P = pd.concat(rows, ignore_index=True)
P.to_parquet(p("cache", "p1_panel.parquet"), index=False)
d = P[P.t <= 2018]
st = d.groupby("t").agg(n=("y", "size"), pos=("y", "sum"), base=("y", "mean"), W=("W", "sum"), ov=("ov", "sum")).reset_index()
st.to_csv(p("p1_panel_stats.csv"), index=False)
print(st.to_string())
log("end")
