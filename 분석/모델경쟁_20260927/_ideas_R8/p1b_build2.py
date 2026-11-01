# -*- coding: utf-8 -*-
"""P1 '기술의 산업 간 이동 레이더' 패널 생성 (p1b_build.py의 벡터화판 — 원판은 행 단위 조회와 조인 조건 때문에 9분 안에 끝나지 않았다).
단위 = (원점 t, IPC 메인그룹 g, 제공 KSIC k). 결과 = g의 KR 문서 중 k 몫이 (t-2..t)→(t+1..t+3)에 10%p 이상 상승(미래 k 문서 ≥3).
🔒 공표연도 2022 이후 레코드는 읽자마자 버린다(assert). 배정 3종: prov / cw / txt.
"""
import sys
sys.dont_write_bytecode = True
from r8_common import *
import duckdb

PY_MAX = 2021
Y0 = 2000
YRS = list(range(Y0, PY_MAX + 1))
con = duckdb.connect(); con.sql("SET threads=2; SET memory_limit='3GB'")
R, C, A = [p('cache', x).replace("\\", "/") for x in ('rec.parquet', 'ipc_cw.parquet', 'kr_assign.parquet')]
con.sql(f"""CREATE TABLE r AS SELECT r.cc, r.mg, r.pyear, r.ksic AS prov, c.cw,
   CASE WHEN r.cc='KR' THEN a.k_txt ELSE c.cw END AS txt
   FROM '{R}' r LEFT JOIN '{C}' c ON c.sg=r.sg AND c.mg=r.mg AND c.sc=r.sc
   LEFT JOIN (SELECT appl_no, k_txt, 'KR' AS cc FROM '{A}') a ON a.cc=r.cc AND a.appl_no=r.appl_no
   WHERE r.mg<>'' AND r.pyear BETWEEN {Y0} AND {PY_MAX}""")
assert con.sql("select max(pyear) from r").fetchone()[0] <= PY_MAX
CCS = ["KR", "US", "JP", "CN", "EP"]


def wide(sql_keys, where="TRUE", lab=None):
    sel = ", ".join(sql_keys)
    q = f"SELECT {sel}, pyear, count(*) n FROM r WHERE {where} GROUP BY ALL"
    d = con.sql(q).df()
    w = d.pivot_table(index=[k.split(" AS ")[-1] for k in sql_keys], columns="pyear", values="n", aggfunc="sum", fill_value=0)
    w = w.reindex(columns=YRS, fill_value=0).astype(np.int32)
    cs = np.cumsum(w.values, axis=1)
    return w.index, np.concatenate([np.zeros((len(w), 1), np.int64), cs], axis=1)  # cs[:, y-Y0+1] = ≤y 누적


class WS:
    def __init__(self, idx, cs):
        self.idx, self.cs = idx, cs

    def get(self, keys, y0, y1):
        """keys: MultiIndex/Index-호환 리스트 → 창 합계 배열(없으면 0)."""
        pos = self.idx.get_indexer(keys)
        a = max(y0, Y0) - Y0; b = min(y1, PY_MAX) - Y0 + 1
        v = self.cs[:, b] - self.cs[:, a] if b > a else np.zeros(len(self.cs))
        out = np.where(pos >= 0, v[np.maximum(pos, 0)], 0)
        return out.astype(float)


TOT = WS(*wide(["cc", "mg"]))
CNT = {a: WS(*wide(["cc", "mg", f"{a} AS lab"], f"{a} IS NOT NULL")) for a in ("prov", "cw", "txt")}
LAB = {a: WS(*wide(["cc", f"{a} AS lab"], f"{a} IS NOT NULL")) for a in ("prov", "cw", "txt")}
ALLP = WS(*wide(["mg", "prov AS lab"]))
log("wide done", {a: len(v.idx) for a, v in CNT.items()})
# mg → 지배 연계표 그룹(KR, 전 기간 ≤2021)
cwg = con.sql("SELECT mg, cw FROM (SELECT mg, cw, count(*) n FROM r WHERE cc='KR' AND cw IS NOT NULL GROUP BY ALL) QUALIFY row_number() OVER (PARTITION BY mg ORDER BY n DESC, cw)=1").df().set_index("mg").cw
MGS = TOT.idx[TOT.idx.get_level_values(0) == "KR"].get_level_values(1)
mi = lambda *cols: pd.MultiIndex.from_arrays(list(cols))


def units(t):
    krg = TOT.get(mi(["KR"] * len(MGS), MGS), t - 2, t)
    gs = set(MGS[krg >= 30])
    idx = ALLP.idx
    v = ALLP.get(idx, t - 5, t)
    m = (v >= 2) & np.array([g in gs for g in idx.get_level_values(0)])
    u = pd.DataFrame({"mg": idx.get_level_values(0)[m], "k": idx.get_level_values(1)[m]})
    u["t"] = t
    return u


def share_feats(u, t, a, lab_kr, lab_f):
    out = {}
    for cc in CCS:
        lab = lab_kr if cc == "KR" else lab_f
        kk = mi([cc] * len(u), u.mg.values, lab); gk = mi([cc] * len(u), u.mg.values)
        sh = {}
        for w, (y0, y1) in {"P": (t - 2, t), "PP": (t - 5, t - 3), "T": (t, t)}.items():
            num = CNT[a].get(kk, y0, y1); den = TOT.get(gk, y0, y1)
            sh[w] = np.where(den > 0, num / np.maximum(den, 1), np.nan)
            if w == "P":
                out[f"{cc}_lnn"] = np.log1p(num); out[f"{cc}_lng"] = np.log1p(den)
            if w == "PP":
                out[f"{cc}_lngPP"] = np.log1p(den)
        out[f"{cc}_shP"] = sh["P"]; out[f"{cc}_dsh"] = sh["P"] - sh["PP"]
        if cc == "KR":
            out["KR_shPP"] = sh["PP"]; out["KR_shT"] = sh["T"]; out["KR_dshT"] = sh["T"] - sh["P"]
        lk = mi([cc] * len(u), lab)
        out[f"{cc}_kgrow"] = np.log1p(LAB[a].get(lk, t - 2, t)) - np.log1p(LAB[a].get(lk, t - 5, t - 3))
    o = pd.DataFrame(out, index=u.index)
    o["g_grow"] = o.KR_lng - o.KR_lngPP
    for cc in CCS[1:]:
        o[f"{cc}_gap"] = o[f"{cc}_shP"] - o["KR_shP"]
    # g 안에서 KR 문서가 있는 라벨 수
    ix = CNT[a].idx
    krm = ix.get_level_values(0) == "KR"
    v = CNT[a].get(ix[krm], t - 2, t)
    d = pd.DataFrame({"mg": ix[krm].get_level_values(1), "n": v})
    nl = d[d.n > 0].groupby("mg").size()
    o["n_lab_g"] = u.mg.map(nl).fillna(0).values
    return o


def event(u, t, a, lab):
    kk = mi(["KR"] * len(u), u.mg.values, lab); gk = mi(["KR"] * len(u), u.mg.values)
    a0 = CNT[a].get(kk, t - 2, t); b0 = CNT[a].get(kk, t + 1, t + 3)
    ga = TOT.get(gk, t - 2, t); gb = TOT.get(gk, t + 1, t + 3)
    shP = a0 / np.maximum(ga, 1); shF = np.where(gb > 0, b0 / np.maximum(gb, 1), 0)
    return ((shF - shP >= 0.10) & (b0 >= 3)).astype(int), shF - shP


def overtake(u, t):
    ix = CNT["prov"].idx; krm = ix.get_level_values(0) == "KR"
    top = {}
    for w, (y0, y1) in (("P", (t - 2, t)), ("F", (t + 1, t + 3))):
        d = pd.DataFrame({"mg": ix[krm].get_level_values(1), "lab": ix[krm].get_level_values(2), "n": CNT["prov"].get(ix[krm], y0, y1)})
        d = d[d.n > 0].sort_values(["mg", "n", "lab"], ascending=[True, False, True]).drop_duplicates("mg")
        top[w] = d.set_index("mg").lab
    return ((u.mg.map(top["P"]) != u.k) & (u.mg.map(top["F"]) == u.k)).astype(int).values


rows = []
for t in range(2006, 2022):
    u = units(t)
    kg = u.k.map(K2G).values
    if t <= 2018:
        assert t + 3 <= PY_MAX
        u["y"], u["dshF"] = event(u, t, "prov", u.k.values)
        u["ov"] = overtake(u, t)
        u["y_cw"], _ = event(u, t, "cw", kg)
        u["y_txt"], _ = event(u, t, "txt", u.k.values)
    else:  # 최종시험 설계용 단위(결과 계산 안 함)
        for c in ("y", "dshF", "ov", "y_cw", "y_txt"):
            u[c] = -1
    u["cw_g"] = u.mg.map(cwg)
    u["W"] = ((u.k.map(K2G) == u.cw_g) & u.k.isin(MULTI_K)).astype(int)  # 연계표 그룹은 맞지만 5자리 지목 불가
    parts = [u]
    for a, (lk, lf) in {"prov": (u.k.values, u.k.values), "cw": (kg, kg), "txt": (u.k.values, kg)}.items():
        f = share_feats(u, t, a, lk, lf)
        f.columns = [f"{a}__{x}" for x in f.columns]
        parts.append(f)
    u = pd.concat(parts, axis=1)
    rows.append(u)
    log(t, len(u), "pos", int((u.y == 1).sum()), "W", int(u.W.sum()))
P = pd.concat(rows, ignore_index=True)
for c in P.columns:
    if P[c].dtype == np.float64:
        P[c] = P[c].astype(np.float32)
P.to_parquet(p("cache", "p1_panel.parquet"), index=False)
d = P[P.t <= 2018]
st = d.groupby("t").agg(n=("y", "size"), pos=("y", "sum"), base=("y", "mean"), ov=("ov", "sum"),
                        pos_cw=("y_cw", "sum"), pos_txt=("y_txt", "sum"), W=("W", "sum")).reset_index()
st.to_csv(p("p1_panel_stats.csv"), index=False)
print(st.to_string())
log("end")
