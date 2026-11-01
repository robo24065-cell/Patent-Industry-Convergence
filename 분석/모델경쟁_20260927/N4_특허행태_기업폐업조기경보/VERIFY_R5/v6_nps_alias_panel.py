# -*- coding: utf-8 -*-
"""누수 영향 추정(N4): NPS 연결키의 '2026-07 상호' 사후 편향을 고친 패널.
사업장 정체성 = (사업자번호 앞6, 최초적용일 adpt_dt). 2026 상호 키 K로 어느 스냅샷에서든 잡힌 사업장의 정체성을 모은 뒤,
각 스냅샷 s에서 K가 없고 같은 정체성이 다른 상호(옛 상호)로 있으면 그 행을 K로 별칭 복사한다(= 원점 당시 상호로 연결했을 때와 같게).
동결 n4lib/n4feat 함수는 그대로 쓰고 NPS 표만 바꿔 넣는다. 결과: cache/final_panel_alias.parquet"""
import sys, os, json, time
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); sys.path.insert(0, TR)
import n4lib
n4lib.CACHE = os.path.join(HERE, "cache")
from n4lib import load_app_firm, load_registry, load_nts, load_nps, a2key, build_base, norm, EXT, log
from n4feat import load_ctx, build_all
import numpy as np, pandas as pd
MODE = "final"; T0S = list(range(2015, 2022)) + [2023]
A2K = a2key(); Kset = set(A2K.values())
ALL = ["201512", "201612", "201712", "201812", "201912", "202012", "202112", "202212", "202312", "202412", "202512", "202608"]
raw = {}
ident = []
for s in ALL:
    d = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{s}.parquet"), columns=["name", "biz6", "adpt_dt"])
    d["key"] = d.biz6.fillna("") + "|" + d.name.map(norm)
    d["ba"] = d.biz6.fillna("") + "|" + d.adpt_dt.fillna("").astype(str)
    raw[s] = d[["key", "ba"]]
    ident.append(d.loc[d.key.isin(Kset) & d.adpt_dt.notna(), ["ba", "key"]])
ident = pd.concat(ident).drop_duplicates()
nK = ident.groupby("ba").key.nunique()
ident = ident[ident.ba.map(nK) == 1].set_index("ba").key
log("identity pairs", len(ident))
NPS = load_nps(MODE)
stats = {}
for snap, agg in NPS.items():
    d = raw[snap.replace("-", "")]
    have = set(agg.key)
    c = d[d.ba.isin(ident.index)].copy()
    c["K"] = c.ba.map(ident)
    c = c[(c.key != c.K) & ~c.K.isin(have) & c.key.isin(have)]
    al = c.groupby(["K", "key"]).size().reset_index(name="cnt").sort_values("cnt", ascending=False).drop_duplicates("K")
    add = agg.set_index("key").loc[al.key].reset_index(drop=True)
    add["key"] = al.K.values
    NPS[snap] = pd.concat([agg, add], ignore_index=True)
    stats[snap] = int(len(al))
log("aliases per snapshot", stats)
AF = load_app_firm(); AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()]
R = load_registry(); ns = load_nts()
rows = [build_base(t0, MODE, AF, R, ns, NPS, A2K) for t0 in T0S]
D = pd.concat(rows, ignore_index=True)
ctx = load_ctx(MODE, set(D.applicantCd)); ctx["NPS"] = NPS
X = pd.concat([build_all(B, t0, ctx, AF, ns) for t0, B in D.groupby("t0")]).reindex(D.index)
D = pd.concat([D, X], axis=1)
D.to_parquet(os.path.join(HERE, "cache", "final_panel_alias.parquet"), index=False)
json.dump(stats, open(os.path.join(HERE, "v6_alias_stats.json"), "w"), indent=1)
log("panel", D.shape, {int(t): float(g.nps_present.mean()) for t, g in D.groupby("t0")})
