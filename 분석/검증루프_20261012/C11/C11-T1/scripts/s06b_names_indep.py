# -*- coding: utf-8 -*-
"""(d2 보강) 독립 이름 원천: 등록 당시 권리자명(owner seq0)·제공 데이터 출원인명(kr_patents). 하위 층 정의·표본 추출·근거 수집.
출력 cache/s06b_pairs.parquet, cache/s06b_evidence.json, cache/s06b_out.json"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
from lib_ck import *
import pandas as pd, numpy as np
V = pd.read_parquet(os.path.join(CACHE, "s06_pairs.parquet")).set_index("cd")
ca = pd.read_parquet(os.path.join(CACHE, "ca.parquet")).set_index("cd")
A = pd.read_parquet(os.path.join(CACHE, "reg_app.parquet"))
r = ro(REG_DB)
napp = A.groupby("rgst_no").cd.transform("nunique")
solo = A[(napp == 1) & A.cd.isin(V.index)][["rgst_no", "cd", "ay"]]
own0 = pd.read_sql("select rgst_no, ownerName, ownerRgstCsDate from owner where seq=0", r)
o = solo.merge(own0, on="rgst_no")
om = o.groupby(["cd", "ownerName"]).agg(n=("rgst_no", "size"), y0=("ay", "min"), y1=("ay", "max")).reset_index().sort_values("n", ascending=False)
own_modal = om.drop_duplicates("cd").set_index("cd").ownerName
own_all = om.groupby("cd").apply(lambda g: list(zip(g.ownerName, g.n, g.y0, g.y1))[:5], include_groups=False)
K = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["등록번호", "출원인명"])
K = K[K["등록번호"].notna() & (K["등록번호"] != "")]
K["first_nm"] = K["출원인명"].astype(str).str.split(r"\s*[|;]\s*").str[0]
ks = solo.merge(K[["등록번호", "first_nm", "출원인명"]], left_on="rgst_no", right_on="등록번호")
km = ks.groupby(["cd", "first_nm"]).size().reset_index(name="n").sort_values("n", ascending=False)
prov_modal = km.drop_duplicates("cd").set_index("cd").first_nm
prov_all = km.groupby("cd").apply(lambda g: list(zip(g.first_nm, g.n))[:5], include_groups=False)
V["own0"] = own_modal.reindex(V.index); V["prov"] = prov_modal.reindex(V.index)
V["s_own0"] = [sim(a, b) if isinstance(b, str) else np.nan for a, b in zip(V.ca_name, V.own0)]
V["s_prov"] = [sim(a, b) if isinstance(b, str) else np.nan for a, b in zip(V.ca_name, V.prov)]
# 원천별 최댓값(어느 이름 하나라도) — 개명 이력 반영
V["s_own0_max"] = [max([sim(V.at[c, "ca_name"], x[0]) or 0 for x in own_all.get(c, [])] or [np.nan]) for c in V.index]
V["s_indep"] = V[["s_own0", "s_prov"]].max(axis=1, skipna=True)
O = {}
O["solo_codes_with_own0"] = int(V.own0.notna().sum()); O["solo_codes_with_prov"] = int(V.prov.notna().sum())
O["codes_no_indep_name"] = int(V.s_indep.isna().sum())
for k in ["s_own0", "s_prov", "s_indep", "s_own0_max"]:
    s = V[k]
    O[k] = dict(n=int(s.notna().sum()), lt05=int((s < 0.5).sum()), lt05_rate=float((s < 0.5).sum() / s.notna().sum()), lt03=int((s < 0.3).sum()), median=float(s.median()))
low = V[V.s_indep < 0.5]
O["low_codes"] = len(low); O["low_share_codes_of_scored"] = float(len(low) / V.s_indep.notna().sum()); O["low_share_codes_of_all"] = float(len(low) / len(V))
O["low_patents"] = int(low.n_pat.sum()); O["low_share_patents"] = float(low.n_pat.sum() / V.n_pat.sum())
O["low_with_biz"] = int((ca.biz.reindex(low.index) != "").sum())
rng = np.random.default_rng(20261013)
smp_low = list(low.index[rng.permutation(len(low))[:100]])
high = V[V.s_indep >= 0.5]
smp_high = list(high.index[rng.permutation(len(high))[:50]])
low1 = list(V.index[V.s1_modal < 0.5])
need = sorted(set(smp_low) | set(smp_high) | set(low1))
own = pd.read_sql("select rgst_no, seq, ownerName, ownerRgstCsDate, ownerRgstCsName from owner", r)
pats = A[A.cd.isin(need)][["cd", "rgst_no"]]
oo = pats.merge(own, on="rgst_no")
S = pd.read_parquet(os.path.join(EXT, "r3nts_status.parquet"))
S["biz"] = S.bizno.astype(str).str.replace(r"\D", "", regex=True)
st = S.drop_duplicates("biz").set_index("biz")[["b_stt", "end_dt"]]
need6 = set(ca.biz.reindex(need).str[:6]) - {""}
nps = {}
for f in sorted(glob.glob(os.path.join(EXT, "firm_nps_wkpl_*.parquet"))):
    w = pd.read_parquet(f, columns=["name", "biz6"])
    w = w[w.biz6.astype(str).isin(need6)]
    for b6, g in w.groupby(w.biz6.astype(str)):
        nps.setdefault(b6, set()).update(g.name.astype(str).tolist())
jcnt = ca[ca.jur != ""].groupby("jur").size()
ev = []
for c in need:
    v = V.loc[c]; jur = ca.at[c, "jur"]; biz = ca.at[c, "biz"]
    oc = oo[oo.cd == c]
    cand = sorted(nps.get(biz[:6], set()), key=lambda x: -max(sim(x, v.ca_name) or 0, sim(x, v.own0 if isinstance(v.own0, str) else "") or 0))[:3]
    strata = [s for s, L in [("low", smp_low), ("high", smp_high), ("low_regname", low1)] if c in L]
    ev.append(dict(strata=strata, cd=c, ca_name=v.ca_name, ca_ename=v.ca_ename, reg_name_current=v.reg_name, reg_ename=v.reg_ename,
                   own0_names=own_all.get(c, []), prov_names=prov_all.get(c, []), dart=v.dart_name,
                   s_own0=None if pd.isna(v.s_own0) else round(float(v.s_own0), 3), s_prov=None if pd.isna(v.s_prov) else round(float(v.s_prov), 3), s1=round(float(v.s1_modal), 3),
                   later_owners=oc[oc.seq > 0].groupby(["ownerName", "ownerRgstCsName"]).size().sort_values(ascending=False).head(4).reset_index().values.tolist(),
                   nts=(st.loc[biz].to_dict() if biz in st.index else None), nps_top=cand, jur_shared=int(jcnt.get(jur, 0)),
                   n_pat=int(v.n_pat), years=[int(v.first_ay), int(v.last_ay)], jkind=jur[4:6], bkind=biz[3:5], iy=c[1:5]))
V.reset_index().to_parquet(guard_path(os.path.join(CACHE, "s06b_pairs.parquet")))
json.dump(ev, open(guard_path(os.path.join(CACHE, "s06b_evidence.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=str)
O["n_evidence"] = len(ev); O["n_smp_low"] = len(smp_low); O["n_smp_high"] = len(smp_high); O["n_low_regname"] = len(low1)
json.dump(O, open(guard_path(os.path.join(CACHE, "s06b_out.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=0, default=str))
