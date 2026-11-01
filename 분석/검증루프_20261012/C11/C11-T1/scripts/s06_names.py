# -*- coding: utf-8 -*-
"""(d2) 이름 유사도 3쌍, 하위 층 표본 100 + 상위 층 50 추출(seed 20261013), 판정 근거 수집.
출력 cache/s06_pairs.parquet, cache/s06_sample_evidence.json, cache/s06_out.json"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
from lib_ck import *
import pandas as pd, numpy as np
ca = pd.read_parquet(os.path.join(CACHE, "ca.parquet")).set_index("cd")
U = pd.read_parquet(os.path.join(CACHE, "U.parquet")).set_index("cd")
A = pd.read_parquet(os.path.join(CACHE, "reg_app.parquet"))
r = ro(REG_DB)
E = pd.read_sql("select rgst_no, applicantCd cd, applicantEngName en from applicant", r)
O = {}
V = U[U.in_ca & ((U.L == 1) | (U.Lj == 1))].copy()
O["denominator_codes"] = len(V); O["denominator_patents"] = int(V.n_pat.sum())
V["ca_name"] = ca.name.reindex(V.index); V["ca_ename"] = ca.ename.reindex(V.index)
V["jur"] = ca.jur.reindex(V.index); V["biz"] = ca.biz.reindex(V.index)
nm = A[A.cd.isin(V.index)].groupby(["cd", "nm"]).agg(n=("rgst_no", "size"), y0=("ay", "min"), y1=("ay", "max")).reset_index()
names_by = nm.groupby("cd").apply(lambda g: list(zip(g.nm, g.n, g.y0, g.y1)), include_groups=False)
V["s1_modal"] = [sim(a, b) for a, b in zip(V.ca_name, V.reg_name)]
V["s1_max"] = [max([sim(V.at[c, "ca_name"], x[0]) or 0 for x in names_by.get(c, [])] or [np.nan]) for c in V.index]
en = E[E.cd.isin(V.index)].dropna().groupby("cd").en.agg(lambda s: s.value_counts().index[0])
V["reg_ename"] = en.reindex(V.index)
V["s_en"] = [sim(a, b) for a, b in zip(V.ca_ename, V.reg_ename)]
dc = ro(DART_DB)
D = pd.read_sql("select corp_name, jurir_no from company", dc)
D["jur"] = D.jurir_no.fillna("").str.replace(r"\D", "", regex=True)
dn = D[D.jur.str.len() == 13].drop_duplicates("jur").set_index("jur").corp_name
V["dart_name"] = V.jur.map(dn)
V["s2"] = [sim(a, b) if isinstance(b, str) else np.nan for a, b in zip(V.ca_name, V.dart_name)]
V["s3"] = [sim(a, b) if isinstance(b, str) else np.nan for a, b in zip(V.reg_name, V.dart_name)]
for k in ["s1_modal", "s1_max", "s_en", "s2", "s3"]:
    s = V[k]
    O[k] = dict(n=int(s.notna().sum()), lt05=int((s < 0.5).sum()), lt05_rate=float((s < 0.5).mean() if s.notna().any() else np.nan),
                lt03=int((s < 0.3).sum()), median=float(s.median()), q05=float(s.quantile(0.05)))
low = V[V.s1_modal < 0.5]
O["low_codes"] = len(low); O["low_share_codes"] = len(low) / len(V)
O["low_patents"] = int(low.n_pat.sum()); O["low_share_patents"] = float(low.n_pat.sum() / V.n_pat.sum())
O["low_also_s1max_lt05"] = int((low.s1_max < 0.5).sum())
O["low_with_dart"] = int(low.dart_name.notna().sum())
O["low_dart_s2_ge05"] = int((low.s2 >= 0.5).sum()); O["low_dart_s3_ge05"] = int((low.s3 >= 0.5).sum())
rng = np.random.default_rng(20261013)
smp_low = low.index[rng.permutation(len(low))[:100]]
high = V[V.s1_modal >= 0.5]
smp_high = high.index[rng.permutation(len(high))[:50]]
# 근거: 최종 권리자 이름(그 코드 특허), NPS biz6 이름, 국세청 상태
own = pd.read_sql("select rgst_no, ownerName, ownerRgstCsDate, ownerRgstCsName, finalOwnerYn from owner", r)
pats = A[A.cd.isin(list(smp_low) + list(smp_high))][["cd", "rgst_no"]]
oo = pats.merge(own, on="rgst_no")
S = pd.read_parquet(os.path.join(EXT, "r3nts_status.parquet"))
S["biz"] = S.bizno.astype(str).str.replace(r"\D", "", regex=True)
st = S.set_index("biz")[["b_stt", "end_dt"]]
nps_names = {}
need6 = set(V.loc[list(smp_low) + list(smp_high), "biz"].str[:6]) - {""}
for f in sorted(glob.glob(os.path.join(EXT, "firm_nps_wkpl_*.parquet"))):
    w = pd.read_parquet(f, columns=["name", "biz6"])
    w = w[w.biz6.astype(str).isin(need6)]
    for b6, g in w.groupby(w.biz6.astype(str)):
        nps_names.setdefault(b6, set()).update(g.name.astype(str).tolist())
ev = []
for grp, idx in [("low", smp_low), ("high", smp_high)]:
    for c in idx:
        v = V.loc[c]
        o = oo[oo.cd == c]
        b6 = (v.biz or "")[:6]
        cand = sorted(nps_names.get(b6, set()), key=lambda x: -max(sim(x, v.ca_name) or 0, sim(x, v.reg_name) or 0))[:3]
        ev.append(dict(stratum=grp, cd=c, ca_name=v.ca_name, ca_ename=v.ca_ename, reg_names=[(a, int(b), int(y0) if y0 == y0 else None, int(y1) if y1 == y1 else None) for a, b, y0, y1 in names_by.get(c, [])],
                       reg_ename=v.reg_ename, dart_name=v.dart_name, s1_modal=round(float(v.s1_modal), 3), s1_max=round(float(v.s1_max), 3),
                       s_en=None if pd.isna(v.s_en) else round(float(v.s_en), 3), s2=None if pd.isna(v.s2) else round(float(v.s2), 3),
                       owners=o.groupby("ownerName").size().sort_values(ascending=False).head(4).to_dict(),
                       owner_events=o[o.ownerRgstCsName.notna()].groupby("ownerRgstCsName").size().to_dict(),
                       nts=(st.loc[v.biz].to_dict() if v.biz in st.index else None), nps_top=cand, n_pat=int(v.n_pat),
                       first_ay=int(v.first_ay), last_ay=int(v.last_ay), jkind=(v.jur or "")[4:6], biz_kind=(v.biz or "")[3:5]))
V.reset_index().drop(columns=["jur", "biz"]).to_parquet(guard_path(os.path.join(CACHE, "s06_pairs.parquet")))
json.dump(ev, open(guard_path(os.path.join(CACHE, "s06_sample_evidence.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=0, default=str)
json.dump(O, open(guard_path(os.path.join(CACHE, "s06_out.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=0, default=str))
