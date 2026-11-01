# -*- coding: utf-8 -*-
"""C12-T1 s3: (c) 연결 선택 편향 — 확장 키 검증(이름 일치·충돌), 연결률, 연결/비연결 SMD, 26429 밖 연결 비율·KSIC 분포·IPC 핵심 집합.
입력: cache/link_records.parquet, link_matches.parquet(s2). 출력: link_bias.csv(주 산출), cache/s3_out.json, cache/ipc_core.json"""
import os, re, sys, json, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import n5lib
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = n5lib.BASE
R = pd.read_parquet(os.path.join(W, "cache", "link_records.parquet"))
MB = pd.read_parquet(os.path.join(W, "cache", "link_matches.parquet"))
M = MB[MB.lvl <= 2].copy()   # 완전 키 = 엄격 ∪ 검증 통과 확장(lvl 2); lvl 3 = 1차 시도에서 기각된 형식
O = {}
rows = []  # link_bias.csv: section, item, group, value, note


def add(section, item, group, value, note=""):
    rows.append(dict(section=section, item=item, group=group, value=value, note=note))


# ---------------- 1. 확장 키 검증: 이름 일치(엄격 매칭에서 배운 기업↔출원인 쌍), 한 기록이 여러 출원에 붙는 충돌
M["apps"] = M.names.fillna("").str.split("|").map(lambda xs: [n5lib.norm(x) for x in xs if x])
M["app1"] = M.apps.str[0]
st = M[M.strict == 1].drop_duplicates(["rid", "appl_no"])
pair = st.explode("apps").groupby(["comp", "apps"]).size()
def known(r, minc):
    return any(pair.get((r.comp, a), 0) >= minc for a in r.apps)
full_only_appl = M.groupby(["rid", "appl_no"]).strict.max().reset_index()
fo = full_only_appl[full_only_appl.strict == 0][["rid", "appl_no"]].merge(M.drop_duplicates(["rid", "appl_no"]), on=["rid", "appl_no"])
st_s = st.sample(min(3000, len(st)), random_state=0)
agree_strict = float(np.mean([known(r, 3) for r in st_s.itertuples()]))   # 자기 자신 1건을 빼는 셈(≥3)
agree_full_only = float(np.mean([known(r, 2) for r in fo.itertuples()])) if len(fo) else np.nan
rnd = fo.copy(); rnd["apps"] = rnd.apps.sample(frac=1, random_state=1).values
agree_random = float(np.mean([known(r, 2) for r in rnd.itertuples()])) if len(fo) else np.nan
MB["apps"] = MB.names.fillna("").str.split("|").map(lambda xs: [n5lib.norm(x) for x in xs if x])
gb = MB.groupby(["rid", "appl_no"]).lvl.min().reset_index()
bo = gb[gb.lvl == 3][["rid", "appl_no"]].merge(MB.drop_duplicates(["rid", "appl_no"]), on=["rid", "appl_no"])
agree_broad_only = float(np.mean([known(r, 2) for r in bo.itertuples()])) if len(bo) else np.nan
add("c0_validation", "이름 일치율(기각 형식: Original_* 필드·KR 접두 없는 공보·실용신안 공보·10자리)", "broad_only(lvl3)", round(agree_broad_only, 4), f"쌍 {len(bo)}")
conf = M.groupby("rid").appl_no.nunique(); conf_rids = conf[conf > 1].index
O["validation"] = dict(full_only_pairs=int(len(fo)), full_only_appl=int(fo.appl_no.nunique()),
                       full_only_26429=int(fo.loc[fo.ksic == "26429", "appl_no"].nunique()),
                       name_agree_strict_sample=agree_strict, name_agree_full_only=agree_full_only, name_agree_full_only_shuffled=agree_random,
                       full_only_by_kt=fo.kt.value_counts().to_dict(),
                       conflict_records=int(len(conf_rids)),
                       conflict_strict_only_records=int((M[M.strict == 1].groupby("rid").appl_no.nunique() > 1).sum()),
                       conflict_examples=M[M.rid.isin(conf_rids[:5])][["rid", "kt", "key", "strict", "appl_no", "comp", "app1"]].to_dict("records"))
add("c0_validation", "확장 키로만 생긴 (기록,출원) 쌍", "full_only", len(fo))
add("c0_validation", "이름 일치율(엄격 매칭 표본, 쌍 빈도≥3)", "strict", round(agree_strict, 4))
add("c0_validation", "이름 일치율(확장 키로만 생긴 매칭)", "full_only", round(agree_full_only, 4))
add("c0_validation", "이름 일치율(확장 매칭 출원인 섞기 대조)", "shuffled", round(agree_random, 4))
add("c0_validation", "한 기록이 2개 이상 출원에 붙는 충돌", "all_keys", int(len(conf_rids)), "엄격 키만: %d" % O["validation"]["conflict_strict_only_records"])

# ---------------- 2. 연결률(c1): 기록 단위, 특허 단위
def ay_of(r):
    if pd.notna(r.appd):
        return int(str(int(r.appd))[:4])
    m = re.search(r"(19|20)\d{2}", re.sub(r"\D", "", str(r.PATT_APPLICATION_NUMBER))[:6])
    return int(m.group(0)) if m else np.nan
R["ay"] = [ay_of(r) for r in R.itertuples()]
R["link_s"] = R.rid.isin(set(M.loc[M.strict == 1, "rid"])).astype(int)
R["link_f"] = R.rid.isin(set(M.rid)).astype(int)
R["reg"] = R.PUBL_NUMBER.fillna("").str.contains(r"KR\s?10\d{7}\s*B|KR\s?0\d{6}\s*B|KR10\d{7}", regex=True).astype(int)
R["refl_y"] = (R.refl // 10000).astype(int)
R["inrange"] = R.ay.between(2000, 2025) & ~R.app_reason.isin(["utility", "foreign_number"])
for lab, msk in [("전체 기록", R.index == R.index), ("출원 2000–2025(제공 범위)·실용신안/외국번호 제외", R.inrange)]:
    d = R[msk]
    add("c1_rate", "기록 연결률 엄격", lab, round(d.link_s.mean(), 4), f"n={len(d)}")
    add("c1_rate", "기록 연결률 완전", lab, round(d.link_f.mean(), 4), f"n={len(d)}")
    dg = d.groupby("DIPG_ID")[["link_s", "link_f"]].max()
    add("c1_rate", "DIPG 연결률 엄격", lab, round(dg.link_s.mean(), 4), f"n={len(dg)}")
    add("c1_rate", "DIPG 연결률 완전", lab, round(dg.link_f.mean(), 4), f"n={len(dg)}")
for y0, y1 in [(2000, 2004), (2005, 2009), (2010, 2014), (2015, 2019), (2020, 2022), (2023, 2025)]:
    d = R[R.inrange & R.ay.between(y0, y1)]
    add("c1_rate_by_ay", "기록 연결률 완전", f"출원 {y0}–{y1}", round(d.link_f.mean(), 4), f"n={len(d)}; 엄격 {d.link_s.mean():.4f}")
for y0, y1 in [(2011, 2011), (2012, 2016), (2017, 2020), (2021, 2022), (2023, 2024), (2025, 2026)]:
    d = R[R.inrange & R.refl_y.between(y0, y1)]
    add("c1_rate_by_refl", "기록 연결률 완전", f"반영 {y0}–{y1}", round(d.link_f.mean(), 4), f"n={len(d)}")

# ---------------- 3. 연결 vs 비연결 SMD(c2), 제공 범위 기록만
d = R[R.inrange].copy()
topc = d.comp.value_counts().index[:8].tolist()
for c in topc:
    d["co_" + c[:25]] = (d.comp == c).astype(float)
d["ay_f"] = d.ay.astype(float); d["refl_yf"] = d.refl_y.astype(float); d["ndecl_log"] = np.log1p(d.n_decl)
vars_ = ["ay_f", "reg", "g5", "g4", "refl_yf", "ndecl_log"] + ["co_" + c[:25] for c in topc]
def smd(a, b):
    a = a.astype(float); b = b.astype(float)
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0
smd_tab = {}
for v in vars_:
    L1, L0 = d.loc[d.link_f == 1, v], d.loc[d.link_f == 0, v]
    smd_tab[v] = dict(linked_mean=float(L1.mean()), unlinked_mean=float(L0.mean()), smd=smd(L1, L0))
    add("c2_smd", v, "연결(완전) vs 비연결", round(smd_tab[v]["smd"], 3), f"연결 평균 {L1.mean():.3f}, 비연결 {L0.mean():.3f}, n {len(L1)}/{len(L0)}" + (" |SMD|≥0.2 치우침" if abs(smd_tab[v]["smd"]) >= 0.2 else ""))
# 등록된 것만(공보 번호가 있어 키가 확실한 경우)
dr = d[d.reg == 1]
for v in ["ay_f", "g5", "refl_yf"] + ["co_" + c[:25] for c in topc[:5]]:
    add("c2_smd_registered", v, "등록 기록 안 연결 vs 비연결", round(smd(dr.loc[dr.link_f == 1, v], dr.loc[dr.link_f == 0, v]), 3),
        f"연결률 {dr.link_f.mean():.3f}, n {len(dr)}")
O["smd"] = smd_tab
O["link_rate_by_company"] = d.groupby("comp").link_f.agg(["mean", "size"]).sort_values("size", ascending=False).head(12).round(4).to_dict("index")
for c, r in O["link_rate_by_company"].items():
    add("c2_company_rate", "기록 연결률 완전", c, r["mean"], f"n={int(r['size'])}")

# ---------------- 4. 26429 밖 연결(c3) + IPC 핵심 집합
k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "메인IPC", "세세분류코드", "출원연도"])
k.columns = ["appl_no", "ipc", "ksic", "ay"]; k["appl_no"] = k.appl_no.astype(str); k["ksic"] = k.ksic.astype(str)
k["sc"] = k.ipc.fillna("").str.replace(" ", "").str[:4]
s26 = k[(k.ksic == "26429") & (k.sc.str.len() == 4)].sc.value_counts(normalize=True)
core = s26.index[: int(np.searchsorted(s26.cumsum().values, 0.90) + 1)].tolist()
json.dump(dict(core=core, share=s26.loc[core].round(4).to_dict(), cum=float(s26.loc[core].sum())), open(os.path.join(W, "cache", "ipc_core.json"), "w"), indent=1)
add("c3_ipc_core", "26429 핵심 IPC 서브클래스(누적 90%)", ",".join(core), round(float(s26.loc[core].sum()), 4))
P = M.groupby("appl_no").agg(refl=("refl", "min"), strict=("strict", "max"), comp=("comp", "first")).reset_index().merge(k, on="appl_no")
P["core"] = P.sc.isin(core)
O["linked_patents"] = int(len(P))
for lab, msk in [("연결 특허 전체", P.index == P.index), ("최초 반영 2014–2024(N5 결과창 범위)", P.refl.between(20140101, 20241231)),
                 ("최초 반영 2023–2024(최종창)", P.refl.between(20230101, 20241231))]:
    q = P[msk]
    add("c3_out26429", "26429 밖 비율", lab, round(float((q.ksic != "26429").mean()), 4), f"n={len(q)}, 26429 밖 {int((q.ksic != '26429').sum())}")
    add("c3_out26429", "26429 밖 중 IPC 핵심 집합 비율", lab, round(float(q.loc[q.ksic != "26429", "core"].mean()), 4),
        f"26429 밖·핵심 {int(((q.ksic != '26429') & q.core).sum())}")
    add("c3_out26429", "26429 안 중 IPC 핵심 집합 비율", lab, round(float(q.loc[q.ksic == "26429", "core"].mean()), 4))
kd = P[P.ksic != "26429"].ksic.value_counts()
for ks, n in kd.head(12).items():
    add("c3_ksic_dist", "26429 밖 연결 특허 KSIC", ks, int(n), f"IPC 핵심 {int(P[(P.ksic == ks) & P.core].shape[0])}")
# 최종시험과 같은 정의의 '모집단 밖 양성': 출원 2017–22, 2022년까지 공개(공개·등록일은 s2 matches에서), 2022까지 미선언, 반영 2023–24
kk = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["출원번호", "공개일자", "등록일자"])
kk.columns = ["appl_no", "pub", "rgd"]; kk["appl_no"] = kk.appl_no.astype(str)
for c in ["pub", "rgd"]:
    kk[c] = pd.to_numeric(kk[c].astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
kk["known"] = kk[["pub", "rgd"]].min(axis=1)
Q = P.merge(kk[["appl_no", "known"]], on="appl_no")
fin = Q[Q.ay.between(2017, 2022) & (Q.known <= 20221231) & Q.refl.between(20230101, 20241231)]
O["final_like_pos"] = dict(total=int(len(fin)), in26429=int((fin.ksic == "26429").sum()), out=int((fin.ksic != "26429").sum()),
                           out_core=int(((fin.ksic != "26429") & fin.core).sum()), out_ksic=fin[fin.ksic != "26429"].ksic.value_counts().head(8).to_dict())
add("c3_final_like", "최종시험 정의 양성(출원 2017–22, 2022 공개, 반영 2023–24) 중 26429 밖", "전체 KSIC", O["final_like_pos"]["out"],
    f"전체 {len(fin)}, 26429 {O['final_like_pos']['in26429']}, 밖·IPC 핵심 {O['final_like_pos']['out_core']}")
# 26429 모집단의 IPC 핵심 밖 비율(공개 IPC 배정이었다면 빠졌을 몫)
k26 = k[k.ksic == "26429"]
add("c3_ipc_core", "26429 출원 중 IPC 핵심 밖 비율", "26429 전체", round(float((~k26.sc.isin(core)).mean()), 4), f"n={len(k26)}")
q26 = P[P.ksic == "26429"]
add("c3_ipc_core", "26429 연결(선언) 특허 중 IPC 핵심 밖 비율", "26429 선언", round(float((~q26.core).mean()), 4), f"n={len(q26)}")
# 같은 IPC 핵심 집합 안에서 KSIC별 선언률(t0 무관 전체 기간)
kc = k[k.sc.isin(core)].copy(); kc["dec"] = kc.appl_no.isin(set(P.appl_no))
dr_ = kc.groupby("ksic").dec.agg(["sum", "size", "mean"]).sort_values("sum", ascending=False).head(10)
for ks, r in dr_.iterrows():
    add("c3_core_rate", "IPC 핵심 집합 안 선언률", ks, round(float(r["mean"]), 4), f"선언 {int(r['sum'])}/{int(r['size'])}")
O["core_ksic_rate"] = dr_.round(4).to_dict("index")
pd.DataFrame(rows).to_csv(os.path.join(W, "link_bias.csv"), index=False, encoding="utf-8-sig")
json.dump(O, open(os.path.join(W, "cache", "s3_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(pd.DataFrame(rows).to_string())
print(json.dumps({k_: O[k_] for k_ in ["validation", "final_like_pos"]}, ensure_ascii=False, indent=1, default=str))
print("secs", round(time.time() - t0))
