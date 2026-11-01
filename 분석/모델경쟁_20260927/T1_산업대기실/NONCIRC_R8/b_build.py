# -*- coding: utf-8 -*-
r"""(2) 외부 결과 B: 기업의 국민연금 등록 업종 변화 — 특허 포트폴리오의 KSIC 구성(제공 vs 연계표)이 업종 전환을 예측하는가.
- 연결: PROTOCOL_R6 §2(사업자번호 앞6 + 정규화 상호 키, 사업장 정체성 biz6|최초적용일로 원점 당시 옛 상호 별칭). label_validity/s2_firm_agree.py와 같은 방식.
- 원점 o ∈ 2015..2021: 스냅샷 s0 = o-12, s1 = (o+4)-12. 기업키별 s0 최대 가입자 사업장(주사업장)을 고정하고, **같은 사업장(biz6|적용일)** 의 s1 업종코드와 비교.
- 결과(라벨 무관, 외부): y_chg = 국세청 6자리 업종코드 변경, y_maj = 앞 2자리 변경, 그룹 이동 = s1 업종의 연계표 그룹(27, 업종명→KSIC10→접두어 규칙, 두 체계 공통) ≠ s0 그룹.
- 피처: 라벨 무관 기반(특허 규모·최근성·IPC 다양성·신규 서브클래스·고용) + 라벨 체계 L∈{P(제공 라벨의 연계표 그룹 27), C(공식 연계표 배정 27)}의 포트폴리오 구성(등록업종 그룹 몫·최근 몫·최빈≠등록·엔트로피·비등록 최대 몫·최근 신규 그룹 몫) + P5(제공 5자리=등록 5자리 몫, 제공만 가능).
- 특허: 제공범위 KR 등록특허, 법인 출원인, 가시 연도 ky ≤ o.
- 주의: NPS 파일은 2015-11 이후만 있다 → 모든 원점이 T1 최종시험 구간(진입 2015–2019) 이후·겹침 → **사후 분석**으로 표기.
사용: python b_build.py → cache/B_o<o>.parquet
"""
from nc_common import *
import warnings; warnings.filterwarnings("ignore")
import glob
sys.path.insert(0, os.path.join(MC, r"_shared_report\label_validity"))
from lv_common import ksic_to_group, SPACE

SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


def nname(s):
    return re.sub(r"[\s·ㆍ,\.\(\)]", "", str(s or ""))


nm2code = {}
for f in sorted(glob.glob(os.path.join(BASE, r"공모전데이터\외부데이터\산업분류별\*.csv"))):
    d = pd.read_csv(f, encoding="cp949", dtype=str, usecols=[2, 3]); d.columns = ["code", "name"]
    d = d[d.code.str.match(r"^[A-Z]\d{5}$", na=False)]
    for c, n in zip(d.code, d.name):
        nm2code.setdefault(nname(n), c[1:])
for r in json.load(open(os.path.join(BASE, r"공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json"), encoding="utf-8")):
    nm2code.setdefault(nname(r["codenmKr"]), r["code"])
log("KSIC 이름 사전", len(nm2code))

P = load_patents()
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet"))
lk = lk[lk.nps_hit]; lk["key"] = lk.biz6 + "|" + lk.nname
A2K = dict(zip(lk.applicantCd, lk.key)); Kset = set(A2K.values()); Kbiz = set(k.split("|")[0] for k in Kset)
ALL = ["201512", "201612", "201712", "201812", "201912", "202012", "202112", "202212", "202312", "202412", "202512", "202608"]
SN = {}
ident = []
for s in ALL:
    d = pd.read_parquet(os.path.join(EXT, f"firm_nps_wkpl_{s}.parquet"), columns=["name", "biz6", "adpt_dt", "indcd", "indnm", "n_emp", "stat"])
    d["key"] = d.biz6.fillna("") + "|" + d.name.map(norm)
    d["ba"] = d.biz6.fillna("") + "|" + d.adpt_dt.fillna("").astype(str)
    d["stat"] = d.stat.astype(str)
    d = d[d.biz6.isin(Kbiz)].copy()  # 메모리: 연결 후보와 사업자번호 앞6이 같은 사업장만(별칭·같은 사업장 추적에 충분)
    ident.append(d.loc[d.key.isin(Kset) & d.adpt_dt.notna(), ["ba", "key"]].drop_duplicates())
    SN[s] = d
ident = pd.concat(ident).drop_duplicates()
nK = ident.groupby("ba").key.nunique()
ident = ident[ident.ba.map(nK) == 1].set_index("ba").key
log("NPS 스냅샷 적재, 정체성 ba", len(ident))


def snap_keyed(s):
    W = SN[s]; W = W[W.stat == "1"]
    have = set(W.key)
    al = W[W.ba.isin(ident.index)].copy(); al["K"] = al.ba.map(ident)
    al = al[(al.key != al.K) & ~al.K.isin(have)]; al["key"] = al.K
    W2 = pd.concat([W, al.drop(columns="K")], ignore_index=True)
    W2 = W2[W2.key.isin(Kset)]
    return W2.sort_values("n_emp", ascending=False).drop_duplicates("key"), len(al)


def grp_of(indnm):
    c = nm2code.get(nname(indnm))
    return c, (ksic_to_group(c, include_other=True) if c else None)


def build(o):
    s0, s1 = f"{o}12", f"{o + 4}12"
    W0, nal = snap_keyed(s0)
    # s1: 같은 기업키(원점 이후 상호 변경은 별칭으로 흡수) 활성 사업장 전체. 우선 같은 사업장(ba) → 없으면 기업키 주사업장.
    W1 = SN[s1]; W1 = W1[W1.stat == "1"]
    have1 = set(W1.key)
    al1 = W1[W1.ba.isin(ident.index)].copy(); al1["K"] = al1.ba.map(ident)
    al1 = al1[(al1.key != al1.K) & ~al1.K.isin(have1)]; al1["key"] = al1.K
    W1 = pd.concat([W1, al1.drop(columns="K")], ignore_index=True)
    W1 = W1[W1.key.isin(Kset)].sort_values("n_emp", ascending=False)
    W1same = W1.drop_duplicates(["key", "ba"]).set_index(["key", "ba"])
    W1main = W1.drop_duplicates("key").set_index("key")
    V = P[P.ky <= o]
    nv = V.groupby("applicantCd").size()
    firms = pd.DataFrame({"applicantCd": nv[nv >= 2].index})
    firms["key"] = firms.applicantCd.map(A2K)
    firms = firms[firms.key.notna()]
    firms["nv"] = firms.applicantCd.map(nv); firms = firms.sort_values("nv", ascending=False).drop_duplicates("key")
    F = firms.merge(W0[["key", "ba", "indcd", "indnm", "n_emp"]], on="key")
    n_linked = len(F)
    ix = pd.MultiIndex.from_arrays([F.key, F.ba])
    same = ix.isin(W1same.index)
    F["same_wkpl"] = same.astype(np.int8)
    F["in_s1"] = same | F.key.isin(W1main.index)
    stat = {"o": o, "s0": s0, "s1": s1, "alias_rows": nal, "firms_ge2": int(len(firms)), "linked_s0": n_linked,
            "same_wkpl_in_s1": int(same.sum()), "key_in_s1": int(F.in_s1.sum())}
    F = F[F.in_s1].copy(); same = F.same_wkpl.values == 1
    ix = pd.MultiIndex.from_arrays([F.key, F.ba])
    for c_new, c in (("indcd1", "indcd"), ("indnm1", "indnm"), ("n_emp1", "n_emp")):
        a = W1same[c].reindex(ix).values; b = F.key.map(W1main[c]).values
        F[c_new] = np.where(same, a, b)
    F = F[F.indcd.notna() & F.indcd1.notna()]
    F["y_chg"] = (F.indcd.astype(str) != F.indcd1.astype(str)).astype(np.int8)
    F["y_maj"] = (F.indcd.astype(str).str[:2] != F.indcd1.astype(str).str[:2]).astype(np.int8)
    z0 = F.indnm.map(grp_of); z1 = F.indnm1.map(grp_of)
    F["rc0"] = [a for a, b in z0]; F["rg0"] = [b for a, b in z0]
    F["rc1"] = [a for a, b in z1]; F["rg1"] = [b for a, b in z1]
    # ---- 라벨 무관 기반 피처
    Vf = V[V.applicantCd.isin(set(F.applicantCd))]
    g = Vf.groupby("applicantCd")
    F["f_n"] = np.log1p(F.applicantCd.map(g.size()).values)
    F["f_rec3"] = np.log1p(F.applicantCd.map(Vf[Vf.ay >= o - 2].groupby("applicantCd").size()).fillna(0).values)
    F["f_age"] = o - F.applicantCd.map(g.ay.min()).values
    F["f_last"] = o - F.applicantCd.map(g.ay.max()).values
    scn = Vf.groupby(["applicantCd", "sc"]).size().rename("k").reset_index()
    scn["p"] = scn.k / scn.groupby("applicantCd").k.transform("sum")
    F["f_nsc"] = F.applicantCd.map(scn.groupby("applicantCd").size()).values
    F["f_scent"] = F.applicantCd.map((-scn.p * np.log(scn.p)).groupby(scn.applicantCd).sum()).values
    fsy = Vf.groupby(["applicantCd", "sc"]).ay.min().reset_index()
    rec = Vf[Vf.ay >= o - 2].merge(fsy.rename(columns={"ay": "fy"}), on=["applicantCd", "sc"])
    F["f_newsc_rec"] = F.applicantCd.map((rec.fy >= o - 2).groupby(rec.applicantCd).mean()).fillna(0).values
    F["f_emp"] = np.log1p(F.n_emp.astype(float))
    F["f_rg_known"] = F.rg0.notna().astype(np.int8)
    F["f_rg_space"] = F.rg0.isin(SPACE).astype(np.int8)
    # ---- 라벨 체계별 포트폴리오 구성
    for L, col in (("P", "pgrp"), ("C", "cw")):
        Vl = Vf[Vf[col].notna()]
        cnt = Vl.groupby(["applicantCd", col]).size().rename("k").reset_index()
        tot = cnt.groupby("applicantCd").k.sum()
        cnt["p"] = cnt.k / cnt.applicantCd.map(tot)
        cntr = Vl[Vl.ay >= o - 2].groupby(["applicantCd", col]).size().rename("k").reset_index()
        totr = cntr.groupby("applicantCd").k.sum(); cntr["p"] = cntr.k / cntr.applicantCd.map(totr)
        key0 = F[["applicantCd", "rg0"]]
        m = key0.merge(cnt, left_on=["applicantCd", "rg0"], right_on=["applicantCd", col], how="left")
        F[f"{L}_sh_rg"] = m.p.fillna(0).values
        mr = key0.merge(cntr, left_on=["applicantCd", "rg0"], right_on=["applicantCd", col], how="left")
        F[f"{L}_sh_rg_rec"] = np.where(F.applicantCd.map(totr).notna(), mr.p.fillna(0).values, np.nan)
        top = cnt.sort_values(["k", col], ascending=[False, True]).drop_duplicates("applicantCd").set_index("applicantCd")[col]
        F[f"{L}_top_ne"] = (F.applicantCd.map(top) != F.rg0).astype(np.int8).where(F.applicantCd.map(top).notna(), np.nan)
        F[f"{L}_ent"] = F.applicantCd.map((-cnt.p * np.log(cnt.p)).groupby(cnt.applicantCd).sum()).values
        oth = cnt.merge(key0, on="applicantCd"); oth = oth[oth[col] != oth.rg0]
        F[f"{L}_maxoth"] = F.applicantCd.map(oth.groupby("applicantCd").p.max()).fillna(0).values
        fy = Vl.groupby(["applicantCd", col]).ay.min().rename("fy").reset_index()
        rr = Vl[Vl.ay >= o - 2].merge(fy, on=["applicantCd", col])
        F[f"{L}_newg_rec"] = F.applicantCd.map((rr.fy >= o - 2).groupby(rr.applicantCd).mean()).values
        F[f"{L}_cov"] = (F.applicantCd.map(tot).fillna(0) / F.applicantCd.map(g.size())).values
        # 그룹 이동 방향 시험용: 기업 × 27그룹 몫(전체·최근) 저장
        piv = cnt.pivot(index="applicantCd", columns=col, values="p").reindex(columns=SPACE).fillna(0)
        pivr = cntr.pivot(index="applicantCd", columns=col, values="p").reindex(columns=SPACE).fillna(0)
        for gname in SPACE:
            F[f"{L}g_{gname}"] = F.applicantCd.map(piv[gname]).fillna(0).values.astype(np.float32)
            F[f"{L}r_{gname}"] = F.applicantCd.map(pivr[gname]).fillna(0).values.astype(np.float32)
    # 제공 5자리(연계표로는 만들 수 없음)
    c5 = Vf.groupby(["applicantCd", "ksic"]).size().rename("k").reset_index()
    c5["p"] = c5.k / c5.groupby("applicantCd").k.transform("sum")
    m5 = F[["applicantCd", "rc0"]].merge(c5, left_on=["applicantCd", "rc0"], right_on=["applicantCd", "ksic"], how="left")
    F["P5_sh_rc"] = m5.p.fillna(0).values
    F["o"] = o
    stat.update({"n_final": int(len(F)), "chg": int(F.y_chg.sum()), "maj": int(F.y_maj.sum()),
                 "rg0_known": int(F.rg0.notna().sum()), "rg0_space": int(F.rg0.isin(SPACE).sum()),
                 "grp_move": int((F.rg0.notna() & F.rg1.notna() & (F.rg0 != F.rg1)).sum()),
                 "grp_move_into_space": int((F.rg0.notna() & F.rg1.isin(SPACE) & (F.rg0 != F.rg1)).sum())})
    F = F.drop(columns=["key"])
    F.to_parquet(os.path.join(CACHE, f"B_o{o}.parquet"), index=False)
    log(stat)
    return stat


if __name__ == "__main__":
    S = [build(o) for o in range(2015, 2022)]
    json.dump(S, open(os.path.join(OUT, "res", "B_build_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
