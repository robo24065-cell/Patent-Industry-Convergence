# -*- coding: utf-8 -*-
r"""C06-T1 s3: 보정 재계산 (B1 IPW, B2 KSIC 차수, B3 업종 변경 제외) + 순환 재점검. PREREG.txt 규칙.
입력: cache/docs_U.parquet, firm_flags.parquet, firm_U.parquet, ksic_sets.json, R12 cache/reg12.parquet, label_origin_R10 cache/fam_reg.parquet, _shared\ext 연계표.
쓰기: rescore_B.csv, out/s3_*.json|csv"""
import os, sys, re, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); CACHE = os.path.join(HERE, "cache"); OUT = os.path.join(HERE, "out")
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"; EXT = os.path.join(MC, r"_shared\ext")
R12 = os.path.join(MC, r"_shared_report\label_origin_R10\confirm_R12"); LO = os.path.join(MC, r"_shared_report\label_origin_R10\cache")
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
S50 = set(KSIC50); S3 = {k[:3] for k in KSIC50}
KS5 = np.array(KSIC50 + [""], dtype=object); KS3 = np.array([k[:3] for k in KSIC50] + [""], dtype=object)
BEST = {"NPS5": "H-sc-1", "DART5": "H-mg-3", "KIND3": "C-30-20", "DART3": "D-100"}
CRIT = {"NPS5": ("nps", 5), "DART5": ("dart", 5), "KIND3": ("kind3", 3), "DART3": ("dart", 3)}
SEED = 20261012; NB = 1000
X = pd.read_parquet(os.path.join(CACHE, "docs_U.parquet")); F = pd.read_parquet(os.path.join(CACHE, "firm_flags.parquet"))
reg = pd.read_parquet(os.path.join(R12, "cache", "reg12.parquet"))
KS = json.load(open(os.path.join(CACHE, "ksic_sets.json"), encoding="utf-8")); V11 = set(KS["V11"]); V10 = set(KS["V10"]); m11to10 = KS["m11to10"]; m9to10 = KS["m9to10"]
J = {}

def build(crit, regcol=None):
    """기업×문서 표: r(등록 코드 집합), PROV·BEST 라벨 문자열"""
    col, dg = CRIT[crit]; Rg = reg[["applicantCd", col]].dropna().rename(columns={col: "r"}) if regcol is None else regcol
    Z = X.merge(Rg, on="applicantCd")
    KSx = KS5 if dg == 5 else KS3
    Z["lp"] = KSx[Z.PROV.values.astype(int)]; Z["lb"] = KSx[Z[BEST[crit]].values.astype(int)]
    return Z

def firm_table(Z, rset_col):
    """rset_col: 기업별 등록코드 집합(set)."""
    mp = np.array([lp in s for lp, s in zip(Z.lp.values, Z[rset_col].values)], dtype=float)
    mb = np.array([lb in s for lb, s in zip(Z.lb.values, Z[rset_col].values)], dtype=float)
    T = pd.DataFrame({"f": Z.applicantCd.values, "P": mp, "B": mb}).groupby("f").agg(P=("P", "mean"), B=("B", "mean"), n=("P", "size"))
    return T

def boot(T, w=None, seed=SEED):
    rng = np.random.default_rng(seed); M = T[["P", "B"]].values; nf = len(T)
    w = np.ones(nf) if w is None else np.asarray(w, float)
    v = (M * w[:, None]).sum(0) / w.sum()
    idx = rng.integers(0, nf, (NB, nf)); wb = w[idx]; B = (M[idx] * wb[:, :, None]).sum(1) / wb.sum(1)[:, None]
    d = B[:, 0] - B[:, 1]
    return {"기업": nf, "특허": int(T.n.sum()), "PROV": round(float(v[0]), 4), "BEST": round(float(v[1]), 4), "PROV-BEST": round(float(v[0] - v[1]), 4),
            "lo": round(float(np.percentile(d, 2.5)), 4), "hi": round(float(np.percentile(d, 97.5)), 4)}

def sample_sets(crit, Z, mode="base"):
    col, dg = CRIT[crit]
    if dg == 5:
        if mode == "base":
            Z["rs"] = Z.r.map(lambda c: {c} if c in S50 else set())
        elif mode == "B2":  # 11차 기반 판정(s2): 11차 코드면 연결된 10차 집합, 아니면 그대로
            Z["rs"] = Z.r.map(lambda c: set(m11to10.get(c, [c])) if (len(c) == 5 and c in V11) else ({c} if len(c) == 5 else set()))
        elif mode == "B2s":
            Z["rs"] = Z.r.map(lambda c: set(m11to10.get(c, [c])) if (len(c) == 5 and c in V11) else ({c} if len(c) == 5 else set()))
            Z["rs"] = Z.rs.map(lambda s: s if len(s) == 1 else set())
        Z = Z[Z.rs.map(lambda s: bool(s & S50))].copy()
    else:
        if mode == "base":
            Z["rs"] = Z.r.str[:3].map(lambda c: {c} if c in S3 else set())
        else:
            Z["rs"] = Z.r.str[:3].map(lambda c: set(M3_11to10.get(c, [c])) if c in V11_3 else {c})
            if mode == "B2s": Z["rs"] = Z.rs.map(lambda s: s if len(s) == 1 else set())
        Z = Z[Z.rs.map(lambda s: bool(s & S3))].copy()
    return Z

# 3자리 11→10 연계(level 3)
L = pd.read_csv(os.path.join(EXT, "r7_data_ksic_ksco_revision_link_raw.csv"), encoding="cp949", dtype=str)
L.columns = ["kind", "rev", "lvl", "new", "new_nm", "old", "old_nm"]
L3 = L[(L.kind == "KSIC") & (L.lvl == "3") & (L.rev == "11")].copy()
for c in ("new", "old"): L3[c] = L3[c].fillna("").str.replace(r"\D", "", regex=True)
V11_3 = set(L3.new) - {""}; V10_3 = set(L3.old) - {""}
M3_11to10 = L3[L3.new != ""].groupby("new").old.apply(lambda s: sorted(set(s) - {""})).to_dict()
J["L3_nonidentity_in_S3"] = {k: v for k, v in M3_11to10.items() if (k in S3 or set(v) & S3) and v != [k]}

# ---------------- B1 가중치
def ipw(firms):
    """firms: 표본 기업 목록. 칸 = size × kgrp, 목표 = U(F 전체)."""
    Fu = F[["applicantCd", "size", "kgrp"]].copy(); S = Fu[Fu.applicantCd.isin(set(firms))].copy()
    S["cell"] = S["size"] + "|" + S.kgrp; Fu["cell"] = Fu["size"] + "|" + Fu.kgrp
    # 1단계: 칸 < 5이면 같은 규모 안 KSIC 그룹 '기타'로 합침
    cnt = S.cell.value_counts(); small = set(cnt[cnt < 5].index) | (set(Fu.cell) - set(S.cell))
    def coll1(df):
        return np.where(df.cell.isin(small), df["size"] + "|합침", df.cell)
    S["cell"] = coll1(S); Fu["cell"] = coll1(Fu)
    cnt = S.cell.value_counts()
    order = ["<10", "10-49", "50-299", ">=300", "미상"]
    small2 = set(cnt[cnt < 5].index) | (set(Fu.cell) - set(S.cell))
    if small2:  # 2단계: 규모 인접 구간과 합침(합친 칸만)
        nb = {"<10": "10-49", "10-49": "<10", "50-299": ">=300", ">=300": "50-299", "미상": "미상"}
        def coll2(df):
            out = df.cell.copy()
            for c in small2:
                sz = c.split("|")[0]; tgt = nb[sz] + "|합침"
                out[df.cell == c] = tgt if not tgt.startswith("미상") else c
            return out
        S["cell"] = coll2(S); Fu["cell"] = coll2(Fu)
        S.loc[S.cell.isin(set(S.cell.value_counts()[lambda s: s < 5].index)), "cell"] = "전체합침"
        Fu.loc[~Fu.cell.isin(set(S.cell)), "cell"] = "전체합침"
    pu = Fu.cell.value_counts(normalize=True); ps = S.cell.value_counts(normalize=True)
    S["w"] = S.cell.map(pu / ps).fillna(0)
    lost = float(pu[~pu.index.isin(ps.index)].sum())
    w = S.w / S.w.mean(); w = w.clip(upper=20); w = w / w.mean()
    info = {"칸수": int(S.cell.nunique()), "목표 미대응 비율": round(lost, 4), "w_max": round(float(w.max()), 2), "ESS": round(float(w.sum() ** 2 / (w ** 2).sum()), 1)}
    return dict(zip(S.applicantCd, w)), info

rows = []
def add(crit, label, T, w=None, note=""):
    r = boot(T, None if w is None else [w[f] for f in T.index]); r.update({"기준": crit, "보정": label, "비고": note}); rows.append(r); return r

chg = set(F[F.chg_post.fillna(False)].applicantCd); chg_all = set(F[F.chg_all.fillna(False)].applicantCd)
npsknown = set(F[F.nps_k5_n_post.notna()].applicantCd); est19 = set(F[F.est_j.fillna("") > "20191231"].applicantCd)
for crit in ("DART5", "NPS5", "KIND3", "DART3"):
    Z0 = build(crit)
    Zb = sample_sets(crit, Z0.copy(), "base"); Tb = firm_table(Zb, "rs"); add(crit, "원값 재현", Tb)
    w, info = ipw(Tb.index); J[f"B1_{crit}"] = info; add(crit, "B1 IPW(규모×KSIC그룹→U)", Tb, w, json.dumps(info, ensure_ascii=False))
    Z2 = sample_sets(crit, Z0.copy(), "B2"); T2 = firm_table(Z2, "rs"); add(crit, "B2 KSIC 차수 정합(11→10)", T2, note=f"기업 변화 {len(T2)-len(Tb):+d}")
    Z2s = sample_sets(crit, Z0.copy(), "B2s"); T2s = firm_table(Z2s, "rs"); add(crit, "B2-strict(1:1만, 판정 외)", T2s)
    T3 = Tb[~Tb.index.isin(chg)]; add(crit, "B3 업종변경 의심 제외(NPS 2019-26)", T3, note=f"제외 {len(Tb)-len(T3)}")
    add(crit, "B3b NPS이력있음·변경없음만(판정 외)", Tb[Tb.index.isin(npsknown) & ~Tb.index.isin(chg)])
    add(crit, "B3c 변경 2015-26 기준(판정 외)", Tb[~Tb.index.isin(chg_all)])
    add(crit, "B3d DART 설립 2020+ 제외(판정 외)", Tb[~Tb.index.isin(est19)], note=f"제외 {int(Tb.index.isin(est19).sum())}")
    T13 = Tb[~Tb.index.isin(chg)]; w13, _ = ipw(T13.index); add(crit, "B1+B3(판정 외)", T13, w13)
    if crit == "DART5":
        # 상장/비상장 층(판정 외)
        lst = set(F[F.cls_j.isin(["Y", "K", "N"])].applicantCd)
        add(crit, "층: 상장(Y/K/N)(판정 외)", Tb[Tb.index.isin(lst)]); add(crit, "층: 기타 E(판정 외)", Tb[~Tb.index.isin(lst)])
        # (c) 이름 경로로만 연결되는 기업(판정 외)
        Fn = F[F.dart_b.isna() & F.dart_n.notna() & ~F.applicantCd.isin(set(reg.dropna(subset=["dart"]).applicantCd))][["applicantCd", "dart_n"]]
        Zn = build(crit, Fn.rename(columns={"dart_n": "r"})); Zn = sample_sets(crit, Zn, "base")
        if len(Zn): add(crit, "(c) 이름 경로로만 연결 기업(판정 외)", firm_table(Zn, "rs"))
        # 규모 미상 제외(판정 외)
        add(crit, "규모 미상 제외(판정 외)", Tb[Tb.index.isin(set(F[F["size"] != "미상"].applicantCd))])
RS = pd.DataFrame(rows)[["기준", "보정", "기업", "특허", "PROV", "BEST", "PROV-BEST", "lo", "hi", "비고"]]
RS.to_csv(os.path.join(HERE, "rescore_B.csv"), index=False, encoding="utf-8-sig")

# ---------------- 판정
base = 0.0731
dart = RS[(RS["기준"] == "DART5") & RS["보정"].str.match(r"^B[123] ")]
cond_fix = bool((dart.lo <= 0).any()); cond_shift = bool(((dart["PROV-BEST"] - base).abs() > 0.02).any())
J["verdict_DART5"] = "수정 필요" if cond_fix else ("단서 필요" if cond_shift else "영향 없음")
J["DART5_main"] = dart[["보정", "PROV-BEST", "lo", "hi"]].to_dict("records")
nps = RS[(RS["기준"] == "NPS5") & RS["보정"].str.match(r"^B[123] ")]
J["NPS5_reversal"] = nps.assign(rev=nps.hi < 0)[["보정", "PROV-BEST", "lo", "hi", "rev"]].to_dict("records")
kind = RS[(RS["기준"] == "KIND3") & RS["보정"].str.match(r"^B[123] ")]
J["KIND3_counter"] = kind.assign(counter=kind.hi < 0)[["보정", "PROV-BEST", "lo", "hi", "counter"]].to_dict("records")

# ---------------- 3) 순환 재점검: R12 DART5 표본에서 KR 라벨 vs 해외 패밀리 라벨
fam = pd.read_parquet(os.path.join(LO, "fam_reg.parquet")); fam["kr_appl_no"] = fam.kr_appl_no.astype(str)
pri = {"US": 0, "EP": 1, "JP": 2, "CN": 3}; fam["pr"] = fam.cc.map(pri)
c = fam.groupby(["kr_appl_no", "fk"]).agg(c=("fk", "size"), p=("pr", "min")).reset_index()
fmaj = c.sort_values(["kr_appl_no", "c", "p"], ascending=[True, False, True]).drop_duplicates("kr_appl_no").set_index("kr_appl_no").fk
circ = []
for crit in ("DART5", "NPS5"):
    Z = sample_sets(crit, build(crit), "base"); Z["fk"] = Z.appl_no.map(fmaj); Zf = Z[Z.fk.notna()].copy()
    Zf["mK"] = [a in s for a, s in zip(Zf.lp, Zf.rs)]; Zf["mF"] = [a in s for a, s in zip(Zf.fk, Zf.rs)]
    T = Zf.groupby("applicantCd").agg(P=("mK", "mean"), B=("mF", "mean"), n=("mK", "size"))
    dis = Zf[Zf.lp != Zf.fk]; dd = {"불일치문서": len(dis), "KR=등록": int(dis.mK.sum()), "해외=등록": int(dis.mF.sum())}
    r = boot(T); r.update({"기준": crit, "가중": "무가중", **dd}); circ.append(r)
    w, _ = ipw(T.index); r = boot(T, [w[f] for f in T.index]); r.update({"기준": crit, "가중": "B1", **dd}); circ.append(r)
CI = pd.DataFrame(circ).rename(columns={"PROV": "KR라벨", "BEST": "해외라벨", "PROV-BEST": "KR-해외"})
CI.to_csv(os.path.join(OUT, "s3_circ.csv"), index=False, encoding="utf-8-sig")
J["circ_DART5"] = {"성립": bool(all((CI[CI["기준"] == "DART5"].lo <= 0) & (CI[CI["기준"] == "DART5"].hi >= 0) & (CI[CI["기준"] == "DART5"]["KR-해외"].abs() <= 0.015))),
                   "순환우려": bool((CI[CI["기준"] == "DART5"].lo > 0).any())}
json.dump(J, open(os.path.join(OUT, "s3_meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("done", J["verdict_DART5"])
