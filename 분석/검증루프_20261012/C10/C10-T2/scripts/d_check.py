# -*- coding: utf-8 -*-
r"""[3] 파생 라벨 데이터 점검 (a)–(d) → res/data_check.json
(a) R12 res_prep.json 문서 일치(H-sg-3·H-sc-1·H-mg-3, dev 표≤2009/2010–14, eval 표≤2014/2015–19) 재현 — 표·hard() 구현 검증
(b) 분포: 결측·중복, 연도별 배정률, 라벨 종류 수, 라벨 분포 집중도
(c) 연결 편향: T1 풀 문서 중 rec_conc sg 연결/미연결 비교, rec_conc sc vs T1 메타 sc 일치
(d) 시점: 표 ≤ t, alt_KR m_free(KR=text_free, 패밀리 경로 없음), k_free(alt) = kr_txtfree"""
import sys
sys.dont_write_bytecode = True
from c10_common import *
from z_feat import load_docs, tables, hard, KSIC50, KI

R = {}
D0 = load_docs()
R["alt_KR_rows"] = len(D0); R["alt_KR_dup_appl"] = int(D0.appl_no.duplicated().sum())
R["alt_m_free_KR"] = D0.m_free.value_counts(dropna=False).to_dict()
R["k_free_alt_eq_t1"] = float((D0.k_free.fillna("NA") == D0.k_free_t1.fillna("NA")).mean())
R["kf_missing"] = float((D0.kf < 0).mean()); R["sg_missing"] = float(D0.sg.isna().mean())
kp = D0.ksic_prov.map(KI).fillna(-1).astype(int).values
# (a)
rep = {}
for tag, cut, lo, hi in (("dev", 2009, 2010, 2014), ("eval", 2014, 2015, 2019)):
    T = tables(D0, cut)
    m = (D0.appl_year >= lo) & (D0.appl_year <= hi)
    keys = D0.loc[m, ["sg", "mg", "sc4"]]
    for nm, (st, n) in {"H-sg-3": ("sg", 3), "H-sc-1": ("sc4", 1), "H-mg-3": ("mg", 3)}.items():
        lab = hard(T, keys, st, n)
        rep.setdefault(nm, {})[tag] = round(float((lab == kp[m.values]).mean()), 4)
        rep[nm][tag + "_assigned"] = round(float((lab >= 0).mean()), 4)
    rep.setdefault("T0", {})[tag] = round(float((D0.kf.values[m.values] == kp[m.values]).mean()), 4)
    rep["n_" + tag] = int(m.sum())
ref = json.load(open(os.path.join(MC, r"_shared_report\label_origin_R10\confirm_R12\res_prep.json"), encoding="utf-8"))["doc_agree_with_prov"]
R["a_R12_repro"] = {k: {"mine": rep[k], "R12": ref[k]} for k in ("T0", "H-sg-3", "H-sc-1", "H-mg-3")}
R["a_R12_repro_ok"] = all(abs(rep[k][s] - ref[k][s]) <= 0.0001 for k in ("T0", "H-sg-3", "H-sc-1", "H-mg-3") for s in ("dev", "eval"))
R["a_n"] = {"dev": rep["n_dev"], "eval": rep["n_eval"]}
# (c) T1 풀
P = pd.read_parquet(os.path.join(NC8, "cache", "patents.parquet"))
P = P.merge(D0[["appl_no", "sg", "mg", "sc4", "appl_year", "ksic_prov"]], on="appl_no", how="left")
R["pool_rows"] = len(P); R["pool_appl"] = int(P.appl_no.nunique())
R["pool_in_altKR"] = float(P.appl_year.notna().mean()); R["pool_sg_matched"] = float(P.sg.notna().mean())
R["pool_sc_eq_rec_sc"] = float((P.sc == P.sc4)[P.sg.notna()].mean())
R["pool_ksic_eq_altprov"] = float((P.ksic == P.ksic_prov)[P.ksic_prov.notna()].mean())
R["pool_ay_eq_altyear"] = float((P.ay == P.appl_year)[P.appl_year.notna()].mean())
mt = P.sg.notna()
R["c_matched_vs_not"] = {"n_matched": int(mt.sum()), "n_unmatched": int((~mt).sum()),
                         "ay_mean": [float(P.ay[mt].mean()), float(P.ay[~mt].mean()) if (~mt).any() else None]}
# (b) 원점별 배정률·라벨 수·집중도 (T1 풀 가시 문서)
by = {}
for t in (2002, 2005, 2008, 2010, 2012):
    T = tables(D0, t)
    V = P[P.ky <= t]
    keys = V[["sg", "mg", "sc4"]]
    row = {"vis_rows": len(V)}
    for nm, (st, n) in {"FIPC(H-sg-3)": ("sg", 3), "HSC1(H-sc-1)": ("sc4", 1), "HMG3(H-mg-3)": ("mg", 3)}.items():
        lab = hard(T, keys, st, n); a = lab[lab >= 0]
        sh = np.bincount(a, minlength=50) / max(len(a), 1)
        row[nm] = {"assigned": round(float((lab >= 0).mean()), 4), "n_labels": int((sh > 0).sum()),
                   "hhi": round(float((sh ** 2).sum()), 4), "agree_prov": round(float((lab == V.ksic.map(KI).fillna(-2).values).mean()), 4)}
    shp = V.ksic.map(KI).value_counts(normalize=True)
    row["prov_hhi"] = round(float((shp ** 2).sum()), 4); row["prov_n_labels"] = int(V.ksic.nunique())
    row["table_docs"] = int(((D0.appl_year <= t) & (D0.kf >= 0)).sum())
    by[t] = row
R["b_by_origin"] = by
R["b_altKR_year_counts"] = D0.appl_year.value_counts().sort_index().to_dict()
json.dump(R, open(os.path.join(RES, "data_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
log(json.dumps(R, ensure_ascii=False, default=str)[:4000])
