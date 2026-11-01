# -*- coding: utf-8 -*-
r"""R7 §3 외부 타당성(기업 단위, 기술통계): 기업의 특허 산업 구성(제공 KSIC 라벨 vs 공식 IPC-KSIC 연계표 배정)이
기업의 '등록 업종'(① DART induty_code = KSIC, ② 국민연금 사업장 업종코드 → 50 KSIC 대응표)과 얼마나 맞는가.
- 대상: 국내법인 출원인(코드 1…), 제공 KR 출원 ≥3건, 등록 업종이 제공 50개 KSIC(또는 그 4자리 상위) 안에 있는 기업.
- 비교는 공정하게 연계표 단위(27개, KSIC 9차 기반)에서: 제공 라벨은 상위 단위로 올려 비교. 5자리 일치는 제공 라벨만 가능(연계표는 원리적으로 불가).
- 지표: (a) 주력 산업 일치율(기업 특허 최빈 산업 = 등록 업종), (b) 특허 비중 일치(기업 특허 중 등록 업종 단위에 속한 비율의 평균).
읽기 전용: N4 cache/app_firm.parquet, DART dart_company.sqlite·corp_applicant.parquet, _shared/ext NPS. 쓰기: 이 폴더."""
import os, sys, json, sqlite3
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE); MC = os.path.dirname(TR)
BASE = os.path.dirname(os.path.dirname(MC))
sys.path.insert(0, os.path.join(MC, "T4_특허포기예측", "ESSENTIAL_R7", "common"))
import numpy as np, pandas as pd
from xwalk import load_app_xw, K10_TO_XW
EXT = os.path.join(MC, "_shared", "ext")

AF = pd.read_parquet(os.path.join(TR, "cache", "app_firm.parquet"))
AF = AF[AF.applicantCd.astype(str).str[0] == "1"].drop_duplicates(["applicantCd", "appl_no"]).copy()
AF["xw"] = AF.appl_no.astype(str).map(load_app_xw().set_index("appl_no").xw)
AF["kpar"] = AF.ksic.astype(str).map(K10_TO_XW)
cnt = AF.groupby("applicantCd").appl_no.nunique()
AF = AF[AF.applicantCd.map(cnt) >= 3]
K4 = {}
for k, u in K10_TO_XW.items():
    K4.setdefault(k[:4], set()).add(u)
K4 = {k: list(v)[0] for k, v in K4.items() if len(v) == 1}


def truth_unit(code):
    code = str(code)
    if len(code) == 5 and code in K10_TO_XW: return K10_TO_XW[code]
    if len(code) >= 4 and code[:4] in K4: return K4[code[:4]]
    return None


def evaluate(truth, name):
    """truth: DataFrame(applicantCd, true5(5자리 또는 None), unit)"""
    t = truth.dropna(subset=["unit"]).drop_duplicates("applicantCd").set_index("applicantCd")
    A = AF[AF.applicantCd.isin(t.index)].copy()
    A["unit"] = A.applicantCd.map(t.unit); A["true5"] = A.applicantCd.map(t.true5)
    top = lambda col: A.dropna(subset=[col]).groupby("applicantCd")[col].agg(lambda s: s.value_counts().index[0])
    tp, tx, t5 = top("kpar"), top("xw"), top("ksic")
    firms = t.index.intersection(tp.index)
    r = dict(source=name, n_firms=int(len(firms)), n_patents=int(A.applicantCd.isin(firms).sum()))
    r["top_unit_match_provided"] = float((tp.reindex(firms) == t.unit.reindex(firms)).mean())
    r["top_unit_match_crosswalk"] = float((tx.reindex(firms) == t.unit.reindex(firms)).mean())
    A2 = A[A.applicantCd.isin(firms)]
    r["share_in_unit_provided"] = float((A2.kpar == A2.unit).groupby(A2.applicantCd).mean().mean())
    r["share_in_unit_crosswalk"] = float((A2.xw == A2.unit).groupby(A2.applicantCd).mean().mean())
    f5 = t.index[t.true5.notna()].intersection(t5.index)
    r["n_firms_5digit_truth"] = int(len(f5))
    r["top_5digit_match_provided"] = float((t5.reindex(f5).astype(str) == t.true5.reindex(f5).astype(str)).mean()) if len(f5) else None
    r["top_5digit_match_crosswalk"] = "불가(연계표에 5자리 없음)"
    # 부트스트랩(기업 재표집 1000회) — 제공−연계표 차이
    rng = np.random.default_rng(0)
    d = ((tp.reindex(firms) == t.unit.reindex(firms)).astype(float) - (tx.reindex(firms) == t.unit.reindex(firms)).astype(float)).values
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(1000)]
    r["top_unit_diff_ci95"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    return r


out = []
# ① DART
c = sqlite3.connect(f"file:{os.path.join(BASE, '공모전데이터', '외부데이터', 'DART', 'dart_company.sqlite')}?mode=ro", uri=True)
dc = pd.read_sql("select jurir_no, induty_code from company where induty_code is not null and induty_code != ''", c)
ca = pd.read_parquet(os.path.join(BASE, "공모전데이터", "외부데이터", "DART", "corp_applicant.parquet"), columns=["출원인코드", "jurir"])
ca.columns = ["applicantCd", "jurir"]
m = ca.merge(dc, left_on="jurir", right_on="jurir_no")
m["unit"] = m.induty_code.map(truth_unit); m["true5"] = np.where(m.induty_code.astype(str).str.len() == 5, m.induty_code.astype(str), None)
m.loc[~m.true5.isin(list(K10_TO_XW)), "true5"] = None
out.append(evaluate(m[["applicantCd", "true5", "unit"]], "DART induty_code(KSIC)"))
# ② 국민연금(2024-12 스냅샷 활성 사업장 중 최대 가입자 사업장의 업종)
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet")); lk = lk[lk.nps_hit]
lk["key"] = lk.biz6 + "|" + lk.nname
mp = pd.read_csv(os.path.join(EXT, "firm_nps_indcd_ksic50_map.csv"), dtype=str)
import re
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
norm = lambda s: re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUF.sub("", str(s or "")).upper())
w = pd.read_parquet(os.path.join(EXT, "firm_nps_wkpl_202412.parquet"), columns=["name", "biz6", "indcd", "n_emp", "stat"])
w = w[w.stat == "1"]; w["key"] = w.biz6.fillna("") + "|" + w.name.map(norm)
w = w.sort_values("n_emp", ascending=False).drop_duplicates("key")
w["ks"] = w.indcd.map(dict(zip(mp.indcd, mp.ksic)))
n = lk.merge(w[["key", "ks"]], on="key")
n["unit"] = n.ks.map(lambda k: K10_TO_XW.get(k) if isinstance(k, str) else None); n["true5"] = n.ks
out.append(evaluate(n[["applicantCd", "true5", "unit"]], "국민연금 업종코드(2024-12, 50 KSIC 대응표)"))
json.dump(out, open(os.path.join(HERE, "ext_validity_firm.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
