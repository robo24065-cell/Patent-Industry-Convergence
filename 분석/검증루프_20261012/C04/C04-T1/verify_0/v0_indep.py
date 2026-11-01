# -*- coding: utf-8 -*-
"""C04-T1 verify_0: 주 스크립트(s1~s3) 코드·캐시를 쓰지 않는 독립 재계산.
(i) RC:001 출처·편향 비, 미등록 수 (ii) 공식 추세 비율 (iii) 대표 시나리오 Δskill(sklearn, 등록원부 sqlite에서 합병 표지 직접 재구성) (iv) 공식 재현."""
import os, re, json, sqlite3
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
N4 = os.path.join(MC, "N4_특허행태_기업폐업조기경보")
HERE = os.path.dirname(os.path.abspath(__file__))
DL = os.path.join(os.path.dirname(HERE), "download")
V = {}

pr = pd.read_parquet(os.path.join(N4, "final_test_pred_r4.parquet"))
def sk(d, w=None):
    a1 = roc_auc_score(d.y, d.p_champ, sample_weight=w); a0 = roc_auc_score(d.y, d.p_M0, sample_weight=w)
    return a1, a0, (a1 - a0) / (1 - a0)
a1, a0, s0 = sk(pr)
V["iv_official"] = dict(n=len(pr), pos=int(pr.y.sum()), auc=a1, m0=a0, skill=s0)

# (i) RC:001 — business_rights applicant_query 상태. 국세청 원응답(jsonl)에 RC 코드가 있는지
raw = open(os.path.join(EXT, r"r3nts_work\nts_status_raw.jsonl"), encoding="utf-8").read()
V["i_rc001_in_nts_raw"] = raw.count("RC:001")
V["i_nts_raw_unregistered_msg"] = raw.count("등록되지 않은")
c = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'business_rights.sqlite')}?mode=ro", uri=True)
q = pd.read_sql("select applicant_cd, status from applicant_query", c); c.close()
st = pr.merge(q, left_on="applicantCd", right_on="applicant_cd", how="left")
st["status"] = st.status.fillna("none")
g = st.groupby("status").y.agg(["size", "mean"])
V["i_test_rate_by_rl_status"] = g.round(5).to_dict("index")
V["i_rate_ratio_RC001_vs_OK"] = float(g.loc["RC:001", "mean"] / g.loc["OK", "mean"])
aps = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno", "registered", "b_stt_cd", "end_dt", "n_prov_rgst"])
V["i_unregistered_bizno_total"] = int((~aps.drop_duplicates("bizno").registered).sum())
V["i_unregistered_with_prov_rgst"] = int(((~aps.registered) & (aps.n_prov_rgst > 0)).sum())
lr = pd.read_csv(os.path.join(N4, "rl_leak_check.csv"))
q21 = lr[(lr.t0 == 2021) & (lr.part == "query_status")]
V["i_dev2021_rc001"] = dict(pos=int(q21[(q21.y == 1) & (q21.grp == "RC:001")].n.iloc[0]), pos_all=int(q21[q21.y == 1].n.sum()),
                            neg=int(q21[(q21.y == 0) & (q21.grp == "RC:001")].n.iloc[0]), neg_all=int(q21[q21.y == 0].n.sum()))

# (ii) 공식 추세 — 다른 파싱(행 단위 td 추출)
t = open(os.path.join(DL, "stbl_282501_2001_2025.html"), encoding="utf-8").read()
off = {}
for tr in re.findall(r"<tr[^>]*>(.*?)</tr>", t, flags=re.S):
    cells = [re.sub(r"<[^>]+>|\s|,", "", x) for x in re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", tr, flags=re.S)]
    if cells and re.fullmatch(r"20\d\d", cells[0] or "") and len(cells) >= 16:
        off[int(cells[0])] = float(cells[6]) * 1000   # 법인 폐업 = 7번째 칸(총계3 + 법인 총계·신규·폐업)
V["ii_official_parsed_years"] = sorted(off)
st2 = aps.drop_duplicates("bizno")
k2 = pd.to_numeric(st2.bizno.str[3:5], errors="coerce")
cl = st2[(st2.b_stt_cd == "03") & k2.isin([81, 85, 86, 87, 88])]
cy = cl.end_dt.dt.year.value_counts()
r = {y: float(cy.get(y, 0) / off[y]) for y in range(2013, 2026) if y in off}
mu = np.mean([r[y] for y in range(2016, 2024)]); cv = np.std([r[y] for y in range(2016, 2024)], ddof=1) / mu
V["ii_ratio"] = r; V["ii_mean_2016_23"] = mu; V["ii_cv"] = cv
V["ii_r2024_over_mean"] = r[2024] / mu; V["ii_r2025_over_mean"] = r[2025] / mu
# 월별 H2/H1
m = cl.end_dt.dt.to_period("M").value_counts()
h = {}
for y in range(2017, 2026):
    h1 = sum(m.get(pd.Period(f"{y}-{i:02d}"), 0) for i in range(1, 7)); h2 = sum(m.get(pd.Period(f"{y}-{i:02d}"), 0) for i in range(7, 13))
    h[y] = h2 / h1
V["ii_h2h1"] = h; V["ii_deficit_2025H2"] = 1 - h[2025] / np.median([h[y] for y in range(2017, 2025)])

# (iii) 대표 시나리오
d = pr.merge(aps.drop_duplicates("applicantCd")[["applicantCd", "b_stt_cd", "end_dt"]], on="applicantCd", how="left")
hy = d.b_stt_cd == "02"
V["iii_hyu_n"] = int(hy.sum()); V["iii_hyu_pos"] = int(d.y[hy].sum())
V["iii_S3a_dskill"] = sk(d[~hy])[2] - s0
defi = V["ii_deficit_2025H2"]
w = np.where((d.y == 1) & (d.end_dt >= "2025-07-01"), 1 / (1 - defi), 1.0)
V["iii_S2_dskill"] = sk(d, w)[2] - s0
# 합병 표지: 등록원부 sqlite에서 직접
c = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'registry_full.sqlite')}?mode=ro", uri=True)
posc = tuple(d.loc[d.y == 1, "applicantCd"].astype(str))
ap = pd.read_sql("select distinct rgst_no, applicantCd from applicant", c)
ap = ap[ap.applicantCd.isin(posc)]
ow = pd.read_sql("select rgst_no, seq, ownerRgstCsDate, ownerRgstCsName from owner where seq > 0", c); c.close()
ow = ow[ow.ownerRgstCsName.fillna("").str.contains("합병|분할")].merge(ap, on="rgst_no")
ow["od"] = pd.to_datetime(ow.ownerRgstCsDate.astype(str).str.replace(r"\D", "", regex=True).str[:8], format="%Y%m%d", errors="coerce")
ow = ow.merge(d[["applicantCd", "end_dt"]], on="applicantCd")
mgv = set(ow[(ow.od - ow.end_dt).dt.days.abs() <= 365].applicantCd)
V["iii_merge_pos_n"] = len(mgv)
V["iii_S3bm_dskill"] = sk(d[~d.applicantCd.isin(mgv)])[2] - s0
V["iii_nps_sub"] = {k: list(sk(d[d.nps_present == k])) for k in [0.0, 1.0]}
json.dump(V, open(os.path.join(HERE, "v0_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(V, ensure_ascii=False, indent=1, default=str))
