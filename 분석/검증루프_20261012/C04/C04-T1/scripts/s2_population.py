# -*- coding: utf-8 -*-
"""C04-T1 s2: N4 시험 원점(t0=2023) 모집단의 선택 편향(P2)·분포(b)·등록원부 대조(P3 일부)·휴업/합병형 폐업(P4).
레이블(2024–25 폐업)은 이미 채점된 동결 예측 파일의 y만 쓴다(새 결과 구간 계산 없음)."""
import os, re, json
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
N4 = os.path.join(MC, "N4_특허행태_기업폐업조기경보")
HERE = os.path.join(BASE, r"분석\검증루프_20261012\C04\C04-T1")
OUT = os.path.join(HERE, "cache")
T0 = 2023; END = T0 * 10000 + 1231
R = {}

def smd(a, b):
    a = pd.Series(a, dtype=float).dropna(); b = pd.Series(b, dtype=float).dropna()
    if len(a) < 2 or len(b) < 2: return np.nan
    s = np.sqrt((a.var() + b.var()) / 2)
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0

# ---------- 원자료 ----------
AF = pd.read_parquet(os.path.join(N4, r"cache\app_firm.parquet"))
AF = AF[(AF.applicantCd.astype(str).str[0] == "1") & AF.known_date.notna()]
pt = pd.read_parquet(os.path.join(N4, r"cache\reg_pt.parquet"))
lap = pd.read_parquet(os.path.join(N4, r"cache\reg_lapnp.parquet"))
own = pd.read_parquet(os.path.join(N4, r"cache\reg_own.parquet"))
py = pd.read_parquet(os.path.join(N4, r"cache\reg_py.parquet"), columns=["rgst_no", "d", "applicantCd"])
aps = pd.read_parquet(os.path.join(EXT, "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno", "b_stt_cd", "end_dt", "registered", "n_prov_rgst"])
import sqlite3
c = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'business_rights.sqlite')}?mode=ro", uri=True)
aq = pd.read_sql("select applicant_cd applicantCd, status, total_count from applicant_query", c)
rl_has = set(pd.read_sql("select distinct applicant_cd from right_list", c).applicant_cd)
c.close()

# ---------- t0=2023 후보 풀 (v5_pop_link 정의) ----------
kn = AF[AF.known_date <= END]
p = pt[pt.rgstDate <= END]
firms = sorted(set(kn.applicantCd) & set(p.applicantCd))
g = kn[kn.applicantCd.isin(firms)].groupby("applicantCd")
f = pd.DataFrame({"n_app": g.appl_no.nunique(), "first_ay": g.ay.min().astype(float), "last_known": g.known_date.max()})
f["yrs_since_last"] = T0 - f.last_known // 10000
pp = p[p.applicantCd.isin(firms)]
f["n_rgst"] = pp.groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0)
f["n_live"] = pp[~(pp.ext_d <= END)].groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0)
f["ext_share"] = 1 - f.n_live / f.n_rgst
oc = own[own.d <= END].merge(pp[["rgst_no", "applicantCd"]].drop_duplicates(), on=["rgst_no", "applicantCd"])
f["own_chg_share"] = oc.groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0) / f.n_rgst
# 사후 지표(감사용, 모형에 쓰지 않음): t0 살아있는 특허 중 2024-01-01~2025-08-22 등록료불납 소멸 비율, t0 이후 권리자 변경 비율
live = pp[~(pp.ext_d <= END)][["rgst_no", "applicantCd"]].drop_duplicates()
lp = lap[(lap.d > END)].merge(live, on=["rgst_no", "applicantCd"])
f["post_lapse_share"] = (lp.groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0) / f.n_live).where(f.n_live > 0)
op = own[own.d > END].merge(live, on=["rgst_no", "applicantCd"])
f["post_own_chg_share"] = (op.groupby("applicantCd").rgst_no.nunique().reindex(f.index).fillna(0) / f.n_live).where(f.n_live > 0)
for v in ["n_app", "n_rgst", "n_live"]:
    f["log_" + v] = np.log1p(f[v])
f["has_live"] = (f.n_live > 0).astype(float)
f["in_rightlist"] = f.index.isin(rl_has).astype(float)
aqs = aq.set_index("applicantCd").status
f["rl_query"] = f.index.map(aqs).fillna("not_queried")
# 국세청 상태
apsr = aps.drop_duplicates("applicantCd").set_index("applicantCd")
f["nts_grp"] = np.where(~f.index.isin(apsr.index), "no_bizno",
                np.where(f.index.map(apsr.registered).fillna(False).astype(bool), "registered", "unregistered"))
f["end"] = pd.to_numeric(f.index.map(apsr.end_dt).astype("datetime64[ns]").strftime("%Y%m%d"), errors="coerce")
f["b_stt_cd"] = f.index.map(apsr.b_stt_cd)
f["closed_by_t0"] = (f.end <= END)
R["pool_n"] = len(f); R["pool_grp"] = f.nts_grp.value_counts().to_dict()
R["pool_grp_excl_closed_by_t0"] = f[~f.closed_by_t0].nts_grp.value_counts().to_dict()

# ---------- 동결 예측과 대조 ----------
pr = pd.read_parquet(os.path.join(N4, "final_test_pred_r4.parquet"))
R["pred_n"] = len(pr); R["pred_pos"] = int(pr.y.sum())
inpool = pr.applicantCd.isin(f.index)
R["pred_in_pool"] = int(inpool.sum())
popset = set(pr.applicantCd)
reg_open = f[(f.nts_grp == "registered") & ~f.closed_by_t0]
R["registered_open_not_in_pred"] = int((~reg_open.index.isin(popset)).sum())
R["pred_not_registered_open"] = int((~pr.applicantCd.isin(reg_open.index)).sum())

# ---------- P2: 집단 비교 SMD ----------
VARS = ["log_n_app", "first_ay", "yrs_since_last", "log_n_rgst", "log_n_live", "has_live", "ext_share", "own_chg_share", "in_rightlist",
        "post_lapse_share", "post_own_chg_share"]
cand = f[~f.closed_by_t0]   # 비연결 쪽은 폐업 여부를 모르므로 t0 이전 폐업으로 알려진 연결 법인만 뺀다
tab = []
ref = cand[cand.nts_grp == "registered"]
for grp in ["no_bizno", "unregistered"]:
    oth = cand[cand.nts_grp == grp]
    for v in VARS:
        tab.append(dict(compare=f"NTS {grp} vs registered(open)", var=v, n_a=len(oth), n_b=len(ref), mean_a=float(oth[v].mean()), mean_b=float(ref[v].mean()), smd=smd(oth[v], ref[v])))
# 연결 쪽 안: 폐업(2024–25, 시험 y) vs 존속 — 사후 지표가 폐업을 얼마나 가리키는지 보정용
pm = pr.set_index("applicantCd")
tp = f.loc[f.index.intersection(pm.index)].copy(); tp["y"] = pm.y.reindex(tp.index); tp["nps_present"] = pm.nps_present.reindex(tp.index)
for v in VARS:
    tab.append(dict(compare="test y=1 vs y=0 (registered pop)", var=v, n_a=int(tp.y.sum()), n_b=int((tp.y == 0).sum()),
                    mean_a=float(tp.loc[tp.y == 1, v].mean()), mean_b=float(tp.loc[tp.y == 0, v].mean()), smd=smd(tp.loc[tp.y == 1, v], tp.loc[tp.y == 0, v])))
for v in VARS:
    a, b = tp[tp.nps_present == 0], tp[tp.nps_present == 1]
    tab.append(dict(compare="NPS unlinked vs linked as-of 2023-12 (test pop)", var=v, n_a=len(a), n_b=len(b), mean_a=float(a[v].mean()), mean_b=float(b[v].mean()), smd=smd(a[v], b[v])))
# RC:001(권리목록 조회 실패) — 국세청과 무관함을 보이기 위해 시험 모집단에서 폐업률
tp["rl_query"] = f.rl_query.reindex(tp.index)
R["test_rate_by_rl_query"] = tp.groupby("rl_query").y.agg(["size", "sum", "mean"]).round(4).to_dict("index")
R["RC001_note"] = "business_rights.sqlite applicant_query.status (KIPRIS 권리목록 조회) 값 — 국세청 r3nts에는 status 필드가 없음"
# 미연결 쪽 '숨은 폐업' 간접 추정: 살아있는 특허가 있는 법인에서 사후 불납소멸 비율 → 연결 쪽 로지스틱 보정
from sklearn.linear_model import LogisticRegression
Xv = ["post_lapse_share", "post_own_chg_share", "log_n_live", "yrs_since_last"]
tr = tp[tp.has_live == 1].dropna(subset=Xv)
lr = LogisticRegression(max_iter=2000).fit(tr[Xv], tr.y)
R["proxy_auc_in_registered"] = float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(tr.y, lr.predict_proba(tr[Xv])[:, 1]))
R["proxy_mean_registered_live"] = float(lr.predict_proba(tr[Xv])[:, 1].mean()); R["obs_rate_registered_live"] = float(tr.y.mean())
nb = cand[(cand.nts_grp == "no_bizno") & (cand.has_live == 1)].dropna(subset=Xv)
R["proxy_mean_nobizno_live"] = float(lr.predict_proba(nb[Xv])[:, 1].mean()); R["n_nobizno_live"] = len(nb)
R["n_nobizno_nolive"] = int(((cand.nts_grp == "no_bizno") & (cand.has_live == 0)).sum())
pd.DataFrame(tab).to_csv(os.path.join(OUT, "s2_smd.csv"), index=False, encoding="utf-8-sig")

# ---------- 분포 (b): 시험 모집단 ----------
ks = kn[kn.applicantCd.isin(tp.index)].copy(); ks["k2"] = ks.ksic.astype(str).str[:2]
main_k2 = ks.groupby(["applicantCd", "k2"]).size().reset_index().sort_values(0, ascending=False).drop_duplicates("applicantCd").set_index("applicantCd").k2
tp["k2"] = main_k2.reindex(tp.index).fillna("NA")
fpn = pd.read_parquet(os.path.join(N4, r"cache\final_panel.parquet"), columns=["applicantCd", "t0", "log_emp", "corp_type", "key"])
fpn = fpn[fpn.t0 == T0].set_index("applicantCd")
emp = np.expm1(fpn.log_emp.reindex(tp.index))
tp["sizeb"] = pd.cut(emp, [-1, 9, 49, 299, 1e9], labels=["1-9", "10-49", "50-299", "300+"]).astype(str).replace("nan", "NPS없음")
tp["corp_type"] = fpn.corp_type.reindex(tp.index)
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
norm = lambda s: re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUF.sub("", str(s or "")).upper())
w = pd.read_parquet(os.path.join(EXT, "firm_nps_wkpl_202312.parquet"), columns=["name", "biz6", "ldong", "stat", "n_emp"])
w = w[w.stat == "1"]; w["key"] = w.biz6.fillna("") + "|" + w.name.map(norm)
sido = w.sort_values("n_emp", ascending=False).drop_duplicates("key").set_index("key").ldong.astype(str).str[:2]
tp["sido"] = fpn.key.reindex(tp.index).map(sido).where(tp.nps_present == 1).fillna("NPS없음")
tp["bz1"] = pm.bizno.reindex(tp.index).str[:1]
dist = []
for col in ["k2", "sizeb", "sido", "corp_type", "bz1", "b_stt_cd"]:
    d = tp.groupby(col).y.agg(["size", "sum", "mean"]).reset_index().rename(columns={col: "level"}); d.insert(0, "dim", col); dist.append(d)
dist = pd.concat(dist); dist.to_csv(os.path.join(OUT, "s2_dist_testpop.csv"), index=False, encoding="utf-8-sig")
# 결측
R["test_missing"] = {c: int(tp[c].isna().sum()) for c in ["first_ay", "n_rgst", "end"]}
R["test_y_vs_end_consistency"] = dict(
    y1_end_in_window=int(((tp.y == 1) & tp.end.between(20240101, 20251231)).sum()),
    y1_total=int(tp.y.sum()),
    y0_end_in_window=int(((tp.y == 0) & tp.end.between(20240101, 20251231)).sum()),
    y0_closed_after_2025=int(((tp.y == 0) & (tp.end > 20251231)).sum()))

# ---------- 휴업·합병형 폐업·폐업 후 활동 ----------
R["test_hyu"] = dict(n=int((tp.b_stt_cd == "02").sum()), pos=int(tp.loc[tp.b_stt_cd == "02", "y"].sum()))
pos = tp[tp.y == 1].copy()
pos["endd"] = pd.to_datetime(pos.end.astype(int).astype(str), format="%Y%m%d")
mine = pt[pt.applicantCd.isin(pos.index)][["rgst_no", "applicantCd"]].drop_duplicates()
ow = own.merge(mine, on=["rgst_no", "applicantCd"])
ow["dd"] = pd.to_datetime(ow.d.astype(str), format="%Y%m%d", errors="coerce")
ow = ow.join(pos.endd, on="applicantCd")
ow["gap"] = (ow.dd - ow.endd).dt.days
mg = ow[(ow.kind == "merge") & ow.gap.between(-365, 365)].applicantCd.unique()
asg = ow[(ow.kind == "assign") & ow.gap.between(-365, 365)].applicantCd.unique()
R["pos_merge_pm1y"] = len(mg); R["pos_assign_pm1y"] = len(asg)
first_chg = ow.groupby(["rgst_no", "applicantCd"]).dd.min()
pyp = py[py.applicantCd.isin(pos.index)].copy()
pyp["dd"] = pd.to_datetime(pyp.d.astype(str), format="%Y%m%d", errors="coerce")
pyp = pyp.join(pos.endd, on="applicantCd")
pyp["fchg"] = pd.MultiIndex.from_frame(pyp[["rgst_no", "applicantCd"]]).map(first_chg)
post = pyp[(pyp.dd > pyp.endd + pd.Timedelta(days=30)) & ~(pyp.fchg <= pyp.dd)]
R["pos_pay_after_close30_no_owner_change"] = int(post.applicantCd.nunique())
R["pos_n"] = len(pos)
# 등록원부 관측 끝(2025-08-22)까지 납부를 볼 기회가 있는 양성(폐업 ≤ 2025-06-30)만 분모로
opp = pos[pos.endd <= "2025-06-30"]
R["pos_opp_n"] = len(opp); R["pos_opp_pay_after"] = int(post[post.applicantCd.isin(opp.index)].applicantCd.nunique())
# 비교: 존속(y=0) 법인 같은 기간 납부 비율(폐업일 대신 2024-01-01 기준)
R["pos_end_month"] = pos.endd.dt.to_period("M").astype(str).value_counts().sort_index().to_dict()
flag_post = set(post.applicantCd) | set(mg)
pd.Series(sorted(set(mg))).to_csv(os.path.join(OUT, "s2_pos_flag_merge.csv"), index=False, header=["applicantCd"])
pd.Series(sorted(flag_post)).to_csv(os.path.join(OUT, "s2_pos_flag_merge_or_postpay.csv"), index=False, header=["applicantCd"])
tp[["y", "nps_present", "b_stt_cd", "end", "k2", "sizeb", "sido", "corp_type", "nts_grp", "rl_query"]].to_parquet(os.path.join(OUT, "s2_testpop_attrs.parquet"))
json.dump(R, open(os.path.join(OUT, "s2_population.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str)[:8000])
print(pd.DataFrame(tab).round(3).to_string())
