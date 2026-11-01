# -*- coding: utf-8 -*-
"""C07-T3 verify_0 v03a (반박 검증자, 별도 에이전트): 원천에서 다시 계산.
읽기: registry_full.sqlite(mode=ro), future_scores_r6.parquet, future_ksic_r6.csv, s02b 기준선 점수(p_M0, owner_v2)
쓰기: verify_0\v03a_out.json 만"""
import os, json, sqlite3, hashlib, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
t = time.time()
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N8_특허분쟁표적레이더"
res = {}
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
res["sha_future_scores"] = sha(os.path.join(H, "future_scores_r6.parquet"))
res["sha_future_ksic"] = sha(os.path.join(H, "future_ksic_r6.csv"))
FS = pd.read_parquet(os.path.join(H, "future_scores_r6.parquet"))
B = pd.read_parquet(os.path.join(W, "cache", "s02b_scores_t2024.parquet"), columns=["rgst_no", "p_M0", "p_SB", "p_champ", "owner_v2"])
B = B.set_index("rgst_no").loc[FS.rgst_no.values].reset_index()
res["n"] = len(FS); res["dup_rgst"] = int(FS.rgst_no.duplicated().sum())
res["champ_eq_s02b"] = bool(np.array_equal(FS.p.values, B.p_champ.values))
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
# 날짜 형식 점검(모든 분쟁 사건)
D = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName from right where rgstCsName like '%무효%' or rgstCsName like '%권리범위%'", c)
res["disp_name_counts_all"] = D.rgstCsName.value_counts().to_dict()
ds = D.rgstCsDate.astype(str)
res["date_bad_format"] = int((~ds.str.fullmatch(r"\d{8}")).sum())
res["date_bad_examples"] = ds[~ds.str.fullmatch(r"\d{8}")].head(10).tolist()
NAMES = ("무효(심판)예고등록", "권리범위(심판)예고등록")
E = D[D.rgstCsName.isin(NAMES)].copy()
E["d"] = pd.to_numeric(E.rgstCsDate, errors="coerce")
res["rgst_no_type"] = [str(type(FS.rgst_no.iloc[0])), str(type(E.rgst_no.iloc[0])), FS.rgst_no.iloc[0], E.rgst_no.iloc[0]]
# 등록번호 형식 일치 여부: 2024 이전 사건 특허 중 모집단 내 비율
pre = set(E[(E.d >= 20230101) & (E.d <= 20241231)].rgst_no)
res["pre2324_events_patents"] = len(pre); res["pre2324_in_pop"] = int(FS.rgst_no.isin(pre).sum())
def lab(lo, hi):
    return FS.rgst_no.isin(set(E[(E.d >= lo) & (E.d <= hi)].rgst_no)).values.astype(int)
y = lab(20250101, 20260630); yA = lab(20250101, 20260815)
res["pos_main"] = int(y.sum()); res["pos_auxA"] = int(yA.sum())
res["events_main_rows"] = int(((E.d >= 20250101) & (E.d <= 20260630)).sum())
res["max_event_date"] = int(E.d.max())
S = {"champ": FS.p.values.astype(float), "M0": B.p_M0.values.astype(float), "SB": B.p_SB.values.astype(float)}
def mets(y):
    o = {}
    for k, s in S.items():
        o[k + "_auc"] = float(roc_auc_score(y, s))
        srt = np.argsort(-s, kind="stable")
        for fr in (0.01, 0.05, 0.10):
            kk = int(round(fr * len(y))); o[f"{k}_top{int(fr*100)}_hits"] = int(y[srt[:kk]].sum())
        o[k + "_hit100"] = int(y[srt[:100]].sum())
    o["skill_M0"] = (o["champ_auc"] - o["M0_auc"]) / (1 - o["M0_auc"])
    o["skill_SB"] = (o["champ_auc"] - o["SB_auc"]) / (1 - o["SB_auc"])
    return o
res["main"] = mets(y); res["auxA"] = mets(yA)
# 군집 부트스트랩(독자 구현, 시드 9001, 2,000회) + 비군집(i.i.d.) 부트스트랩
ow = B.owner_v2.fillna("").astype(str).values
grp = np.where(ow == "", "R" + FS.rgst_no.astype(str).values, ow)
codes, uq = pd.factorize(grp); G = len(uq); res["G"] = G
inv = {k: np.unique(s, return_inverse=True)[1] for k, s in S.items()}
def wauc(k, w, yy):
    iv = inv[k]; m = iv.max() + 1
    pos = np.bincount(iv, weights=w * yy, minlength=m); neg = np.bincount(iv, weights=w * (1 - yy), minlength=m)
    cn = np.cumsum(neg) - neg
    return (pos * (cn + 0.5 * neg)).sum() / (pos.sum() * neg.sum())
def boot(yy, seed, n, cluster=True):
    rng = np.random.default_rng(seed); A, SM, SS = [], [], []
    for _ in range(n):
        if cluster:
            w = np.bincount(rng.integers(0, G, G), minlength=G)[codes].astype(float)
        else:
            w = np.bincount(rng.integers(0, len(yy), len(yy)), minlength=len(yy)).astype(float)
        a = wauc("champ", w, yy); b = wauc("M0", w, yy); s_ = wauc("SB", w, yy)
        A.append(a); SM.append((a - b) / (1 - b)); SS.append((a - s_) / (1 - s_))
    q = lambda v: [float(x) for x in np.percentile(v, [2.5, 97.5])]
    return dict(auc=q(A), skill_M0=q(SM), skill_SB=q(SS), p_skillM0_le0=float(np.mean(np.array(SM) <= 0)))
# 점추정 일치 확인
w1 = np.ones(len(y)); res["wauc_check"] = [wauc("champ", w1, y), wauc("M0", w1, y)]
import sys
if "--noboot" not in sys.argv:
    res["boot_cluster_9001"] = boot(y, 9001, 2000, True); print("b1", round(time.time() - t), flush=True)
    res["boot_iid_9002"] = boot(y, 9002, 1000, False); print("b2", round(time.time() - t), flush=True)
    res["boot_cluster_auxA"] = boot(yA, 9003, 1000, True); print("b3", round(time.time() - t), flush=True)
# 산업 집계 future_ksic_r6.csv 재계산
K = pd.read_csv(os.path.join(H, "future_ksic_r6.csv"), dtype={"ksic": str})
g = FS.assign(ksic=FS.ksic.astype(str), y=y).groupby("ksic").agg(n=("p", "size"), exp=("p", "sum"), mp=("p", "mean"), t1=("top1pct", "sum"), pos=("y", "sum"))
res["ksic_rows_csv"] = len(K); res["ksic_matched"] = int(K.ksic.isin(g.index).sum())
res["ksic_n_csv_vs_recalc_maxdiff"] = float(np.abs(K.set_index("ksic").n.values - g.reindex(K.ksic).n.values).max())
res["ksic_exp_maxdiff"] = float(np.abs(K.set_index("ksic").exp_disputes.values - g.reindex(K.ksic)["exp"].values).max())
res["ksic_meanp_maxdiff"] = float(np.abs(K.set_index("ksic").mean_p.values - g.reindex(K.ksic).mp.values).max())
res["ksic_top1_maxdiff"] = float(np.abs(K.set_index("ksic").n_top1pct.values - g.reindex(K.ksic).t1.values).max())
res["ksic_pop_covered"] = int(g.reindex(K.ksic).n.sum()); res["pop_ksic_levels"] = int(g.shape[0])
res["top1pct_flag_sum"] = int(FS.top1pct.sum())
gg = g.reindex(K.ksic); gg["rate"] = gg.pos / gg.n
res["rho_meanp_rate"] = float(spearmanr(gg.mp, gg.rate).correlation)
res["rho_exp_count"] = float(spearmanr(gg["exp"], gg.pos).correlation)
res["pos_in_ksic50"] = int(gg.pos.sum())
res["sum_p"] = float(S["champ"].sum()); res["calib_ratio"] = float(S["champ"].sum() * 546 / 730 / y.sum())
res["sec"] = round(time.time() - t)
json.dump(res, open(os.path.join(V, "v03a_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(res, ensure_ascii=False, indent=1, default=str))
