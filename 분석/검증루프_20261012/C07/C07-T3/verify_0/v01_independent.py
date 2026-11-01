# -*- coding: utf-8 -*-
"""C07-T3 verify_0 v01: 독립 검증(다른 코드 경로). n8lib·c07lib·s04 산출물(레이블·지표)을 쓰지 않는다.
입력: sqlite(mode=ro), future_scores_r6.parquet(동결 챔피언), s02b_scores_t2024.parquet(기준선 점수만), final_preds_r6.parquet(지속 지수용)
- 레이블: SQL 이름 패턴(LIKE)으로 다시 찾고 이름 변형 확인
- AUC: 순위합(Mann–Whitney, scipy rankdata), 상위 k: 경계 동점 평균, PR-AUC: 직접 구현
- 군집: 원점 당시 권리자를 sqlite에서 다시 구성(정규화 규칙 독립 구현), 부트스트랩 시드 777/778
- 누수 반박: (a) 피처 own_disp를 sqlite로 재계산해 s02a 값과 대조, (b) reg_lt20250101 rt 행 수 = sqlite(< 2025) 행 수,
  (c) 위약: KSIC 층 안 레이블 섞기 1,000회, (d) 2025–26 정정심판 예고 특허(청구항 수 스냅샷 경로) 제외 재채점
→ v01_out.json"""
import os, re, json, sqlite3, time
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
t = time.time()
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N8_특허분쟁표적레이더"
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
FS = pd.read_parquet(os.path.join(H, "future_scores_r6.parquet"))
B = pd.read_parquet(os.path.join(W, "cache", "s02b_scores_t2024.parquet"), columns=["rgst_no", "p_M0", "p_SB", "own_disp"])
assert (FS.rgst_no.values == B.rgst_no.values).all()
res = {}
# ---------- 레이블(패턴)
names = pd.read_sql("select rgstCsName, count(*) n from right where rgstCsDate >= '20250101' and (rgstCsName like '%무효%' or rgstCsName like '%권리범위%') group by 1", c)
res["name_variants_ge2025"] = names.to_dict("records")
ev = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName from right where (rgstCsName like '%무효%예고%' or rgstCsName like '%권리범위%예고%') and rgstCsDate between '20250101' and '20260630'", c)
y = FS.rgst_no.isin(set(ev.rgst_no)).values.astype(int)
res["label"] = dict(events=int(len(ev)), patents=int(ev.rgst_no.nunique()), pos_in_pop=int(y.sum()))
# ---------- 지표(독립 구현)
def auc_rank(y, s):
    r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def topk_tie(y, s, k):
    o = np.sort(s)[::-1]; thr = o[k - 1]
    above = s > thr; at = s == thr; need = k - above.sum()
    return float(y[above].sum() + y[at].sum() * need / at.sum())
def ap(y, s):
    o = np.argsort(-s, kind="mergesort"); yy = y[o]; tp = np.cumsum(yy); prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())
S = {"champ": FS.p.values.astype(float), "M0": B.p_M0.values.astype(float), "SB": B.p_SB.values.astype(float)}
n = len(y); k1 = int(round(0.01 * n))
m = {}
for k, s in S.items():
    m[k] = dict(auc=auc_rank(y, s), top1=topk_tie(y, s, k1), hit100=topk_tie(y, s, 100), prauc=ap(y, s))
m["skill_vs_M0"] = (m["champ"]["auc"] - m["M0"]["auc"]) / (1 - m["M0"]["auc"])
m["skill_vs_SB"] = (m["champ"]["auc"] - m["SB"]["auc"]) / (1 - m["SB"]["auc"])
res["metrics"] = m
# ---------- 군집(독립 재구성)
def nn(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = re.sub(r"\(주\)|㈜|주식회사|유한회사|\(유\)|유한책임회사|\(사\)|사단법인|재단법인|\(재\)|株式会社|co\.?,?\s*ltd\.?|inc\.?|corp(oration)?\.?|limited", "", str(s), flags=re.I)
    return re.sub(r"[\s\.,\-·()\[\]]", "", s).lower()
ow = pd.read_sql("""select rgst_no, seq, ownerName nm from owner where ownerRgstCsDate <= '20241231'
    union all select rgst_no, seq, utilityModelOwnerName nm from utilityModelOwner where utilityModelOwnerRgstCsDate <= '20241231'""", c)
ow = ow.sort_values(["rgst_no", "seq"], kind="mergesort").drop_duplicates("rgst_no", keep="last")
cur = dict(zip(ow.rgst_no, ow.nm.map(nn)))
grp = np.array([cur.get(r, "") or ("R" + r) for r in FS.rgst_no])
codes, uq = pd.factorize(grp); G = len(uq); res["n_clusters"] = G
rng = np.random.default_rng(777); bs = {"M0": [], "SB": []}; ba = []
rk = {k: rankdata(s) for k, s in S.items()}
for _ in range(1000):
    w = np.bincount(rng.integers(0, G, G), minlength=G)[codes].astype(float)
    def wauc(s):
        # 가중 Mann–Whitney: 점수 순서로 누적 음성 가중
        o = np.argsort(s, kind="mergesort"); ss = s[o]; ww = w[o]; yy = y[o]
        u, inv = np.unique(ss, return_inverse=True)
        pos = np.bincount(inv, weights=ww * yy); neg = np.bincount(inv, weights=ww * (1 - yy))
        cn = np.cumsum(neg) - neg
        return (pos * (cn + 0.5 * neg)).sum() / (pos.sum() * neg.sum())
    a = wauc(S["champ"]); ba.append(a)
    for r in ("M0", "SB"):
        b = wauc(S[r]); bs[r].append((a - b) / (1 - b))
res["boot"] = dict(seed=777, champ_auc_ci=list(np.percentile(ba, [2.5, 97.5])), skill_vs_M0_ci=list(np.percentile(bs["M0"], [2.5, 97.5])),
                   skill_vs_SB_ci=list(np.percentile(bs["SB"], [2.5, 97.5])))
print("boot", res["boot"], round(time.time() - t), flush=True)
# ---------- 누수 반박
# (a) own_disp 재계산
d_all = pd.read_sql("select rgst_no, rgstCsDate d from right where rgstCsName in ('무효(심판)예고등록','권리범위(심판)예고등록')", c)
d_all["d"] = pd.to_numeric(d_all.d, errors="coerce")
c24 = FS.rgst_no.map(d_all[d_all.d <= 20241231].groupby("rgst_no").size()).fillna(0).values
c26 = FS.rgst_no.map(d_all[d_all.d <= 20260815].groupby("rgst_no").size()).fillna(0).values
res["leak_a_own_disp"] = dict(eq_le2024=bool(np.array_equal(c24, B.own_disp.values.astype(float))), eq_le2026=bool(np.array_equal(c26, B.own_disp.values.astype(float))),
                              n_diff_if_2026=int((c26 != B.own_disp.values).sum()))
# (b) 추출본 행 수
n_rt = c.execute("select count(*) from right where rgstCsDate < '20250101'").fetchone()[0]
R = pd.read_pickle(os.path.join(H, r"cache\reg_lt20250101.pkl"))
res["leak_b_rt_rows"] = dict(sqlite_lt2025=int(n_rt), pickle=int(len(R["rt"])), equal=bool(n_rt == len(R["rt"])))
del R
# (c) 위약: KSIC 층 안 레이블 섞기
ks = FS.ksic.astype(str).values; rng2 = np.random.default_rng(2024); pl = []
idx_by = [np.where(ks == k)[0] for k in np.unique(ks)]
for _ in range(1000):
    yp = y.copy()
    for ix in idx_by:
        yp[ix] = yp[rng2.permutation(ix)]
    a = auc_rank(yp, S["champ"]); b = auc_rank(yp, S["M0"]); s_ = auc_rank(yp, S["SB"])
    pl.append(((a - b) / (1 - b), (a - s_) / (1 - s_), a))
pl = np.array(pl)
res["leak_c_placebo"] = dict(skill_M0_mean=float(pl[:, 0].mean()), skill_M0_q=list(np.percentile(pl[:, 0], [2.5, 97.5])), skill_M0_max=float(pl[:, 0].max()),
                             skill_SB_mean=float(pl[:, 1].mean()), skill_SB_q=list(np.percentile(pl[:, 1], [2.5, 97.5])), auc_mean=float(pl[:, 2].mean()),
                             ge_obs_M0=int((pl[:, 0] >= m["skill_vs_M0"]).sum()), ge_obs_SB=int((pl[:, 1] >= m["skill_vs_SB"]).sum()))
print("placebo", res["leak_c_placebo"], round(time.time() - t), flush=True)
# (d) 2025–26 정정심판 예고 특허 제외 재채점(청구항 수 스냅샷 경로)
cor = pd.read_sql("select distinct rgst_no from right where rgstCsName='정정(심판)예고등록' and rgstCsDate >= '20250101'", c)
cm = FS.rgst_no.isin(set(cor.rgst_no)).values
keep = ~cm
res["leak_d_correction"] = dict(n_corr_patents_in_pop=int(cm.sum()), pos_with_corr=int((cm & (y == 1)).sum()),
                                corr_in_champ_top1=int(cm[np.argsort(-S["champ"], kind="stable")[:k1]].sum()),
                                auc_champ_excl=auc_rank(y[keep], S["champ"][keep]), auc_M0_excl=auc_rank(y[keep], S["M0"][keep]), auc_SB_excl=auc_rank(y[keep], S["SB"][keep]))
dd = res["leak_d_correction"]
dd["skill_vs_M0_excl"] = (dd["auc_champ_excl"] - dd["auc_M0_excl"]) / (1 - dd["auc_M0_excl"]); dd["skill_vs_SB_excl"] = (dd["auc_champ_excl"] - dd["auc_SB_excl"]) / (1 - dd["auc_SB_excl"])
# ---------- 산업 지수·보정 독립 재계산
K = pd.read_csv(os.path.join(H, "future_ksic_r6.csv"), encoding="utf-8-sig", dtype={"ksic": str}).set_index("ksic")
D = pd.DataFrame({"k": FS.ksic.astype(str), "y": y}).groupby("k").y.agg(["sum", "size"])
D["rate"] = D["sum"] / D["size"]; D = D.join(K)
FP = pd.read_parquet(os.path.join(H, "final_preds_r6.parquet"), columns=["ksic", "y"]); FP["ksic"] = FP.ksic.astype(str)
pr = FP.groupby("ksic").y.mean()
res["ksic"] = dict(rho_meanp_rate=float(spearmanr(D.mean_p, D.rate).correlation), rho_exp_count=float(spearmanr(D.exp_disputes, D["sum"]).correlation),
                   rho_persist_rate=float(spearmanr(pr.reindex(D.index), D.rate).correlation), n_zero=int((D["sum"] == 0).sum()))
res["calibration"] = dict(sum_p=float(S["champ"].sum()), adj_546=float(S["champ"].sum() * 546 / 730), realized=int(y.sum()), ratio=float(S["champ"].sum() * 546 / 730 / y.sum()))
res["sec"] = round(time.time() - t)
json.dump(res, open(os.path.join(V, "v01_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("done", res["sec"])
