# verify_1: C03-T3 N9 '기업' 하위집단 교정 재채점 반박 시도. 독립 경로(panel_final 캐시 + 원천 텍스트 필드 + 자체 순위 AUC). 읽기 전용.
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from scipy.stats import rankdata
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
N9 = MC + r"\N9_정부RnD과제_특허성과예측"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C03\C03-T3\verify_1"

def auc(y, s):
    y = np.asarray(y, int); r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

out = {}
D = pd.read_parquet(N9 + r"\cache\panel_final.parquet", columns=["pjt_no", "year", "t0", "perform_grp", "firm", "y"])
te = D[D.t0 == 2022].reset_index(drop=True)
P = pd.read_parquet(N9 + r"\final_preds_r6.parquet")
out["te_n"] = len(te); out["preds_n"] = len(P)
out["te_pjt_unique"] = bool(te.pjt_no.is_unique)
out["order_aligned"] = bool((te.pjt_no.values == P.pjt_no.values).all()) if len(te) == len(P) else False
out["y_aligned"] = bool((te.y.values == P.y.values).all()) if out["order_aligned"] else None
X = P.copy(); X["firm"] = te.firm.values; X["grp_panel"] = te.perform_grp.astype(str).values

R = pd.read_parquet(MC + r"\_shared\ext\ntis_project_raw.parquet",
                    columns=["pjt_no", "year", "research_agency", "corp_rgst_no", "perform_agent_cd", "perform_agent", "perform_grp", "secret_flag", "sbiz_funds", "nongov_funds", "total_funds"])
R["year"] = R.year.astype(int)
# 매핑 진위: 코드 06/08 기관명·법인번호·민간부담 비율
for cd in ["04", "05", "06", "08", "99"]:
    s = R[R.perform_agent_cd == cd]
    out[f"code{cd}"] = dict(n=int(len(s)), text=s.perform_agent.value_counts().head(3).to_dict(), grp=s.perform_grp.value_counts().head(3).to_dict(),
                           corp_no_share=float(s.corp_rgst_no.notna().mean()),
                           nongov_pos_share=float((s.nongov_funds.fillna(0) > 0).mean()),
                           top_orgs=s.research_agency.value_counts().head(12).to_dict(),
                           years=[int(s.year.min()), int(s.year.max())] if len(s) else None)
# 연도별 06/08 텍스트 일관성
yy = R[R.perform_agent_cd.isin(["06", "08"])].groupby(["perform_agent_cd", "perform_agent"]).year.agg(["min", "max", "size"]).reset_index()
out["code_text_year"] = yy.astype(str).values.tolist()

# 2022 원천 행: 과제당 여러 행 여부
R22 = R[(R.year == 2022)]
dup = R22.groupby("pjt_no").perform_agent.nunique()
out["r22_pjt_multi_text"] = int((dup > 1).sum())
R22u = R22.groupby("pjt_no").agg(txt=("perform_agent", lambda s: "|".join(sorted(set(s.dropna().astype(str))))),
                                 grp=("perform_grp", lambda s: "|".join(sorted(set(s.dropna().astype(str))))))
X = X.join(R22u, on="pjt_no")
out["unmatched"] = int(X.txt.isna().sum())
out["panel_grp_vs_raw_grp_match"] = float((X.grp_panel == X.grp).mean())
FIRM = {"대기업", "중소기업", "중견기업"}
m_off = (X.firm.values == 1)
m_cor = X.txt.fillna("").apply(lambda t: any(x in FIRM for x in t.split("|"))).values  # 텍스트 기준 교정 정의

def sub(m):
    y = X.y.values[m]; a1 = auc(y, X.p_champ.values[m]); a0 = auc(y, X.p_M0_L2.values[m])
    k = max(1, int(round(0.01 * m.sum())))
    o = np.argsort(-X.p_champ.values[m], kind="stable")[:k]
    return dict(n=int(m.sum()), pos=int(y.sum()), auc=float(a1), auc_M0L2=float(a0), skill=float((a1 - a0) / (1 - a0)), top1=float(y[o].mean()), k=k)
out["official_recomputed"] = sub(m_off)
out["corrected_text"] = sub(m_cor)
out["in_only_cor"] = dict(n=int((m_cor & ~m_off).sum()), pos=int(X.y.values[m_cor & ~m_off].sum()), txt=X.txt[m_cor & ~m_off].value_counts().to_dict())
out["in_only_off"] = dict(n=int((m_off & ~m_cor).sum()), pos=int(X.y.values[m_off & ~m_cor].sum()), txt=X.txt[m_off & ~m_cor].value_counts().to_dict())
# 하위 분해: 중견기업만, 대기업+중견 (공식 vs 교정)
m_mid = X.txt.fillna("").str.contains("중견기업").values
if X.y.values[m_mid].sum() >= 10: out["mid_only_corrected"] = sub(m_mid)
m_big_off = X.grp_panel.isin(["대기업", "중견기업"]).values
m_big_cor = X.txt.fillna("").str.contains("대기업|중견기업").values
out["big_mid_official"] = sub(m_big_off); out["big_mid_corrected"] = sub(m_big_cor)
# 부트스트랩(과제 단위, 교정-공식 skill 차의 분포 — 대략적 불확실성)
rng = np.random.default_rng(7); d = []
idx_all = np.arange(len(X))
for b in range(200):
    i = rng.integers(0, len(X), len(X)); mo = m_off[i]; mc = m_cor[i]; yb = X.y.values[i]; pc = X.p_champ.values[i]; p0 = X.p_M0_L2.values[i]
    def sk(m):
        a1 = auc(yb[m], pc[m]); a0 = auc(yb[m], p0[m]); return (a1 - a0) / (1 - a0)
    d.append(sk(mc) - sk(mo))
out["boot_skill_diff_p2_5_p97_5"] = [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
json.dump(out, open(OUT + r"\v1_n9.json", "w", encoding="utf8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
