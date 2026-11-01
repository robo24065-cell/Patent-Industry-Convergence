# C10-T4 s1: v10->v11 변경 5행 원천 직접 재계산 (읽기 전용)
import sys, os, json, csv, hashlib
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
MC = os.path.join(ROOT, "모델경쟁_20260927"); LP = os.path.join(ROOT, "검증루프_20261012")
W = os.path.join(LP, r"C10\C10-T4")
out = {}
# --- 75/134: FREE-IPC 재현율 ---
c1 = pd.read_csv(os.path.join(MC, r"_shared_report\fusion_advantage\c5_R10\c1_label_agreement.csv"), encoding="utf-8-sig")
r = c1[c1.window == "출원2015-19_발명"].set_index("cc")
out["c1_windows"] = sorted(c1.window.unique().tolist())
out["free5_2015_19"] = {cc: round(float(r.loc[cc, "free5"]) * 100, 4) for cc in r.index}
out["free5_labeled_2015_19"] = {cc: round(float(r.loc[cc, "free5_labeled"]) * 100, 4) for cc in r.index}
lab = pd.read_parquet(os.path.join(MC, r"_shared_report\label_origin_R10\confirm_R12\cache\labels.parquet"), columns=["appl_no", "ay", "split", "PROV", "H-sg-3"])
for sp in ["eval", "dev"]:
    d = lab[lab.split == sp]
    out[f"KR_Hsg3_{sp}"] = dict(n=int(len(d)), ay=[int(d.ay.min()), int(d.ay.max())], dup=int(d.appl_no.duplicated().sum()),
                               agree=round(float((d["H-sg-3"] == d.PROV).mean()) * 100, 4))
rp = json.load(open(os.path.join(MC, r"_shared_report\label_origin_R10\confirm_R12\res_prep.json"), encoding="utf-8"))
try:
    out["res_prep_Hsg3"] = rp["doc_agree_with_prov"]["H-sg-3"]
except Exception as e:
    out["res_prep_Hsg3"] = str(e)
kr = out["KR_Hsg3_eval"]["agree"]; fo = [v for k, v in out["free5_2015_19"].items() if k != "KR"]
out["line75_nonrepro"] = dict(KR=round(100 - kr, 3), foreign={k: round(100 - v, 3) for k, v in out["free5_2015_19"].items() if k != "KR"},
                              range_raw=[round(min([100 - kr] + [100 - v for v in fo]), 3), round(max([100 - kr] + [100 - v for v in fo]), 3)])
out["line134_foreign_range_raw"] = [round(min(fo), 3), round(max(fo), 3)]
# --- 125: T4 독립 재합성 ---
from sklearn.metrics import roc_auc_score
F = os.path.join(MC, r"T4_특허포기예측\R2\runs_final\F22")
spec = json.load(open(os.path.join(MC, r"T4_특허포기예측\R2\final_spec_r2.json"), encoding="utf-8"))["spec"]
ws = spec["ensemble"]["weights"]; out["t4_weights"] = ws; out["t4_how"] = spec["ensemble"].get("how", "logit")
fn = {"cls": "cls_99f270b3cd", "haz": "haz_0568d4297b", "logit": "logit_3008b8b195", "hgb": "hgb_b4bd0fa46a"}
ev = pd.read_parquet(os.path.join(F, "ev.parquet"), columns=["y"])
m = ev.y.notna().values; y = ev.y.values[m]
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
z = sum(w * lg(np.load(os.path.join(F, "pred", fn[k] + ".npy"))) for k, w in ws.items() if w) / sum(w for w in ws.values() if w)
p = 1 / (1 + np.exp(-z)); pb = np.load(os.path.join(F, "pred", "base_46526c56d8.npy"))
auc = roc_auc_score(y, p[m]); aucb = roc_auc_score(y, pb[m]); sk = (auc - aucb) / (1 - aucb)
fr = json.load(open(os.path.join(MC, r"T4_특허포기예측\R2\final_result.json"), encoding="utf-8"))
out["t4"] = dict(n=int(m.sum()), auc=auc, auc_base=aucb, skill=sk, off_auc=fr["model"]["auc"], off_base=fr["baseline"]["auc"], off_skill=fr["skill_auc"],
                 d_auc=abs(auc - fr["model"]["auc"]), d_base=abs(aucb - fr["baseline"]["auc"]), d_skill=abs(sk - fr["skill_auc"]))
n5 = json.load(open(os.path.join(LP, r"C04\C04-T2\n5_rescore.json"), encoding="utf-8"))
out["n5_c04_diff"] = n5.get("diff_vs_official")
t4c = json.load(open(os.path.join(LP, r"C04\C04-T2\t4_rescore.json"), encoding="utf-8"))
out["t4_c04_diff"] = t4c.get("diff_vs_official")
# --- 168: 28511 KR P3 1위 출원인 몫(제공 meta 직접) ---
import duckdb
con = duckdb.connect(); con.sql("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false")
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
FIRST = "nullif(trim(split_part(coalesce(applicant,''),'|',1)),'')"
q = f"""select ksic, {FIRST} a, count(*) n from read_parquet('{META}')
 where cc='KR' and substr(appl_no,1,2)='10' and appl_year between 2017 and 2019 and ksic in ('28511','26212')
 group by 1,2"""
d = con.sql(q).df()
res168 = {}
for k, g in d.groupby("ksic"):
    g = g.sort_values("n", ascending=False); tot = g.n.sum()
    res168[k] = dict(n_rows=int(tot), top1=str(g.a.iloc[0])[:3] + "…", top1_share=round(float(g.n.iloc[0] / tot) * 100, 2),
                     top2_share=round(float(g.n.iloc[:2].sum() / tot) * 100, 2), n_missing=int(g.a.isna().sum()))
out["line168_conc_raw_rows"] = res168
q2 = f"""select ksic, {FIRST} a, count(distinct appl_no) n from read_parquet('{META}')
 where cc='KR' and substr(appl_no,1,2)='10' and appl_year between 2017 and 2019 and ksic in ('28511','26212')
 group by 1,2"""
d2 = con.sql(q2).df(); r2 = {}
for k, g in d2.groupby("ksic"):
    g = g.sort_values("n", ascending=False); tot = g.n.sum()
    r2[k] = dict(top1_share=round(float(g.n.iloc[0] / tot) * 100, 2), top2_share=round(float(g.n.iloc[:2].sum() / tot) * 100, 2))
out["line168_conc_uniq"] = r2
sens = pd.read_csv(os.path.join(LP, r"C08\C08-T1\sensitivity.csv"), encoding="utf-8-sig")
out["line168_sens_rows"] = sens[sens.item.str.contains("28511|26212|1위 산업 수", regex=True)][["item", "S0_reproduced", "S1_main", "verify0_S1", "note"]].to_dict("records")
# --- 217: S3 사후 재채점 ---
rs = pd.read_csv(os.path.join(LP, r"C05\C05-T1\rescore_scenarios.csv"), encoding="utf-8-sig").set_index("scenario")
out["line217"] = dict(S0=float(rs.loc["S0 공식 재현", "skill"]), S3=float(rs.loc["S3", "skill"]), S3m=float(rs.loc["S3m(민감도)", "skill"]))
json.dump(out, open(os.path.join(W, r"cache\s1_diff.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
