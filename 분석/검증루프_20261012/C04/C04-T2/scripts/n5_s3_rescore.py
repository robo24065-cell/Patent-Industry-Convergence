# -*- coding: utf-8 -*-
"""C04-T2 N5 재채점: 동결 예측 final_preds_r4.parquet + 독립 레이블(n5_s2). 강한 기준선: VERIFY_R5 sens_preds p_M0plus + 동결 파라미터 재적합 대조.
출력: cache/n5_s3_out.json"""
import os, json, time, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N5 = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N5_표준특허선언레이더"
OFF = json.load(open(os.path.join(N5, "final_result_r4.json"), encoding="utf-8"))
SPEC = json.load(open(os.path.join(N5, "final_spec_r4.json"), encoding="utf-8"))
sha = lambda p: hashlib.sha1(open(p, "rb").read()).hexdigest()[:16]
mt = lambda p: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(p)))
F = pd.read_parquet(os.path.join(N5, "final_preds_r4.parquet")); F["appl_no"] = F.appl_no.astype(str)
U = pd.read_parquet(os.path.join(W, "cache", "n5_units_indep.parquet")).set_index("appl_no")
y = U.y.reindex(F.appl_no).values
assert not np.isnan(y).any(); y = y.astype(int)
A = lambda yy, pp: float(roc_auc_score(yy, pp))
sk = lambda a, b: (a - b) / (1 - b)


def wauc(yy, s, w):
    o = np.argsort(s, kind="stable"); ss = s[o]; yo = yy[o].astype(float); wo = w[o]
    g = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])]); G = int(g[-1]) + 1
    Pg = np.bincount(g, wo * yo, G); Ng = np.bincount(g, wo * (1 - yo), G)
    return float((Pg * (np.cumsum(Ng) - Ng + 0.5 * Ng)).sum() / (Pg.sum() * Ng.sum()))


cl_codes, inv = np.unique(F.app.values.astype(str), return_inverse=True); G = len(cl_codes)


def boot(p1, p0, B, seed, yy=None):
    yy = y if yy is None else yy
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        pick = rng.integers(0, G, G)
        w = np.bincount(pick, minlength=G).astype(float)[inv]
        if yy[w > 0].min() == yy[w > 0].max():
            continue
        out.append(sk(wauc(yy, p1, w), wauc(yy, p0, w)))
    return dict(ci=[round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)], mean=float(np.mean(out)), B=B, seed=seed)


pc, p0 = F.p_champ.values, F.p_M0.values
R = dict(inputs=dict(final_preds=dict(sha1=sha(os.path.join(N5, "final_preds_r4.parquet")), mtime=mt(os.path.join(N5, "final_preds_r4.parquet"))),
                     final_result_mtime=mt(os.path.join(N5, "final_result_r4.json"))),
         n=int(len(y)), pos=int(y.sum()), label_equal_frozen=bool((y == F.y.values).all()), n_app_clusters=int(G))
R["auc"] = {k: A(y, F[f"p_{k}"].values) for k in ["champ", "M0", "M1_pilot", "champ_logit", "M0_logit"]}
R["skill"] = sk(R["auc"]["champ"], R["auc"]["M0"])
R["diff_vs_official"] = dict(auc_champ=R["auc"]["champ"] - OFF["auc"]["champ"], auc_M0=R["auc"]["M0"] - OFF["auc"]["M0"], skill=R["skill"] - OFF["skill_champ_vs_M0"])
R["top100"] = {k: int(y[np.argsort(-F[f"p_{k}"].values)[:100]].sum()) for k in ["champ", "M0", "M1_pilot"]}
R["prauc"] = dict(champ=float(average_precision_score(y, pc)), M0=float(average_precision_score(y, p0)))
R["official"] = dict(n=OFF["n"], pos=OFF["pos"], auc=OFF["auc"], skill=OFF["skill_champ_vs_M0"], ci=OFF["ci95_cluster_app"], top100=OFF["top100_hits"])
R["boot_official_procedure_seed2022"] = boot(pc, p0, 1000, 2022)
R["boot_other_seeds"] = [boot(pc, p0, 1000, s) for s in (1, 2, 3, 4)]
print("official", R["auc"], R["skill"], R["boot_official_procedure_seed2022"], round(time.time() - t0), flush=True)
# ---------------- 강한 기준선
S = pd.read_parquet(os.path.join(N5, r"VERIFY_R5\cache\sens_preds.parquet")); S["appl_no"] = S.appl_no.astype(str)
assert (S.appl_no.values == F.appl_no.values).all()
pS = S.p_M0plus.values
R["strong_file"] = dict(sha1=sha(os.path.join(N5, r"VERIFY_R5\cache\sens_preds.parquet")), auc=A(y, pS), skill=sk(R["auc"]["champ"], A(y, pS)),
                        top100=int(y[np.argsort(-pS)[:100]].sum()),
                        boot_verify_procedure_seed55=boot(pc, pS, 1000, 55), boot_other_seeds=[boot(pc, pS, 1000, s) for s in (1, 2, 3, 4)])
print("strong", R["strong_file"]["auc"], R["strong_file"]["skill"], R["strong_file"]["boot_verify_procedure_seed55"], round(time.time() - t0), flush=True)
# 재적합(기준선 재현용, 동결 XGB 파라미터·시드 0,1,2; VERIFY_R5 재생성 패널)
tr = pd.read_parquet(os.path.join(N5, r"VERIFY_R5\cache\panel_train.parquet")); te = pd.read_parquet(os.path.join(N5, r"VERIFY_R5\cache\panel_final.parquet"))
te["appl_no"] = te.appl_no.astype(str)
assert (te.appl_no.values == F.appl_no.values).all() and int(tr.t0.max()) + 2 < 2023
from xgboost import XGBClassifier
f0p = SPEC["baseline_M0"] + ["sg_rate", "sg_logn", "app_rec_rate", "mg_rec_rate"]
pr = np.zeros(len(te))
for s in SPEC["seeds"]:
    m = XGBClassifier(random_state=s, **SPEC["xgb_params"]).fit(tr[f0p].astype(float), tr.y)
    pr += m.predict_proba(te[f0p].astype(float))[:, 1] / len(SPEC["seeds"])
R["strong_refit"] = dict(maxdiff_vs_file=float(np.abs(pr - pS).max()), auc=A(y, pr), skill=sk(R["auc"]["champ"], A(y, pr)), train_rows=int(len(tr)), train_t0=sorted(tr.t0.unique().tolist()))
R["official_strong"] = dict(auc=0.8580, skill=0.0002, ci=[-0.072, 0.070], src="VERIFY_R5/robustness_table.md §3.1")
# ---------------- 연결 키 확장 민감도(정보): ext 규칙 단위·레이블로 다시 채점
Ue = U.reindex(F.appl_no)
excl = (Ue.refl_e <= 20221231).values
ye = ((Ue.refl_e > 20221231) & (Ue.refl_e <= 20241231)).astype(int).values
mk = ~excl
R["sens_ext_key"] = dict(n=int(mk.sum()), pos=int(ye[mk].sum()), n_excluded=int(excl.sum()), pos_excluded=int(y[excl].sum()), flips=int((ye[mk] != y[mk]).sum()),
                         auc_champ=A(ye[mk], pc[mk]), auc_M0=A(ye[mk], p0[mk]), skill=sk(A(ye[mk], pc[mk]), A(ye[mk], p0[mk])),
                         skill_vs_strong=sk(A(ye[mk], pc[mk]), A(ye[mk], pS[mk])))
R["secs"] = round(time.time() - t0, 1)
json.dump(R, open(os.path.join(W, "cache", "n5_s3_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
