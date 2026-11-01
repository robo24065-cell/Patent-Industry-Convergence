# -*- coding: utf-8 -*-
"""C04-T2 T4 재채점: 동결 예측 npy(R2/runs_final/F22/pred)만 사용(재적합 없음) + 독립 레이블.
강화 기준선: (A) VERIFY_R5 strong_full.npy, (B) 동결 학습 프레임으로 같은 13변수 로지스틱 재적합(기준선 재현용).
출력: cache/t4_s2_out.json"""
import os, json, time, hashlib
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TR = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\T4_특허포기예측"
FD = os.path.join(TR, r"R2\runs_final\F22")
OFF = json.load(open(os.path.join(TR, r"R2\final_result.json"), encoding="utf-8"))
FS = json.load(open(os.path.join(TR, r"R2\final_spec_r2.json"), encoding="utf-8"))
ws = FS["spec"]["ensemble"]["weights"]; how = FS["spec"]["ensemble"].get("how", "logit")


def sha(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:16]


files = {"cls": "cls_99f270b3cd.npy", "haz": "haz_0568d4297b.npy", "logit": "logit_3008b8b195.npy", "hgb": "hgb_b4bd0fa46a.npy", "base": "base_46526c56d8.npy"}
P = {k: np.load(os.path.join(FD, "pred", f)) for k, f in files.items()}
meta = {k: dict(file=f, sha1=sha(os.path.join(FD, "pred", f)), mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(os.path.join(FD, "pred", f))))) for k, f in files.items()}
meta["final_result.json_mtime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(os.path.join(TR, r"R2\final_result.json"))))
meta["ev.parquet"] = dict(sha1=sha(os.path.join(FD, "ev.parquet")), mtime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(os.path.join(FD, "ev.parquet")))))
SB_COLS = ["age", "claims", "rgst_year", "atype", "ksic", "next_annl", "paid_ahead", "n_due_win", "tier_in_win", "a_lapse_rate", "a_lapse_rate2", "n_late", "yrs_since_pay"]
ev = pd.read_parquet(os.path.join(FD, "ev.parquet"), columns=["pid", "acd", "y"] + SB_COLS)
lab = pd.read_parquet(os.path.join(W, "cache", "t4_labels_indep.parquet")).set_index("pid")
y_frz = ev.y.values.astype(int)
y = lab.y_indep.reindex(ev.pid).values
assert not np.isnan(y).any()
y = y.astype(int)
y_lap = lab.y_lapse_event.reindex(ev.pid).values.astype(int)
assert ws and how == "logit", (ws, how)
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
z = sum(w * lg(P[k]) for k, w in ws.items() if w != 0) / sum(w for w in ws.values() if w != 0)
pm = 1 / (1 + np.exp(-z)); pb = P["base"]
A = lambda yy, pp: float(roc_auc_score(yy, pp))
sk = lambda a, b: (a - b) / (1 - b)


class WA:
    """가중 AUC(동점 0.5) — 점수 오름차순 정렬·동점 묶음을 한 번만 계산"""
    def __init__(self, yy, s):
        o = np.argsort(s, kind="stable"); ss = s[o]
        self.o = o; self.y = yy[o].astype(float)
        self.g = np.concatenate([[0], np.cumsum(ss[1:] != ss[:-1])]); self.G = int(self.g[-1]) + 1

    def __call__(self, w):
        wo = w[self.o]
        Pg = np.bincount(self.g, wo * self.y, self.G); Ng = np.bincount(self.g, wo * (1 - self.y), self.G)
        below = np.cumsum(Ng) - Ng
        return float((Pg * (below + 0.5 * Ng)).sum() / (Pg.sum() * Ng.sum()))


cl = np.where(ev.acd.values == "NA", "P" + ev.pid.astype(str).values, ev.acd.values)
codes, uniq = pd.factorize(cl); K = len(uniq)


def boot(p1, p0, B, seed, yy=None):
    yy = y if yy is None else yy
    a1, a0 = WA(yy, p1), WA(yy, p0)
    rng = np.random.default_rng(seed); s = []; aa = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[codes]
        x1, x0 = a1(w), a0(w); s.append(sk(x1, x0)); aa.append(x1)
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return dict(skill_ci=q(s), auc_ci=q(aa), seed=seed, B=B)


R = dict(inputs=meta, spec_weights=ws, n=int(len(y)), pos=int(y.sum()), pos_frozen_ev=int(y_frz.sum()), label_equal_frozen=bool((y == y_frz).all()), n_clusters=int(K))
R["auc_model"] = A(y, pm); R["auc_base"] = A(y, pb); R["skill"] = sk(R["auc_model"], R["auc_base"])
R["prauc_model"] = float(average_precision_score(y, pm)); R["prauc_base"] = float(average_precision_score(y, pb))
R["brier_model"] = float(brier_score_loss(y, pm)); R["mean_pred"] = float(pm.mean())
R["components"] = {k: A(y, v) for k, v in P.items()}
top = np.argsort(-pm)[:int(0.1 * len(pm))]; topb = np.argsort(-pb)[:int(0.1 * len(pb))]
R["precision_top10"] = dict(model=float(y[top].mean()), baseline=float(y[topb].mean()))
R["top100"] = dict(model_quicksort=int(y[np.argsort(-pm)[:100]].sum()), model_stable=int(y[np.argsort(-pm, kind="stable")[:100]].sum()),
                   base=int(y[np.argsort(-pb)[:100]].sum()), score_at_100=float(np.sort(pm)[::-1][99]), ties_at_100=int((pm == np.sort(pm)[::-1][99]).sum()))
R["official"] = dict(auc_model=OFF["model"]["auc"], auc_base=OFF["baseline"]["auc"], skill=OFF["skill_auc"], ci=OFF["bootstrap_95ci"]["skill"], n=OFF["n"], pos=int(round(OFF["base_rate"] * OFF["n"])))
R["diff_vs_official"] = dict(auc_model=R["auc_model"] - OFF["model"]["auc"], auc_base=R["auc_base"] - OFF["baseline"]["auc"], skill=R["skill"] - OFF["skill_auc"])
print("point", R["auc_model"], R["auc_base"], R["skill"], round(time.time() - t0), flush=True)
R["boot_official_procedure_seed0_B1000"] = boot(pm, pb, 1000, 0)
R["boot_other_seeds_B1000"] = [boot(pm, pb, 1000, s) for s in (11, 22, 33, 44)]
print("boot", R["boot_official_procedure_seed0_B1000"], round(time.time() - t0), flush=True)
# ---------------- 강화 기준선
ps_frz = np.load(os.path.join(TR, r"VERIFY_R5\cache\strong_full.npy"))
R["strong_frozen_file"] = dict(sha1=sha(os.path.join(TR, r"VERIFY_R5\cache\strong_full.npy")))
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
fr = pd.read_parquet(os.path.join(FD, "frame_cls.parquet"), columns=["y", "L"] + SB_COLS)
R["frame_cls"] = dict(rows=int(len(fr)), pos=float(fr.y.mean()), L_max=str(fr.L.max()))
fr = fr.sample(n=min(600_000, len(fr)), random_state=0)
num = ["age", "claims", "rgst_year", "next_annl", "paid_ahead", "n_due_win", "tier_in_win", "a_lapse_rate", "a_lapse_rate2", "n_late", "yrs_since_pay"]


def prep(d):
    X = d[num + ["atype", "ksic"]].copy(); X["claims"] = np.log1p(X.claims.astype(float))
    for c in ["atype", "ksic"]: X[c] = X[c].astype(str)
    return X


ct = ColumnTransformer([("n", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), num), ("c", OneHotEncoder(handle_unknown="ignore", min_frequency=20), ["atype", "ksic"])])
mdl = make_pipeline(ct, LogisticRegression(C=1.0, max_iter=2000)).fit(prep(fr), fr.y.values)
ps_re = mdl.predict_proba(prep(ev))[:, 1]
R["strong_refit_vs_frozen_maxdiff"] = float(np.abs(ps_re - ps_frz).max())
for nm, ps in (("frozen_file", ps_frz), ("refit", ps_re)):
    a0 = A(y, ps)
    d = dict(auc_strong=a0, skill_vs_strong=sk(R["auc_model"], a0), top100_strong=int(y[np.argsort(-ps)[:100]].sum()))
    if nm == "frozen_file":
        d["boot_verify_procedure_seed3_B300"] = boot(pm, ps, 300, 3)
        d["boot_B1000_5seeds"] = [boot(pm, ps, 1000, s) for s in (0, 11, 22, 33, 44)]
    R["strong_" + nm] = d
    print(nm, d.get("auc_strong"), d.get("skill_vs_strong"), round(time.time() - t0), flush=True)
R["official_strong"] = dict(auc=0.6586, skill=0.124, ci=[-0.003, 0.208], src="VERIFY_R5/robustness_table.md §3.1")
# ---------------- 레이블 정의 민감도(정보, 판정 외): 소멸등록 사건 기준(전체 스냅샷)
R["sens_label_lapse_event"] = dict(pos=int(y_lap.sum()), auc_model=A(y_lap, pm), auc_base=A(y_lap, pb), skill=sk(A(y_lap, pm), A(y_lap, pb)),
                                   skill_vs_strong=sk(A(y_lap, pm), A(y_lap, ps_frz)))
dis = (y == 0) & (y_lap == 1)
R["sens_label_lapse_event"]["np0_lap1"] = int(dis.sum()); R["sens_label_lapse_event"]["np0_lap1_recov"] = int((dis & lab.recov_after_L.reindex(ev.pid).values).sum())
R["secs"] = round(time.time() - t0, 1)
json.dump(R, open(os.path.join(W, "cache", "t4_s2_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in R.items() if k not in ("inputs",)}, ensure_ascii=False, indent=1, default=float))
