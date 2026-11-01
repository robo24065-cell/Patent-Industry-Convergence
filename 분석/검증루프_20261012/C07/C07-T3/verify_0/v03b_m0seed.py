# -*- coding: utf-8 -*-
"""C07-T3 verify_0 v03b (반박): 새로 적합한 기준선 M0가 음성 표본 시드 0의 '운'인지. 시드 0(재현)·1·2·3·4로 M0 재적합 → skill vs M0 분포.
읽기만: panel_dev.parquet, s02a_tr2122/te2024.parquet, future_scores_r6.parquet, sqlite(mode=ro, 레이블). 쓰기: v03b_out.json"""
import os, json, sqlite3, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
t = time.time()
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N8_특허분쟁표적레이더"
REG = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
SP = json.load(open(os.path.join(H, "final_spec_r6.json"), encoding="utf-8"))
F = SP["references"]["M0"]["feats"]; PAR = SP["champion"]["params"]
D = pd.read_parquet(os.path.join(H, "cache", "panel_dev.parquet"), columns=["y"] + F)
T2 = pd.read_parquet(os.path.join(W, "cache", "s02a_tr2122.parquet"), columns=["y"] + F)
TR = pd.concat([D, T2[D.columns]], ignore_index=True); del D, T2
TE = pd.read_parquet(os.path.join(W, "cache", "s02a_te2024.parquet"), columns=["rgst_no"] + F)
FS = pd.read_parquet(os.path.join(H, "future_scores_r6.parquet"))
assert (TE.rgst_no.values == FS.rgst_no.values).all()
B = pd.read_parquet(os.path.join(W, "cache", "s02b_scores_t2024.parquet"), columns=["p_M0"])
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
E = pd.read_sql("select distinct rgst_no from right where rgstCsName in ('무효(심판)예고등록','권리범위(심판)예고등록') and rgstCsDate between '20250101' and '20260630'", c)
y = FS.rgst_no.isin(set(E.rgst_no)).values.astype(int)
ac = roc_auc_score(y, FS.p.values)
ytr = TR.y.values.astype(int); X = TR[F].astype(float); Xte = TE[F].astype(float)
out = dict(pos=int(y.sum()), n_TR=len(TR), champ_auc=ac, runs=[])
for sd in (0, 1, 2, 3, 4):
    rng = np.random.default_rng(sd)
    keep = (ytr == 1) | (rng.random(len(TR)) < 0.05)
    w = np.where(ytr == 1, 1.0, 20.0)[keep]
    m = XGBClassifier(random_state=0, **PAR); m.fit(X[keep], ytr[keep], sample_weight=w)
    p = m.predict_proba(Xte)[:, 1]
    a = roc_auc_score(y, p)
    r = dict(seed=sd, M0_auc=a, skill=(ac - a) / (1 - a))
    if sd == 0:
        r["maxdiff_vs_s02b"] = float(np.abs(p - B.p_M0.values).max())
    out["runs"].append(r); print(r, round(time.time() - t), flush=True)
out["sec"] = round(time.time() - t)
json.dump(out, open(os.path.join(V, "v03b_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
