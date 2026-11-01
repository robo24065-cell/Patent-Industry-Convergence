# -*- coding: utf-8 -*-
"""S6 위약(사전 등록): 학습 레이블을 원점×KSIC 층 안에서 섞고 챔피언 사양으로 재학습. 시험 레이블은 실제 값.
skill은 실제 M0(E00) 예측 대비. 사용: python s6_placebo.py <rep_from> <rep_to>"""
import sys, json, os
sys.dont_write_bytecode = True
from n8lib import *
from e_run import load_panel, score, M0
a, b = int(sys.argv[1]), int(sys.argv[2])
cx = json.load(open(f"{CACHE}/champ_xgb.json", encoding="utf-8"))
D = load_panel(list(dict.fromkeys(M0 + cx["feats"])))
P0 = pd.read_parquet(f"{CACHE}/preds_xgb_E00.parquet")
f = f"{HERE}/s6_placebo.csv"
for r in range(a, b):
    rng = np.random.default_rng(1000 + r)
    ys = D.y.values.copy()
    for _, idx in D.groupby(["t0", "ksic"]).indices.items():
        ys[idx] = rng.permutation(ys[idx])
    p = score(D, cx["feats"], cx["cfg"], "xgb", ytrain=ys)
    sk = []
    for v in DEV:
        y = D.y.values[D.t0.values == v]
        sk.append(skill(auc(y, p[v]), auc(y, P0[P0.t0 == v].p.values)))
    row = pd.DataFrame([dict(rep=r, mean_skill=np.mean(sk), **{f"s_{v}": s for v, s in zip(DEV, sk)})])
    row.to_csv(f, mode="a", header=not os.path.exists(f), index=False, encoding="utf-8-sig")
    log(r, round(np.mean(sk), 4))
