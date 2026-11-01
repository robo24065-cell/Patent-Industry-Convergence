# -*- coding: utf-8 -*-
"""최종시험 사후 민감도(보고 전용, 모형·공식 숫자 불변): 보정(예측 합 대 실제), 권리자 집중도와 상위 권리자 제외 AUC·skill,
KSIC별 양성 분포, 전체 분쟁 특허(모집단 밖 포함) 대비 포착. 권리자 이름은 출력하지 않는다. → v03_final_sensitivity.json"""
import sys, os, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from n8lib import HERE, auc, skill, topk
P = pd.read_parquet(os.path.join(HERE, "final_preds_r6.parquet"))
V = json.load(open(os.path.join(HERE, "v01_final_independent.json"), encoding="utf-8"))
y = P.y.values
out = {"calibration": {m: dict(sum_p=float(P[f"p_{m}"].sum()), actual=int(y.sum()), ratio=float(P[f"p_{m}"].sum() / y.sum())) for m in ["champ", "M0", "SB"]}}
# 권리자 군집: v01과 같은 규칙으로 다시 만든다
import sqlite3
from n8lib import REG, norm_name
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True); E = "20221231"
ow = pd.read_sql(f"""select rgst_no, seq, ownerName nm, ownerRgstCsDate d from owner where ownerRgstCsDate is null or ownerRgstCsDate='' or ownerRgstCsDate <= '{E}'
    union all select rgst_no, seq, utilityModelOwnerName nm, utilityModelOwnerRgstCsDate d from utilityModelOwner
    where utilityModelOwnerRgstCsDate is null or utilityModelOwnerRgstCsDate='' or utilityModelOwnerRgstCsDate <= '{E}'""", c)
ow = ow.sort_values(["rgst_no", "seq"]).drop_duplicates("rgst_no", keep="last")
cur = dict(zip(ow.rgst_no, ow.nm.map(norm_name)))
P["g"] = [cur.get(r, "") or ("R" + r) for r in P.rgst_no]
pc = P[P.y == 1].g.value_counts()
out["owner_concentration"] = dict(n_owner_clusters_with_pos=int(len(pc)), top1_pos=int(pc.iloc[0]), top5_pos=int(pc.iloc[:5].sum()),
                                  top10_pos=int(pc.iloc[:10].sum()), pos_counts_top10=[int(v) for v in pc.iloc[:10]])
for kx in (1, 3, 5, 10):
    drop = set(pc.index[:kx]); S = P[~P.g.isin(drop)]
    a, b, s = auc(S.y, S.p_champ), auc(S.y, S.p_M0), auc(S.y, S.p_SB)
    out[f"excl_top{kx}_owners"] = dict(n=int(len(S)), pos=int(S.y.sum()), auc_champ=a, auc_M0=b, auc_SB=s,
                                       skill_vs_M0=skill(a, b), skill_vs_SB=skill(a, s),
                                       top1pct_capture=topk(S.y.values, S.p_champ.values, 0.01)[1] / S.y.sum())
# 한 권리자당 1건만 남기기(권리자 내 양성 중복 영향): 양성 군집별 최고 점수 양성 1건 + 모든 음성
kk = P[P.y == 1].sort_values("p_champ", ascending=False).drop_duplicates("g")
S = pd.concat([P[P.y == 0], kk])
a, b, s = auc(S.y, S.p_champ), auc(S.y, S.p_M0), auc(S.y, S.p_SB)
out["one_pos_per_owner_bestscore"] = dict(pos=int(S.y.sum()), auc_champ=a, skill_vs_M0=skill(a, b), skill_vs_SB=skill(a, s), note="양성 쪽에 유리한 선택(상한 쪽)")
kk = P[P.y == 1].sort_values("p_champ", ascending=True).drop_duplicates("g")
S = pd.concat([P[P.y == 0], kk])
a, b, s = auc(S.y, S.p_champ), auc(S.y, S.p_M0), auc(S.y, S.p_SB)
out["one_pos_per_owner_worstscore"] = dict(pos=int(S.y.sum()), auc_champ=a, skill_vs_M0=skill(a, b), skill_vs_SB=skill(a, s), note="양성 쪽에 불리한 선택(하한 쪽)")
kp = P[P.y == 1].ksic.value_counts()
out["ksic_pos"] = dict(n_ksic_with_pos=int(len(kp)), top3_share=float(kp.iloc[:3].sum() / kp.sum()))
out["all_event_patents"] = dict(n=V["n_event_patents"], in_population=V["pos_final"], outside=V["n_event_patents_outside_pop"],
                                top1pct_hits=int(P.sort_values("p_champ", ascending=False).head(3178).y.sum()))
json.dump(out, open(os.path.join(HERE, "v03_final_sensitivity.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(out, ensure_ascii=False, indent=1, default=float))
