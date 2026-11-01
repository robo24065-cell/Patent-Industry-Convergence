# -*- coding: utf-8 -*-
# C05-T3 verify_0 v2: E-C05T3-01(415행 '강한 기준선 수준') 반박 시도 — 공식 preds에서 상위 1% 직접 재계산
import pandas as pd, numpy as np, json, os, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
B=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
N9=os.path.join(B,r"모델경쟁_20260927\N9_정부RnD과제_특허성과예측")
V=os.path.join(B,r"검증루프_20261012\C05\C05-T3\verify_0")
pr=pd.read_parquet(os.path.join(N9,"final_preds_r6.parquet"))
o={'cols':list(pr.columns),'n':len(pr),'pos':int(pr.y.sum())}
y=pr.y.values; k=int(np.ceil(0.01*len(y)))
for c in [c for c in pr.columns if c.startswith('p_')]:
    s=pr[c].values
    top=np.argsort(-s,kind='stable')[:k]
    # 동점 민감도: 무작위 tie-break 50회
    rng=np.random.default_rng(0); alt=[]
    for _ in range(50):
        t=np.lexsort((rng.random(len(s)),-s))[:k]; alt.append(y[t].mean())
    o[c]={'top1':float(y[top].mean()),'top1_tie_min':float(min(alt)),'top1_tie_max':float(max(alt)),'hit100':int(y[np.argsort(-s,kind='stable')[:100]].sum())}
o['k']=k
json.dump(o,open(os.path.join(V,'v2_top1_names_out.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=1)
print(json.dumps(o,ensure_ascii=False,indent=1))
