import os, numpy as np, pandas as pd, json
d=os.path.dirname(os.path.abspath(__file__))
H=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N8_특허분쟁표적레이더"
P=pd.read_parquet(os.path.join(H,"final_preds_r6.parquet"),columns=["rgst_no","y","p_champ","p_M0","p_SB"])
L=pd.read_csv(os.path.join(d,"labels_rebuilt.csv"),comment="#",dtype={"rgst_no":str}).set_index("rgst_no")
k=3178; out={}
tops={}
for m in ["p_champ","p_SB","p_M0"]:
    o=np.argsort(-P[m].values,kind="stable")[:k]; tops[m]=set(P.rgst_no.values[o][P.y.values[o]==1])
for m,s in tops.items():
    sub=L.loc[sorted(s)]
    out[m]=dict(hits=len(s), prior=int(sub.prior_disp_pre2023.sum()), new=int((sub.prior_disp_pre2023==0).sum()),
               inv_only=int(((sub.n_inv_win>0)&(sub.n_sco_win==0)).sum()), scope_only=int(((sub.n_inv_win==0)&(sub.n_sco_win>0)).sum()), both=int(((sub.n_inv_win>0)&(sub.n_sco_win>0)).sum()),
               scope_dir=sub.sco_dir.value_counts().to_dict(), withdrawn=int(sub.withdrawn_after.sum()))
out["champ_and_SB"]=len(tops["p_champ"]&tops["p_SB"]); out["champ_not_SB"]=len(tops["p_champ"]-tops["p_SB"]); out["SB_not_champ"]=len(tops["p_SB"]-tops["p_champ"])
out["champ_not_SB_prior"]=int(L.loc[sorted(tops["p_champ"]-tops["p_SB"])].prior_disp_pre2023.sum())
out["prior_total"]=int(L.prior_disp_pre2023.sum())
# 적극 점유 20% 문턱: 권리범위 사건 양성 중 몇 %가 적극이어야 하나
nsc=int((L.n_sco_win>0).sum()); out["scope_pos"]=nsc; out["need_active_for_20pct"]=int(np.ceil(0.2*len(L))); out["need_share_of_scope_pos"]=float(np.ceil(0.2*len(L))/nsc)
# 대체 가정: 적극 인용률이 소극의 1/2, 1/3 일 때 미판정분 적극 확률
p0=0.15
for r in [1,0.5,1/3]:
    q=p0/(p0+(1-p0)*r)  # 인용 표본 적극비율 p0 -> 청구 표본 적극비율
    out[f"active_share_if_win_ratio_{r:.2f}"]=q; out[f"active_pos_est_if_{r:.2f}"]= 2 + 29*q
json.dump(out,open(os.path.join(d,"s04_toplist.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(json.dumps(out,ensure_ascii=False,indent=1))
