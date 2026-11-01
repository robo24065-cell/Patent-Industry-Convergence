# -*- coding: utf-8 -*-
import json, os, sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
W=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T1"
O=os.path.join(W,"verify_1")
fi=pd.DataFrame.from_dict(json.load(open(os.path.join(O,"v1_first.json"))),orient="index"); fi.index.name="appl_no"; fi=fi.reset_index()
v0=pd.read_parquet(os.path.join(W,"verify_0","v0_refl.parquet"))
m=v0.dropna(subset=["refl_strict"]).merge(fi,on="appl_no",how="left")
res={"v0_strict_n":int(len(m)),"v0_strict_missing_in_v1":int(m.refl.isna().sum()),"v0_strict_refl_mismatch":int((m.refl!=m.refl_strict.astype("int64")).sum())}
rows=[]
for t in [2012,2013,2014,2015,2016,2017,2018,2019,2020,2022]:
    p=pd.read_parquet(os.path.join(W,"cache","panels",f"off_{t}.parquet"),columns=["appl_no","y"])
    q=p[p.y==1].merge(fi,on="appl_no",how="left")
    inwin=q.refl.between((t+1)*10000,(t+2)*10000+1231)
    rows.append(dict(t0=t,n=len(p),pos=len(q),pos_unlinked=int(q.refl.isna().sum()),pos_refl_in_window=int(inwin.sum()),
        g5_first=int(q.g5.fillna(0).sum()),g4_first=int(q.g4.fillna(0).sum()),nonblank_first=int(q.nonblank.fillna(0).sum()),any5ever=int(q.any5ever.fillna(0).sum())))
D=pd.DataFrame(rows)
D["g5_share"]=(D.g5_first/D.pos).round(4); D["g5_share_nonblank"]=(D.g5_first/D.nonblank_first).round(4); D["g4_share"]=(D.g4_first/D.pos).round(4)
tr=D[D.t0<=2020]; te=D[D.t0==2022]
res["train_pooled_pos"]=int(tr.pos.sum()); res["train_pooled_g5_share"]=round(tr.g5_first.sum()/tr.pos.sum(),4)
res["train_pooled_g4_share"]=round(tr.g4_first.sum()/tr.pos.sum(),4)
res["train_2018_2020_g5_share"]=round(tr[tr.t0>=2018].g5_first.sum()/tr[tr.t0>=2018].pos.sum(),4)
res["test_g5_share"]=float(te.g5_share.iloc[0]); res["test_g4_share"]=float(te.g4_share.iloc[0])
res["diff_pp"]=round(100*(res["test_g5_share"]-res["train_pooled_g5_share"]),1)
D.to_csv(os.path.join(O,"v1_n5_by_origin.csv"),index=False,encoding="utf-8-sig")
json.dump(res,open(os.path.join(O,"v1_n5.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(D.to_string()); print(json.dumps(res,ensure_ascii=False,indent=1))
