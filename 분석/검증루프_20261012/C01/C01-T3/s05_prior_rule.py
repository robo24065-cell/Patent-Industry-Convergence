import os, sqlite3, numpy as np, pandas as pd, json
d=os.path.dirname(os.path.abspath(__file__))
H=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\N8_특허분쟁표적레이더"
REG=r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
c=sqlite3.connect(f"file:{REG}?mode=ro",uri=True)
P=pd.read_parquet(os.path.join(H,"final_preds_r6.parquet"),columns=["rgst_no","y","p_champ","p_SB"])
OT=["무효(심판)예고등록","권리범위(심판)예고등록","심판예고등록","정정(심판)예고등록","취소신청예고등록","이의신청예고등록","심결취소소송예고등록","취소(심판)예고등록","상고예고등록"]
q="select distinct rgst_no, rgstCsName from right where rgstCsDate < '20230101' and rgstCsName in (%s)"%",".join("'%s'"%x for x in OT)
h=pd.read_sql(q,c)
disp=set(h[h.rgstCsName.isin(OT[:2])].rgst_no); anyt=set(h.rgst_no)
out={}
for nm,s in (("prior_disp",disp),("prior_any_trial",anyt)):
    m=P.rgst_no.isin(s).values
    out[nm]=dict(n=int(m.sum()), share_pop=float(m.mean()), pos=int(P.y.values[m].sum()), capture=float(P.y.values[m].sum()/108), precision=float(P.y.values[m].mean()))
    k=int(m.sum()); o=np.argsort(-P.p_champ.values,kind="stable")[:k]
    out[nm]["champ_hits_same_k"]=int(P.y.values[o].sum())
    o3=np.argsort(-P.p_champ.values,kind="stable")[:3178]
    out[nm]["in_champ_top1pct"]=int(m[o3].sum())
json.dump(out,open(os.path.join(d,"s05_prior_rule.json"),"w",encoding="utf-8"),ensure_ascii=False,indent=1)
print(json.dumps(out,indent=1))
