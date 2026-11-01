import os, json, pandas as pd
d=os.path.dirname(os.path.abspath(__file__))
pt=json.load(open(os.path.join(d,"s02_point.json"),encoding="utf-8"))
cis={s:json.load(open(os.path.join(d,f"s02_{s}.json"),encoding="utf-8")) for s in ["owner","bundle","bundle30"]}
desc={"L_all":"공식(무효∪권리범위) 재현","L_inv":"무효 단독(권리범위만 양성 제외, 사전등록 주분석)","L_inv_negs":"무효 단독(권리범위만 양성=음성)",
"L_scope":"권리범위 단독(무효만 양성 제외)","L_3p":"제3자 추정(무효∪소극판정∪미판정 적극확률<0.5; 사전등록)","L_3p_strict":"무효∪소극 판정만(사후 추가)","L_all_no_prior":"2023 전 분쟁 이력 없는 양성만(참고)"}
rows=[]
for lab,v in pt.items():
    for mt in ["auc_champ","auc_M0","auc_SB","skill_M0","skill_SB","cap_champ","cap_M0","cap_SB","dcap_M0","dcap_SB"]:
        r=dict(label=lab,label_desc=desc[lab],n_eval=v["n_eval"],pos=v["pos"],k1pct=v["k1pct"],metric=mt,point=round(v[mt],4))
        for s in cis:
            r[f"ci_{s}_lo"]=round(cis[s][lab][mt][0],4); r[f"ci_{s}_hi"]=round(cis[s][lab][mt][1],4)
        rows.append(r)
    for mm in ["champ","M0","SB"]:
        rows.append(dict(label=lab,label_desc=desc[lab],n_eval=v["n_eval"],pos=v["pos"],k1pct=v["k1pct"],metric=f"hits1pct_{mm}",point=v[f"hits1pct_{mm}"]))
        rows.append(dict(label=lab,label_desc=desc[lab],n_eval=v["n_eval"],pos=v["pos"],k1pct=v["k1pct"],metric=f"hit100_{mm}",point=v[f"hit100_{mm}"]))
df=pd.DataFrame(rows); df["B"]="owner=%d bundle=%d bundle30=%d"%(cis["owner"]["B_done"],cis["bundle"]["B_done"],cis["bundle30"]["B_done"])
df.to_csv(os.path.join(d,"rescoring_sensitivity.csv"),index=False,encoding="utf-8-sig"); print(len(df))
