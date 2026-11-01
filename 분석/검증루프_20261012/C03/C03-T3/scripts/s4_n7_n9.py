# C03-T3 (C): perform_grp 매핑 오류의 N7·N9 민감도. 재학습 없음. 원천 parquet 읽기 전용.
import sys, json, time; sys.path.insert(0, sys.argv[1]); sys.stdout.reconfigure(encoding='utf-8')
from _paths import *
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score as AUC
t0c=time.time()
R=pd.read_parquet(EXT+r"\ntis_project_raw.parquet",columns=["pjt_no","year","corp_rgst_no","perform_agent_cd","perform_agent","perform_grp","secret_flag","gov_funds"])
R["year"]=R.year.astype(int)
out={}
# 1) 코드-텍스트 대조(교정 매핑의 독립 확인)
ct=R.groupby(["perform_agent_cd","perform_agent","perform_grp"],dropna=False).size().reset_index(name="n").sort_values(["perform_agent_cd","n"],ascending=[True,False])
out["code_text_grp"]=[dict(cd=str(a),text=str(b),grp=str(c),n=int(n)) for a,b,c,n in ct.values if n>=50]
GRP_WRONG={"01":"국공립연","02":"출연연","03":"대학","04":"대기업","05":"중소기업","06":"중견기업","07":"정부부처","08":"기타","99":"기타"}
GRP_CORR=dict(GRP_WRONG); GRP_CORR["06"]="정부부처"; GRP_CORR["08"]="중견기업"
chk=R.perform_agent_cd.map(GRP_WRONG).fillna("x")
out["stored_grp_equals_wrong_map_share_nonnull"]=float((chk[R.perform_agent_cd.notna()]==R.perform_grp[R.perform_agent_cd.notna()]).mean())
R["grp_corr"]=R.perform_agent_cd.map(GRP_CORR)
R.loc[R.grp_corr.isna(),"grp_corr"]=np.where(R.loc[R.grp_corr.isna(),"secret_flag"],"보안과제","기타")
FIRM={"대기업","중소기업","중견기업"}
R["bz"]=R.corp_rgst_no.astype("string").str.replace(r"[^0-9]","",regex=True)
# 2) N7 공식 정의 재구성: 과제연도 2010~, 사업자번호 첫 등장 연도 (perform_grp 미사용)
def first_year(df):
    d=df[(df.year>=2010)&df.bz.notna()&(df.bz!="")]
    return d.groupby("bz").year.min()
F_all=first_year(R[R.year<=2024])
F_firm_wrong=first_year(R[(R.year<=2024)&R.perform_grp.isin(FIRM)])
F_firm_corr=first_year(R[(R.year<=2024)&R.grp_corr.isin(FIRM)])
P=pd.read_parquet(MC+r"\N7_특허기업_정부RnD첫수행\final_test_pred_r5.parquet")
P["bz"]=P.bizno.astype("string").str.replace(r"[^0-9]","",regex=True)
fa=P.bz.map(F_all)
P["y_recon"]=fa.between(2023,2024).astype(float)
out["N7_final"]=dict(n=int(len(P)),pos_frozen=int(P.y.sum()),pos_recon=int(P.y_recon.sum()),
    label_match=float((P.y==P.y_recon).mean()),pop_violation_first_le_2022=int((fa<=2022).sum()))
# 공식 정의는 grp를 쓰지 않으므로 교정 매핑 후에도 동일: 매핑을 바꿔 다시 계산해 확인
R2=R.copy(); R2["perform_grp"]=R2.grp_corr
F_all2=first_year(R2[R2.year<=2024])
out["N7_final"]["official_first_year_identical_after_correction"]=bool(F_all.equals(F_all2))
# 양성의 첫 진입 과제 코드 구성
R23=R[(R.year.between(2023,2024))&R.bz.isin(set(P.bz[P.y==1]))]
fy=pd.Series(P.bz[P.y==1].map(F_all).values,index=P.bz[P.y==1].values); fy.index.name="bz"
first_rows=R23.merge(fy.rename("fy").reset_index(),on="bz"); first_rows=first_rows[first_rows.year==first_rows.fy]
by_firm=first_rows.groupby("bz").perform_agent_cd.agg(lambda s: "|".join(sorted(set(s.dropna().astype(str)))))
out["N7_final"]["positives_entry_code_mix"]=by_firm.value_counts().head(12).to_dict()
out["N7_final"]["positives_entering_only_via_code08"]=int((by_firm=="08").sum())
out["N7_final"]["positives_entering_only_via_code06"]=int((by_firm=="06").sum())
# 3) 민감도: '기업 유형 과제로 첫 진입' 대안 정의(비공식) — 틀린 매핑 vs 교정 매핑
def score(y, df):
    y=np.asarray(y,int); r={}
    for c in ["p_champ","p_M0","p_SB"]: r["auc_"+c[2:]]=float(AUC(y,df[c]))
    r["skill_vs_M0"]=(r["auc_champ"]-r["auc_M0"])/(1-r["auc_M0"]); r["skill_vs_SB"]=(r["auc_champ"]-r["auc_SB"])/(1-r["auc_SB"])
    for k in (123,614,1229):
        o=np.argsort(-df.p_champ.values,kind="stable")[:k]; r[f"champ_top{k}_hits"]=int(y[o].sum())
        o=np.argsort(-df.p_M0.values,kind="stable")[:k]; r[f"M0_top{k}_hits"]=int(y[o].sum())
    r["pos"]=int(y.sum()); return r
res={"official_frozen":score(P.y,P)}
for nm,F in [("alt_firm_wrong",F_firm_wrong),("alt_firm_corr",F_firm_corr)]:
    f=P.bz.map(F); yy=f.between(2023,2024).astype(int); res[nm]=score(yy,P); res[nm]["pop_firm_project_le2022"]=int((f<=2022).sum())
    res[nm]["sym_diff_vs_official"]=int((yy!=P.y.astype(int)).sum())
yw=P.bz.map(F_firm_wrong).between(2023,2024); yc=P.bz.map(F_firm_corr).between(2023,2024)
res["alt_wrong_vs_corr_sym_diff"]=int((yw!=yc).sum())
out["N7_rescore"]=res
# 공식 CI 폭
fr=json.load(open(MC+r"\N7_특허기업_정부RnD첫수행\final_result_r5.json",encoding="utf8"))["results"][0]
out["N7_official_ci"]=dict(auc=fr["auc_champ_ci95"],skill_M0=fr["skill_vs_M0_ci95"],skill_SB=fr["skill_vs_SB_ci95"])
# 4) 개발 원점 양성률(공식 정의 vs 대안 정의, 매핑별)
D=pd.read_parquet(MC+r"\N7_특허기업_정부RnD첫수행\cache\final_panel.parquet")
bzcol=[c for c in D.columns if c=="bizno"]
dev={}
if bzcol:
    D["bz"]=D.bizno.astype("string").str.replace(r"[^0-9]","",regex=True)
    Rd=R[R.year<=2022]
    Fa=first_year(Rd); Fw=first_year(Rd[Rd.perform_grp.isin(FIRM)]); Fc=first_year(Rd[Rd.grp_corr.isin(FIRM)])
    for t0,g in D[D.t0<=2020].groupby("t0"):
        ya=g.bz.map(Fa).between(t0+1,t0+2); yw_=g.bz.map(Fw).between(t0+1,t0+2); yc_=g.bz.map(Fc).between(t0+1,t0+2)
        dev[int(t0)]=dict(n=int(len(g)),rate_frozen=float(g.y.mean()),rate_recon_official=float(ya.mean()),match=float((ya.astype(float)==g.y).mean()),
                         rate_alt_wrong=float(yw_.mean()),rate_alt_corr=float(yc_.mean()),alt_diff_pp=float(100*(yc_.mean()-yw_.mean())))
out["N7_dev_origins"]=dev
# 5) N9 '기업' 하위집단 교정 재채점
N=pd.read_parquet(MC+r"\N9_정부RnD과제_특허성과예측\final_preds_r6.parquet")
R22=R[R.year==2022].drop_duplicates("pjt_no").set_index("pjt_no")
N=N.join(R22[["perform_agent_cd","perform_grp","grp_corr","secret_flag"]],on="pjt_no")
out["N9"]=dict(n=int(len(N)),unmatched=int(N.perform_grp.isna().sum()),secret_in_test=int(N.secret_flag.fillna(False).sum()),
              grp_missing=int(N.perform_grp.isna().sum()))
def sub(m):
    y=N.y.values[m]; a1=AUC(y,N.p_champ.values[m]); a0=AUC(y,N.p_M0_L2.values[m])
    k=max(1,int(round(0.01*m.sum()))); o=np.argsort(-N.p_champ.values[m],kind="stable")[:k]
    return dict(n=int(m.sum()),pos=int(y.sum()),auc=float(a1),auc_M0L2=float(a0),skill=float((a1-a0)/(1-a0)),top1=float(y[o].mean()))
mw=N.perform_grp.astype(str).isin(FIRM).values; mc=N.grp_corr.astype(str).isin(FIRM).values
out["N9"]["firm_wrong"]=sub(mw); out["N9"]["firm_corr"]=sub(mc)
out["N9"]["moved_in_code08"]=int((mc&~mw).sum()); out["N9"]["moved_out_code06"]=int((mw&~mc).sum())
out["N9"]["moved_in_pos"]=int(N.y.values[mc&~mw].sum()); out["N9"]["moved_out_pos"]=int(N.y.values[mw&~mc].sum())
mi=(N.perform_agent_cd.astype(str)=="08").values
if N.y.values[mi].sum()>=10: out["N9"]["code08_only"]=sub(mi)
fr9=json.load(open(MC+r"\N9_정부RnD과제_특허성과예측\final_result_r6.json",encoding="utf8"))
out["N9"]["official_subgroup_firm"]=fr9["subgroups"]["기업"]
out["elapsed_s"]=round(time.time()-t0c,1)
json.dump(out,open(OUT+r"\_s4_raw.json","w",encoding="utf8"),ensure_ascii=False,indent=1,default=str)
print(json.dumps(out,ensure_ascii=False,indent=1,default=str)[:9000])
