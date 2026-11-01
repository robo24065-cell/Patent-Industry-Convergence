# -*- coding: utf-8 -*-
r"""C06-T1 verify_0: 독립 재계산(다른 조인 경로). DuckDB SQL로 DART\corp_applicant.parquet(법인번호 열) + dart_company.sqlite(ro, pandas 읽기)
+ R12 labels.parquet + pat_firm.parquet 를 직접 조인해 DART5 표본, 제공·H-mg-3 기업 평균, B1 점추정을 다시 계산한다.
s1~s3 캐시(firm_U, docs_U, firm_flags)는 쓰지 않는다. 규모는 R12 reg12.parquet n_emp(NPS 원천 재현 1.0 확인값)만 가져온다."""
import os, sys, json, sqlite3
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, duckdb
HERE = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"; MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
R12 = os.path.join(MC, r"_shared_report\label_origin_R10\confirm_R12"); LO = os.path.join(MC, r"_shared_report\label_origin_R10\cache")
f = lambda p: p.replace("\\", "/")
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
con = duckdb.connect(); con.execute("set memory_limit='2GB'; set threads=2")
c = sqlite3.connect("file:" + os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_company.sqlite") + "?mode=ro", uri=True)
dc = pd.read_sql("select jurir_no, induty_code from company", c); c.close()
con.register("dc", dc)
con.register("k50", pd.DataFrame({"i": range(50), "k": KSIC50}))
ca_cols = duckdb.sql(f"select * from read_parquet('{f(os.path.join(BASE, r'공모전데이터\외부데이터\DART\corp_applicant.parquet'))}') limit 1").columns
code_col, jur_col = ca_cols[0], ca_cols[2]   # 출원인코드, 법인번호(원문)
q = f"""
with ca as (select distinct cast("{code_col}" as varchar) applicantCd, replace(trim(cast("{jur_col}" as varchar)), '-', '') jurir
            from read_parquet('{f(os.path.join(BASE, r'공모전데이터\외부데이터\DART\corp_applicant.parquet'))}')),
     d as (select replace(trim(jurir_no), '-', '') jurir, trim(induty_code) ind from dc where trim(induty_code) <> ''),
     d1 as (select jurir, min(ind) ind, count(distinct ind) nind from d where length(jurir)=13 group by jurir),
     link as (select ca.applicantCd, d1.ind from ca join d1 using(jurir) where length(ca.jurir)=13),
     lb as (select cast(appl_no as varchar) appl_no, PROV, "H-mg-3" HB from read_parquet('{f(os.path.join(R12, 'cache', 'labels.parquet'))}') where split='eval'),
     pf as (select cast(applicantCd as varchar) applicantCd, cast(appl_no as varchar) appl_no from read_parquet('{f(os.path.join(LO, 'pat_firm.parquet'))}')),
     x as (select pf.applicantCd, lb.appl_no, lb.PROV, lb.HB from pf join lb using(appl_no))
select x.applicantCd, x.appl_no, link.ind r,
       cast(kp.k = link.ind as int) mP, cast(coalesce(kb.k = link.ind, false) as int) mB, x.PROV
from x join link using(applicantCd) left join k50 kp on kp.i = x.PROV left join k50 kb on kb.i = x.HB
where link.ind in (select k from k50)
"""
Z = con.sql(q).df()
Z["mP"] = Z.mP.fillna(0)
T = Z.groupby("applicantCd")[["mP", "mB"]].mean()
R = {"firms": len(T), "docs": len(Z), "PROV": round(float(T.mP.mean()), 4), "BEST": round(float(T.mB.mean()), 4), "diff": round(float(T.mP.mean() - T.mB.mean()), 4)}
# ---- B1 독립 구현: 모집단 = 평가 문서 있는 법인 전체
allx = con.sql(f"""select cast(pf.applicantCd as varchar) applicantCd, lb.PROV from read_parquet('{f(os.path.join(LO, 'pat_firm.parquet'))}') pf
                  join (select cast(appl_no as varchar) appl_no, PROV from read_parquet('{f(os.path.join(R12, 'cache', 'labels.parquet'))}') where split='eval') lb
                  on cast(pf.appl_no as varchar) = lb.appl_no""").df()
m = allx[allx.PROV >= 0].groupby(["applicantCd", "PROV"]).size().reset_index(name="c").sort_values(["applicantCd", "c", "PROV"], ascending=[True, False, True]).drop_duplicates("applicantCd")
pop = pd.DataFrame({"applicantCd": allx.applicantCd.unique()}).merge(m[["applicantCd", "PROV"]], how="left")
def grp(i):
    if pd.isna(i): return "미상"
    d2 = KSIC50[int(i)][:2]
    return {"26": "26", "27": "27", "28": "28", "29": "29-30", "30": "29-30", "20": "20-21", "21": "20-21"}.get(d2, "기타")
pop["g"] = pop.PROV.map(grp)
ne = pd.read_parquet(os.path.join(R12, "cache", "reg12.parquet"), columns=["applicantCd", "n_emp"]).drop_duplicates("applicantCd")
pop = pop.merge(ne, how="left")
def sz(v):
    if pd.isna(v): return "미상"
    return "<10" if v <= 9 else "10-49" if v <= 49 else "50-299" if v <= 299 else ">=300"
pop["s"] = pop.n_emp.map(sz); pop["cell"] = pop.s + "|" + pop.g
smp = pop[pop.applicantCd.isin(set(T.index))].copy()
cnt = smp.cell.value_counts()
bad = {c for c in pop.cell.unique() if cnt.get(c, 0) < 5}
for df in (pop, smp): df.loc[df.cell.isin(bad), "cell"] = df.loc[df.cell.isin(bad), "s"] + "|합침"
cnt2 = smp.cell.value_counts(); R["small_after_step1"] = {k: int(v) for k, v in cnt2.items() if v < 5}
R["pop_cells_not_in_sample"] = sorted(set(pop.cell) - set(smp.cell))
w = (pop.cell.value_counts(normalize=True) / smp.cell.value_counts(normalize=True))
smp["w"] = smp.cell.map(w); smp["w"] = smp.w / smp.w.mean(); smp["w"] = smp.w.clip(upper=20); smp["w"] = smp.w / smp.w.mean()
TT = T.join(smp.set_index("applicantCd").w)
R["B1_PROV"] = round(float(np.average(TT.mP, weights=TT.w)), 4); R["B1_BEST"] = round(float(np.average(TT.mB, weights=TT.w)), 4)
R["B1_diff"] = round(R["B1_PROV"] - R["B1_BEST"], 4); R["B1_cells"] = int(smp.cell.nunique())
ref = {"firms": 811, "diff": 0.0731, "B1_diff": 0.0618}
R["pass"] = bool(R["firms"] == 811 and abs(R["diff"] - ref["diff"]) <= 0.001 and abs(R["B1_diff"] - ref["B1_diff"]) <= 0.001)
json.dump(R, open(os.path.join(HERE, "v0_result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=True))
