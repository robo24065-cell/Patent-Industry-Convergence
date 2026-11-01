# -*- coding: utf-8 -*-
r"""C11-T2 verify_3: C-LAND-free_F/txt_F/free_H/txt_H 정정 주장 반박 시도(제3 구현).
s7(pandas)·verify_1(row_number, 원천 family 파싱)과 다르게:
 - REG 연결: links.parquet를 쓰지 않고 label_origin_R10 fam_reg ∪ R8 fam_link CN 행으로 직접 구성(C10 s1 정의) 후 links.parquet REG와 대조.
 - TXT 풀 문서: C11 cache/txt_assign_reg를 쓰지 않고 R9 s01 규칙(풀 = R3 base_rows ∩ rec_conc 27코드, 복수 코드 해외 = 연결 KR 풀 문서 같은 코드 최빈
   → (코드,sg/mg/sc) ≥3 → 코드 최빈)을 duckdb arg_min(원 c1/s01 방식)으로 재구성.
 - FREE 표: arg_min 방식(원 c1).  ρ: scipy spearmanr. RTA는 SQL 집계.
쓰기: verify_3/v3_rows.csv, v3_out.json"""
import sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, duckdb
from scipy.stats import spearmanr
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"; T3 = MC + "/T3_기술셀부상예측"
W = BASE + "/분석/검증루프_20261012/C11/C11-T2"; C10 = BASE + "/분석/검증루프_20261012/C10/C10-T3"
ALT = MC + "/_shared_report/fusion_advantage/c5_R10/alt_assign/*.parquet"
RC = T3 + "/ESSENTIAL_R7/rec_conc.parquet"; KF = MC + "/T1_산업대기실/TXT_R9/cache/kr_txtfree.parquet"
TA = T3 + "/TXT_R9/txt_assign.parquet"; SRC = T3 + "/R3/cache_pub/base_rows.parquet"
FAM = MC + "/_ideas_R8/cache/fam_link.parquet"; FREG = MC + "/_shared_report/label_origin_R10/cache/fam_reg.parquet"
LINKS = C10 + "/cache/links.parquet"
CCS = ["KR", "US", "JP", "CN", "EP"]
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
con = duckdb.connect(); con.execute(f"set memory_limit='3GB'; set threads=2; set enable_progress_bar=false; set preserve_insertion_order=false; set temp_directory='{W}/verify_3/_tmp'")
km = pd.read_csv(T3 + "/ESSENTIAL_R7/conc_ksic_map.csv", dtype=str)[["ksic", "code"]]; con.register("km", km)
K50 = sorted(km.ksic); MULTI = sorted(km.code.value_counts().loc[lambda s: s > 1].index)
ML = ",".join(f"'{c}'" for c in MULTI)
MODE = "arg_min(k, -n*100000 + cast(k as int))"
R = {}
con.execute(f"""create table d as select a.cc, a.appl_no, a.ksic_prov, a.code, a.k_txt, a.k_free, a.m_txt, a.m_free, r.sg
   from read_parquet('{ALT}') a left join read_parquet('{RC}') r using(cc, appl_no) where a.is_inv and a.appl_year between 2017 and 2019""")
R["window"] = con.sql("select cc, count(*), count(distinct appl_no) from d group by 1 order by 1").fetchall(); log("창", R["window"])
# ---- 연결 ----
con.execute(f"create table l_r8 as select distinct cc, appl_no, kr_appl_no from read_parquet('{FAM}')")
con.execute(f"create table l_reg as select distinct cc, appl_no, kr_appl_no from read_parquet('{FREG}') union select * from l_r8 where cc='CN'")
con.execute(f"create table l_regC10 as select distinct cc, appl_no, kr_appl_no from read_parquet('{LINKS}') where link='REG'")
R["reg_vs_links"] = con.sql("""select (select count(*) from l_reg), (select count(*) from l_regC10),
   (select count(*) from (select * from l_reg intersect select * from l_regC10))""").fetchone()
R["links_in_window"] = {nm: con.sql(f"select l.cc, count(*), count(distinct l.appl_no) from {nm} l join d using(cc, appl_no) group by 1 order by 1").fetchall()
                        for nm in ("l_r8", "l_reg")}
R["r8_cn_years"] = con.sql(f"""select a.appl_year, count(*) from l_r8 l join read_parquet('{ALT}') a using(cc, appl_no) where l.cc='CN' group by 1 order by 1""").fetchall()
log("연결", R["reg_vs_links"], R["links_in_window"])
# ---- FREE ----
con.execute(f"create table kf as select appl_no, k_free k from read_parquet('{KF}') where k_free is not null")
R["kf_dup"] = con.sql("select count(*)-count(distinct appl_no) from kf").fetchone()[0]
con.execute(f"""create table krf as select r.sg, split_part(r.sg,'/',1) mg, left(r.sg,4) sc, kf.k from read_parquet('{ALT}') a join kf using(appl_no)
   left join read_parquet('{RC}') r using(cc, appl_no) where a.cc='KR' and a.appl_year<=2019""")
for lvl in ("sg", "mg", "sc"):
    con.execute(f"""create table fT_{lvl} as with g as (select {lvl} key_, k, count(*) n from krf where {lvl} is not null and {lvl}<>'' group by all)
       select key_, {MODE} k from g group by key_ having sum(n)>=3""")
con.execute("""create table ftab as select d.cc, d.appl_no, coalesce(a.k, b.k, c.k) k from d left join fT_sg a on a.key_=d.sg
   left join fT_mg b on b.key_=split_part(d.sg,'/',1) left join fT_sc c on c.key_=left(d.sg,4) where d.cc<>'KR'""")
R["free_tab_vs_stored_ipc"] = con.sql("select avg((f.k is not distinct from d.k_free)::int), count(*) from ftab f join d using(cc,appl_no) where d.m_free like 'ipc_%' or d.m_free='none'").fetchone()
def free_fam(link):
    con.execute(f"""create or replace table ffam as with g as (select l.cc, l.appl_no, kf.k, count(*) n from {link} l join d using(cc, appl_no) join kf on kf.appl_no=l.kr_appl_no group by all)
       select cc, appl_no, {MODE} k from g group by 1,2""")
# ---- TXT 풀(R9 s01 재구성) ----
con.execute(f"""create table r as select b.cc, b.appl_no, rc.code, rc.sg from read_parquet('{SRC}') b join read_parquet('{RC}') rc using(cc, appl_no) where rc.code in (select code from km)""")
con.execute(f"create table krt as select appl_no, code, ksic_txt k, \"method\" m from read_parquet('{TA}') where cc='KR'")
R["pool"] = con.sql("select cc, count(*) from r group by 1 order by 1").fetchall()
con.execute("""create table krk as select k.code, k.k, r.sg, split_part(r.sg,'/',1) mg, left(r.sg,4) sc from krt k join r on r.cc='KR' and r.appl_no=k.appl_no where k.m='text'""")
for lvl in ("sg", "mg", "sc"):
    con.execute(f"""create table tT_{lvl} as with g as (select code, {lvl} key_, k, count(*) n from krk where {lvl} is not null and {lvl}<>'' group by all)
       select code, key_, {MODE} k from g group by code, key_ having sum(n)>=3""")
con.execute(f"create table tT_code as with g as (select code, k, count(*) n from krk group by all) select code, {MODE} k from g group by code")
# 창 안 해외 풀 문서만
con.execute(f"create table rw as select r.* from r join d using(cc, appl_no) where r.cc<>'KR'")
con.execute(f"""create table ttab as select rw.cc, rw.appl_no, rw.code, coalesce(a.k, b.k, c.k, e.k) k from rw
   left join tT_sg a on a.code=rw.code and a.key_=rw.sg left join tT_mg b on b.code=rw.code and b.key_=split_part(rw.sg,'/',1)
   left join tT_sc c on c.code=rw.code and c.key_=left(rw.sg,4) left join tT_code e on e.code=rw.code where rw.code in ({ML})""")
def txt_pool(link):
    famq = "select null::varchar cc, null::varchar appl_no, null::varchar k where false" if link is None else f"""with g as (
        select l.cc, l.appl_no, k.k, count(*) n from {link} l join rw on rw.cc=l.cc and rw.appl_no=l.appl_no join krt k on k.appl_no=l.kr_appl_no and k.code=rw.code
        where rw.code in ({ML}) group by all) select cc, appl_no, {MODE} k from g group by 1,2"""
    con.execute(f"create or replace table tfam as {famq}")
    con.execute(f"""create or replace table tpool as select rw.cc, rw.appl_no, case when rw.code in ({ML}) then coalesce(f.k, t.k) else s.ksic end k
        from rw left join tfam f using(cc, appl_no) left join ttab t using(cc, appl_no)
        left join (select code, min(ksic) ksic from km group by 1 having count(*)=1) s on s.code=rw.code""")
def labels(link):
    if link is None:
        con.execute("create or replace table ffam as select null::varchar cc, null::varchar appl_no, null::varchar k where false")
    else:
        free_fam(link)
    txt_pool(link)
    return con.sql("""select d.cc, d.appl_no, d.ksic_prov,
        case when d.cc='KR' then d.k_free else coalesce(ff.k, ft.k) end kf_,
        case when d.cc='KR' then d.k_txt when tp.appl_no is not null then tp.k else d.k_txt end kt_,
        (tp.appl_no is not null) inpool
        from d left join ffam ff using(cc, appl_no) left join ftab ft using(cc, appl_no) left join tpool tp using(cc, appl_no)""").df()

def rta(L, kr_col, for_col):
    n = pd.DataFrame({cc: L.loc[L.cc == cc, kr_col if cc == "KR" else for_col].value_counts().reindex(K50).fillna(0) for cc in CCS})
    s = n / n.sum(); return s.div(s.mean(axis=1), axis=0)
rows = []; LAB = {}
for nm, link in (("r8", "l_r8"), ("reg", "l_reg"), ("regC10", "l_regC10"), ("pure", None)):
    L = labels(link); LAB[nm] = L
    prov = rta(L, "ksic_prov", "ksic_prov"); lp = prov.idxmax(axis=1); kp = set(lp.index[lp == "KR"])
    for scen in ("F", "H"):
        for lab, col in (("free", "kf_"), ("txt", "kt_")):
            rt = rta(L, col if scen == "F" else "ksic_prov", col); ld = rt.idxmax(axis=1); krl = set(ld.index[ld == "KR"])
            rho = float(spearmanr(rt.KR, prov.KR).statistic)
            rows.append(dict(arm=nm, item=f"{lab}_{scen}", kr_rta_gt1=int((rt.KR > 1).sum()), kr_leader=len(krl), kept=len(krl & kp), rho=round(rho, 3), rho_raw=rho))
    log(nm)
# 저장값 대조(r8 = alt_assign 저장 라벨)
L8 = LAB["r8"]; o = L8[L8.cc != "KR"].merge(con.sql("select cc, appl_no, k_free, k_txt from d where cc<>'KR'").df(), on=["cc", "appl_no"])
R["r8_free_eq_stored"] = float((o.kf_.fillna("NA") == o.k_free.fillna("NA")).mean()); R["r8_txt_eq_stored"] = float((o.kt_.fillna("NA") == o.k_txt.fillna("NA")).mean())
# reg TXT 풀 라벨 vs C11 txt_assign_reg (참고)
c11 = con.sql(f"select cc, appl_no, ksic_txt from read_parquet('{W}/cache/txt_assign_reg.parquet') where cc<>'KR'").df()
Lr = LAB["reg"]; m = Lr[Lr.inpool & (Lr.cc != "KR")].merge(c11, on=["cc", "appl_no"], how="left")
R["reg_txt_pool_eq_C11"] = [float((m.kt_ == m.ksic_txt).mean()), int(len(m)), int(m.ksic_txt.isna().sum())]
# 변경 레코드 수(해외 창)
for nm in ("reg", "pure"):
    a = LAB[nm][LAB[nm].cc != "KR"].set_index(["cc", "appl_no"]); b = L8[L8.cc != "KR"].set_index(["cc", "appl_no"]).reindex(a.index)
    R[f"changed_{nm}"] = dict(free=int((a.kf_.fillna("NA") != b.kf_.fillna("NA")).sum()), txt=int((a.kt_.fillna("NA") != b.kt_.fillna("NA")).sum()))
T = pd.DataFrame(rows); T.to_csv(W + "/verify_3/v3_rows.csv", index=False, encoding="utf-8-sig"); print(T.to_string())
R["rows"] = rows
json.dump(R, open(W + "/verify_3/v3_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: v for k, v in R.items() if k != "rows"}, ensure_ascii=False, default=str)); log("end")
