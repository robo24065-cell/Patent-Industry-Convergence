# -*- coding: utf-8 -*-
r"""C11-T2 verify_1: 지형도 ρ(168행 '0.19–0.37' 정정판, C-LAND-*_F/H) 반박 시도.
작업자 s7·C10 s3_rho·verify_5와 다른 경로:
 - 연결을 links.parquet가 아니라 원천 _shared/meta 'family' 필드에서 직접 파싱(US·JP·EP: KR(10|20)+7자리 등록번호 토큰 ↔ KR rgst_no 앞 9자리,
   CN: KR+13자리 출원번호 토큰 ↔ KR appl_no). → REG_raw 팔.
 - 최빈·표는 SQL row_number()(작업자는 pandas sort/arg_min).
 - Spearman은 pandas rank(average) + 피어슨, RTA 동률 1위 점검, 반올림 경로(raw→2자리 1회 vs 3자리→2자리) 비교.
 - 추가 반박: (i) 링크 KR 문서 다수일 때 최빈 대신 '첫 KR 문서(출원번호 최소)' 상속, (ii) 최빈 동률을 코드 큰 쪽으로, (iii) FREE 표 임계 ≥3 → ≥1.
쓰기: verify_1/v1_out.json, v1_rows.csv"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, duckdb
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"; T3 = MC + "/T3_기술셀부상예측"
W = BASE + "/분석/검증루프_20261012/C11/C11-T2"; C10 = BASE + "/분석/검증루프_20261012/C10/C10-T3"
META = MC + "/_shared/meta/*.parquet"; ALT = MC + "/_shared_report/fusion_advantage/c5_R10/alt_assign/*.parquet"
RC = T3 + "/ESSENTIAL_R7/rec_conc.parquet"; KF = MC + "/T1_산업대기실/TXT_R9/cache/kr_txtfree.parquet"; TA = T3 + "/TXT_R9/txt_assign.parquet"
CCS = ["KR", "US", "JP", "CN", "EP"]
t = time.time(); log = lambda *a: print(*a, f"[{time.time()-t:.0f}s]", flush=True)
con = duckdb.connect(); con.execute(f"set memory_limit='3GB'; set threads=2; set enable_progress_bar=false; set preserve_insertion_order=false; set temp_directory='{W}/cache/_tmp'")
km = pd.read_csv(T3 + "/ESSENTIAL_R7/conc_ksic_map.csv", dtype=str)[["ksic", "code"]]; con.register("km", km)
K50 = sorted(km.ksic); multi = tuple(km.code.value_counts().loc[lambda s: s > 1].index)
R = {}
# 창 문서(발명 2017–19): alt_assign 정의 그대로(is_inv), 라벨 저장값 + sg
con.execute(f"""create table d as select a.cc, a.appl_no, a.ksic_prov, a.code, a.k_txt, a.k_free, a.m_free, a.m_txt, r.sg from read_parquet('{ALT}') a
    left join read_parquet('{RC}') r using(cc, appl_no) where a.is_inv and a.appl_year between 2017 and 2019""")
# ---- 원천 family 파싱 ----
con.execute(f"create table krm as select appl_no kr_appl_no, left(rgst_no, 9) rg9 from read_parquet('{META}') where cc='KR'")
con.execute(f"""create table tok as select m.cc, m.appl_no, unnest(string_split(m.family, '|')) tk from read_parquet('{META}') m
    join (select cc, appl_no from d where cc<>'KR') w using(cc, appl_no) where m.family like '%KR%'""")
con.execute("""create table lraw as
   select distinct t.cc, t.appl_no, k.kr_appl_no from tok t join krm k on k.rg9 = substr(t.tk, 3, 9)
     where t.cc in ('US','JP','EP') and regexp_full_match(t.tk, 'KR(10|20)[0-9]{7}')
   union
   select distinct t.cc, t.appl_no, k.kr_appl_no from tok t join krm k on k.kr_appl_no = substr(t.tk, 3)
     where t.cc='CN' and regexp_full_match(t.tk, 'KR(10|20)[0-9]{11}')""")
R["raw_links_window"] = con.sql("select cc, count(*), count(distinct appl_no) from lraw group by 1 order by 1").fetchall()
# links.parquet REG(창 문서)와 쌍 겹침
con.execute(f"create table lreg as select l.cc, l.appl_no, l.kr_appl_no from read_parquet('{C10}/cache/links.parquet') l join (select cc, appl_no from d where cc<>'KR') w using(cc, appl_no) where l.link='REG'")
R["raw_vs_links_REG_pairs"] = con.sql("""select coalesce(a.cc,b.cc) cc, count(a.appl_no) n_raw, count(b.appl_no) n_reg, sum((a.appl_no is not null and b.appl_no is not null)::int) both_
   from (select distinct * from lraw) a full join (select distinct * from lreg) b using(cc, appl_no, kr_appl_no) group by 1 order by 1""").fetchall()
log("링크", R["raw_links_window"], R["raw_vs_links_REG_pairs"])
con.execute(f"create table kf as select appl_no, k_free k from read_parquet('{KF}') where k_free is not null")
con.execute(f"create table ta as select cc, appl_no, code, ksic_txt, \"method\" m from read_parquet('{TA}')")
# ---- FREE 표(KR ≤2019) by row_number ----
con.execute(f"""create table krf as select r.sg, split_part(r.sg,'/',1) mg, left(r.sg,4) sc, kf.k from read_parquet('{ALT}') a join kf using(appl_no)
    left join read_parquet('{RC}') r using(cc, appl_no) where a.cc='KR' and a.appl_year<=2019""")
def free_tab(minn, desc_tie=False):
    o = "k desc" if desc_tie else "cast(k as int)"
    for lvl in ("sg", "mg", "sc"):
        con.execute(f"""create or replace table ft_{lvl} as select key_, k from (select key_, k, row_number() over (partition by key_ order by n desc, {o}) rn, sum(n) over (partition by key_) tot
            from (select {lvl} key_, k, count(*) n from krf where {lvl} is not null and {lvl}<>'' group by all)) where rn=1 and tot>={minn}""")
    con.execute("""create or replace table fr as select d.cc, d.appl_no, coalesce(a.k, b.k, c.k) k from d left join ft_sg a on a.key_=d.sg
        left join ft_mg b on b.key_=split_part(d.sg,'/',1) left join ft_sc c on c.key_=left(d.sg,4) where d.cc<>'KR'""")
free_tab(3)
R["free_tab_eq_stored_nonfamily"] = con.sql("select avg((fr.k is not distinct from d.k_free)::int) from fr join d using(cc, appl_no) where d.m_free<>'family'").fetchone()[0]
# ---- TXT 표(코드 안, KR text 문서) ----
con.execute("""create table krt as select ta.code, r.sg, split_part(r.sg,'/',1) mg, left(r.sg,4) sc, ta.ksic_txt k from ta
    join read_parquet('""" + RC + """') r using(cc, appl_no) where ta.cc='KR' and ta.m='text'""")
for lvl in ("sg", "mg", "sc"):
    con.execute(f"""create table tt_{lvl} as select code, key_, k from (select code, key_, k, row_number() over (partition by code, key_ order by n desc, cast(k as int)) rn,
        sum(n) over (partition by code, key_) tot from (select code, {lvl} key_, k, count(*) n from krt where {lvl} is not null and {lvl}<>'' group by all)) where rn=1 and tot>=3""")
con.execute("create table tt_code as select code, k from (select code, k, row_number() over (partition by code order by n desc, cast(k as int)) rn from (select code, k, count(*) n from krt group by all)) where rn=1")
con.execute("""create table txtab as select ta.cc, ta.appl_no, ta.code, ta.ksic_txt k_stored, ta.m, coalesce(a.k, b.k, c.k, e.k) k_tab
   from ta join d using(cc, appl_no) left join tt_sg a on a.code=ta.code and a.key_=d.sg left join tt_mg b on b.code=ta.code and b.key_=split_part(d.sg,'/',1)
   left join tt_sc c on c.code=ta.code and c.key_=left(d.sg,4) left join tt_code e on e.code=ta.code where ta.cc<>'KR'""")
R["txt_tab_eq_stored_nonfamily_multi"] = con.sql(f"select avg((k_tab=k_stored)::int) from txtab where m<>'family' and code in {multi} ").fetchone()[0]
log("표 점검", R["free_tab_eq_stored_nonfamily"], R["txt_tab_eq_stored_nonfamily_multi"])

def arm_labels(link_tbl, rule="mode", desc_tie=False):
    o = "k desc" if desc_tie else "cast(k as int)"
    if link_tbl is None:
        con.execute("create or replace table ff as select null::varchar cc, null::varchar appl_no, null::varchar k where false")
        con.execute("create or replace table ftx as select null::varchar cc, null::varchar appl_no, null::varchar k where false")
    elif rule == "mode":
        con.execute(f"""create or replace table ff as select cc, appl_no, k from (select cc, appl_no, k, row_number() over (partition by cc, appl_no order by n desc, {o}) rn
            from (select l.cc, l.appl_no, kf.k, count(*) n from {link_tbl} l join kf on kf.appl_no=l.kr_appl_no group by all)) where rn=1""")
        con.execute(f"""create or replace table ftx as select cc, appl_no, k from (select cc, appl_no, k, row_number() over (partition by cc, appl_no order by n desc, {o}) rn
            from (select l.cc, l.appl_no, kt.ksic_txt k, count(*) n from {link_tbl} l join txtab x on x.cc=l.cc and x.appl_no=l.appl_no
                  join ta kt on kt.cc='KR' and kt.appl_no=l.kr_appl_no and kt.code=x.code where x.code in {multi} group by all)) where rn=1""")
    else:   # first: 출원번호 가장 작은 KR 문서 하나
        con.execute(f"""create or replace table ff as select cc, appl_no, k from (select l.cc, l.appl_no, kf.k, row_number() over (partition by l.cc, l.appl_no order by l.kr_appl_no) rn
            from {link_tbl} l join kf on kf.appl_no=l.kr_appl_no) where rn=1""")
        con.execute(f"""create or replace table ftx as select cc, appl_no, k from (select l.cc, l.appl_no, kt.ksic_txt k, row_number() over (partition by l.cc, l.appl_no order by l.kr_appl_no) rn
            from {link_tbl} l join txtab x on x.cc=l.cc and x.appl_no=l.appl_no join ta kt on kt.cc='KR' and kt.appl_no=l.kr_appl_no and kt.code=x.code where x.code in {multi}) where rn=1""")
    return con.sql(f"""select d.cc, d.appl_no, d.ksic_prov,
        case when d.cc='KR' then d.k_free else coalesce(ff.k, fr.k) end k_free_arm,
        case when d.cc='KR' then d.k_txt when x.appl_no is null then d.k_txt when x.code in {multi} then coalesce(ftx.k, case when x.m='family' then x.k_tab else x.k_stored end)
             else x.k_stored end k_txt_arm
        from d left join ff using(cc, appl_no) left join fr using(cc, appl_no) left join txtab x using(cc, appl_no) left join ftx using(cc, appl_no)""").df()

def rta(lab_kr, lab_for, L):
    n = pd.DataFrame(0.0, index=K50, columns=CCS)
    for cc in CCS:
        s = L.loc[L.cc == cc, lab_kr if cc == "KR" else lab_for].dropna()
        n[cc] = s.value_counts().reindex(K50).fillna(0)
    sh = n / n.sum(axis=0); return sh.div(sh.mean(axis=1), axis=0)

def sp(a, b):
    return float(np.corrcoef(a.rank(method="average"), b.rank(method="average"))[0, 1])

rows = []
def land(name, L):
    prov = rta("ksic_prov", "ksic_prov", L); lp = prov.idxmax(axis=1); kp = set(lp.index[lp == "KR"])
    for scen, lk in (("F", None), ("H", "ksic_prov")):
        for lab in ("free", "txt"):
            rt = rta(lk or f"k_{lab}_arm", f"k_{lab}_arm", L); ld = rt.idxmax(axis=1); krl = set(ld.index[ld == "KR"])
            ties = int((rt.eq(rt.max(axis=1), axis=0).sum(axis=1) > 1).sum())
            rows.append(dict(arm=name, item=f"{lab}_{scen}", kr_rta_gt1=int((rt.KR > 1).sum()), kr_leader=len(krl), kept=len(krl & kp), rho_raw=sp(rt.KR, prov.KR), leader_ties=ties))
    return prov

arms = {"r8": f"(select cc, appl_no, kr_appl_no from read_parquet('{MC}/_ideas_R8/cache/fam_link.parquet'))",
        "REG_raw": "lraw", "REG_links": "lreg", "pure": None}
for nm, tbl in arms.items():
    L = arm_labels(tbl); land(nm, L); log(nm)
# 반박 변형(REG_raw 기준)
L = arm_labels("lraw", rule="first"); land("REG_raw_firstKR", L)
L = arm_labels("lraw", desc_tie=True); land("REG_raw_tie_desc", L)
free_tab(1); L = arm_labels("lraw"); land("REG_raw_freetab_ge1", L); free_tab(3)
# cweq F (링크 무관)
w = km.assign(w=1 / km.groupby("code").ksic.transform("size"))
Dw = con.sql("select cc, code, ksic_prov from d").df()
prov = rta("ksic_prov", "ksic_prov", Dw)
n = pd.DataFrame({cc: Dw[Dw.cc == cc][["code"]].merge(w, on="code").groupby("ksic").w.sum().reindex(K50).fillna(0) for cc in CCS})
sh = n / n.sum(axis=0); rcw = sh.div(sh.mean(axis=1), axis=0)
cw_rho = sp(rcw.KR, prov.KR)
T = pd.DataFrame(rows); T["rho3"] = T.rho_raw.round(3)
T.to_csv(W + "/verify_1/v1_rows.csv", index=False, encoding="utf-8-sig"); print(T.to_string())
rng_ = {}
for a in T.arm.unique():
    v = [cw_rho, T[(T.arm == a) & (T.item == "txt_F")].rho_raw.iloc[0], T[(T.arm == a) & (T.item == "free_F")].rho_raw.iloc[0]]
    rng_[a] = dict(min_raw=min(v), max_raw=max(v), once=f"{min(v):.2f}–{max(v):.2f}", double=f"{round(round(min(v),3),2):.2f}–{round(round(max(v),3),2):.2f}")
R["cweq_rho"] = cw_rho; R["range_168"] = rng_; R["rows"] = rows
json.dump(R, open(W + "/verify_1/v1_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(rng_, ensure_ascii=False, indent=1)); log("end")
