# -*- coding: utf-8 -*-
r"""c5_R10 c1: 5개국 전체 문서(_shared/meta)에 대체 KSIC 배정(CW·TXT·FREE-IPC·PKR-IPC 진단) → alt_assign.parquet, (1) 라벨 일치 표.
읽기만: _shared/meta, T3 ESSENTIAL_R7/rec_conc.parquet·conc_ksic_map.csv, T3 TXT_R9/txt_assign.parquet, T1 TXT_R9/cache/kr_txtfree.parquet, _ideas_R8/cache/fam_link.parquet
정의는 PREREG.txt."""
import os, sys, json, time
sys.dont_write_bytecode = True
import pandas as pd, duckdb
HERE = os.path.dirname(os.path.abspath(__file__)); MC = os.path.abspath(os.path.join(HERE, r"..\..\.."))
f = lambda *p: os.path.join(MC, *p).replace("\\", "/")
META = f("_shared", "meta", "*.parquet"); RC = f("T3_기술셀부상예측", "ESSENTIAL_R7", "rec_conc.parquet")
TXT = f("T3_기술셀부상예측", "TXT_R9", "txt_assign.parquet"); FREE = f("T1_산업대기실", "TXT_R9", "cache", "kr_txtfree.parquet")
FAM = f("_ideas_R8", "cache", "fam_link.parquet")
OUT = os.path.join(HERE, "alt_assign.parquet").replace("\\", "/")
t = time.time(); log = lambda *a: print(*a, f"[{time.time() - t:.0f}s]", flush=True)
DB = os.path.join(HERE, "_duck_tmp", "c1.duckdb"); os.makedirs(os.path.dirname(DB), exist_ok=True)
STAGE = sys.argv[1] if len(sys.argv) > 1 else "build"
con = duckdb.connect(DB)
con.execute(f"set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false; set temp_directory='{HERE.replace(chr(92), '/')}/_duck_tmp'")
km = pd.read_csv(f("T3_기술셀부상예측", "ESSENTIAL_R7", "conc_ksic_map.csv"), dtype=str)[["ksic", "code"]]
km["k_in_code"] = km.groupby("code").ksic.transform("size")
con.register("km", km)
MODE = "arg_min(k, -n*100000 + cast(k as int))"
if STAGE == "build":
    km = pd.read_csv(f("T3_기술셀부상예측", "ESSENTIAL_R7", "conc_ksic_map.csv"), dtype=str)[["ksic", "code"]]
    km["k_in_code"] = km.groupby("code").ksic.transform("size")
    con.register("km", km)
    inv = """case when m.cc='KR' then substr(m.appl_no,1,2)='10'
     when m.cc='CN' then length(m.appl_no) in (12,14) and substr(m.appl_no,5,1) in ('1','8')
     when m.cc='JP' then length(m.appl_no) in (8,17) else true end"""
    con.execute(f"""create or replace table d as select m.cc, m.appl_no, m.appl_year, ({inv}) is_inv,
       split_part(coalesce(m.prio_cc,''),'|',1)='KR' kr_origin, m.ksic ksic_prov, r.sg, split_part(r.sg,'/',1) mg, left(r.sg,4) sc,
       case when r.code in (select code from km) then r.code end code
       from read_parquet('{META}') m join read_parquet('{RC}') r using(cc, appl_no)""")
    log("문서", con.sql("select cc, count(*) from d group by 1 order by 1").fetchall())
    MODE = "arg_min(k, -n*100000 + cast(k as int))"   # 최빈, 동률은 코드 작은 것


    def cascade(name, krsrc, famsrc, use_code):
        """krsrc: (appl_no, k) KR 배정; famsrc: True면 패밀리 상속. use_code: 연계표 코드 안에서만(TXT식) vs 코드 무관(FREE식)."""
        key = "code, " if use_code else ""
        con.execute(f"create or replace table krk as select {key} d.sg, d.mg, d.sc, s.k from d join {krsrc} s on d.cc='KR' and d.appl_no=s.appl_no"
                    + (" where d.code is not null" if use_code else " where d.appl_year<=2019"))
        for lvl in ("sg", "mg", "sc"):
            con.execute(f"""create or replace table tab_{lvl} as with g as (select {key} {lvl} key_, k, count(*) n from krk
                where {lvl} is not null and {lvl}<>'' group by all), tot as (select {key} key_ from g group by all having sum(n)>=3)
                select {key} key_, {MODE} k from g join tot using({key} key_) group by all""")
        if use_code:
            con.execute(f"create or replace table tab_code as with g as (select code, k, count(*) n from krk group by all) select code, {MODE} k from g group by 1")
        on = lambda lvl: (f"a_{lvl}.code=d.code and " if use_code else "") + f"a_{lvl}.key_=d.{lvl}"
        famj, famc = "", "null"
        if famsrc:
            con.execute(f"""create or replace table fam_ as with g as (select fl.cc, fl.appl_no, s.k, count(*) n from read_parquet('{FAM}') fl
                join {krsrc} s on s.appl_no=fl.kr_appl_no group by all) select cc, appl_no, {MODE} k from g group by 1,2""")
            famj = "left join fam_ fm on fm.cc=d.cc and fm.appl_no=d.appl_no"; famc = "fm.k"
        codej = "left join tab_code a_code on a_code.code=d.code" if use_code else ""
        codec = "a_code.k" if use_code else "null"
        con.execute(f"""create or replace table x_{name} as select d.cc, d.appl_no,
            coalesce({famc}, a_sg.k, a_mg.k, a_sc.k, {codec}) k,
            case when {famc} is not null then 'family' when a_sg.k is not null then 'ipc_sg' when a_mg.k is not null then 'ipc_mg'
                 when a_sc.k is not null then 'ipc_sc' when {codec} is not null then 'code_mode' else 'none' end m
            from d {famj} left join tab_sg a_sg on {on('sg')} left join tab_mg a_mg on {on('mg')} left join tab_sc a_sc on {on('sc')} {codej}
            where d.cc<>'KR'""")
        log(name, "해외 배정", con.sql(f"select m, count(*) from x_{name} group by 1 order by 2 desc").fetchall())


    # ---- FREE-IPC(공개) ----
    con.execute(f"create or replace table kfree as select appl_no, k_free k from read_parquet('{FREE}')")
    cascade("free", "kfree", True, False)
    # ---- PKR-IPC(진단, 제공 KR 라벨) ----
    con.execute("create or replace table kprov as select appl_no, ksic_prov k from d where cc='KR'")
    cascade("pkr", "kprov", True, False)
    # ---- TXT(공개, R9 T3 그대로 + 풀 밖 문서는 같은 규칙의 표) ----
    con.execute(f"create or replace table t3 as select cc, appl_no, ksic_txt, \"method\" m from read_parquet('{TXT}')")
    con.execute("""create or replace table ktxt as select appl_no, ksic_txt k from t3 where cc='KR' and m='text'""")
    # T3 s01 표: KR 'text' 배정 문서(풀)의 (코드, sg/mg/sc) 최빈. 패밀리 상속은 txt_assign 안 문서에만 있음(풀 밖 문서는 표로).
    cascade("txtx", "ktxt", False, True)
    # KR 문서 중 txt_assign에 없는 것(풀 밖)도 같은 표로
    con.execute("""create or replace table krx as select d.cc, d.appl_no, coalesce(a_sg.k, a_mg.k, a_sc.k, a_code.k) k from d
       left join tab_sg a_sg on a_sg.code=d.code and a_sg.key_=d.sg left join tab_mg a_mg on a_mg.code=d.code and a_mg.key_=d.mg
       left join tab_sc a_sc on a_sc.code=d.code and a_sc.key_=d.sc left join tab_code a_code on a_code.code=d.code
       where d.cc='KR' and d.code is not null""")

    log("build done"); sys.exit(0)
ODIR = os.path.join(HERE, "alt_assign").replace("\\", "/"); os.makedirs(ODIR, exist_ok=True)
if STAGE == "copy":
    for CC in sys.argv[2].split(","):
        if os.path.exists(f"{ODIR}/alt_{CC}.parquet"): continue
        con.execute(f"""copy (select d.cc, d.appl_no, d.appl_year, d.is_inv, d.kr_origin, d.ksic_prov, d.code, km1.k_in_code,
             case when km1.k_in_code=1 then km1.ksic end k_cw,
             case when d.code is null then null when km1.k_in_code=1 then km1.ksic
                  when t3.ksic_txt is not null then t3.ksic_txt when d.cc='KR' then krx.k else xt.k end k_txt,
             case when d.code is null then 'none' when km1.k_in_code=1 then 'single' when t3.m is not null then t3.m when d.cc='KR' then 'kr_extra_tab'
                  else 'extra_'||xt.m end m_txt,
             case when d.cc='KR' then kf.k else xf.k end k_free, case when d.cc='KR' then 'text_free' else xf.m end m_free,
             case when d.cc='KR' then d.ksic_prov else xp.k end k_pkr, case when d.cc='KR' then 'prov' else xp.m end m_pkr
           from d left join (select code, min(ksic) ksic, min(k_in_code) k_in_code from km group by 1) km1 on km1.code=d.code
           left join t3 on t3.cc=d.cc and t3.appl_no=d.appl_no
           left join krx on krx.cc=d.cc and krx.appl_no=d.appl_no
           left join x_txtx xt on xt.cc=d.cc and xt.appl_no=d.appl_no
           left join kfree kf on kf.appl_no=d.appl_no
           left join x_free xf on xf.cc=d.cc and xf.appl_no=d.appl_no
           left join x_pkr xp on xp.cc=d.cc and xp.appl_no=d.appl_no
           where d.cc='{CC}') to '{ODIR}/alt_{CC}.part' (format parquet, compression zstd)""")

        os.replace(f"{ODIR}/alt_{CC}.part", f"{ODIR}/alt_{CC}.parquet"); log("저장", CC)
    sys.exit(0)
OUT = ODIR + "/*.parquet"
A = f"read_parquet('{OUT}')"
n = con.sql(f"select count(*), count(distinct cc||appl_no) from {A}").fetchone(); assert n[0] == n[1], n
assert con.sql(f"select count(*) from {A} a join km on km.ksic=a.k_txt where km.code<>a.code").fetchone()[0] == 0
# 재현: txt_assign 안 문서의 k_txt = ksic_txt (정의상), multi-code 풀 밖 비중
st = {}
st["txt_extra_share_by_cc"] = con.sql(f"select cc, avg((m_txt like 'extra%' or m_txt='kr_extra_tab')::int) from {A} where code is not null group by 1 order by 1").fetchall()


def agree(where, tag):
    q = f"""select a.cc, count(*) n,
      avg((a.code is not null)::int) cov_cw, avg((a.k_txt is not null)::int) cov_txt, avg((a.k_free is not null)::int) cov_free, avg((a.k_pkr is not null)::int) cov_pkr,
      avg(coalesce(km.code=a.code, false)::int) cw_group,
      avg(coalesce(a.k_cw=a.ksic_prov, false)::int) cw5,
      avg(case when km.code=a.code then 1.0/a.k_in_code else 0 end) cw_eq_expect,
      avg(coalesce(a.k_txt=a.ksic_prov, false)::int) txt5,
      avg(coalesce(a.k_free=a.ksic_prov, false)::int) free5,
      avg(coalesce(a.k_pkr=a.ksic_prov, false)::int) pkr5,
      avg(case when a.k_txt is not null then (a.k_txt=a.ksic_prov)::int end) txt5_labeled,
      avg(case when a.k_free is not null then (a.k_free=a.ksic_prov)::int end) free5_labeled,
      avg(case when a.k_pkr is not null then (a.k_pkr=a.ksic_prov)::int end) pkr5_labeled
      from {A} a left join km on km.ksic=a.ksic_prov where {where} group by rollup(a.cc) order by a.cc"""
    d = con.sql(q).df(); d.insert(0, "window", tag); return d


T = pd.concat([agree("is_inv and appl_year between 2015 and 2019", "출원2015-19_발명"), agree("true", "전체")])
T.to_csv(os.path.join(HERE, "c1_label_agreement.csv"), index=False, encoding="utf-8-sig")
# 방법별 일치(해외, 2015–19)
for col, mcol in (("k_txt", "m_txt"), ("k_free", "m_free"), ("k_pkr", "m_pkr")):
    st[f"by_method_{col}"] = con.sql(f"""select cc, {mcol} m, count(*) n, avg(coalesce({col}=ksic_prov,false)::int) agree5 from {A}
        where is_inv and appl_year between 2015 and 2019 and cc<>'KR' group by all order by 1,3 desc""").df().to_dict("records")
# KR 기원 해외 문서만(지표 (2)의 분자 집합), 2017–19
st["kr_origin_abroad_agree_1719"] = con.sql(f"""select cc, count(*) n, avg(coalesce(k_txt=ksic_prov,false)::int) txt5, avg(coalesce(k_free=ksic_prov,false)::int) free5,
    avg(coalesce(k_pkr=ksic_prov,false)::int) pkr5, avg(coalesce(k_cw=ksic_prov,false)::int) cw5 from {A}
    where is_inv and kr_origin and cc<>'KR' and appl_year between 2017 and 2019 group by rollup(cc) order by 1""").df().to_dict("records")
# 패밀리 상속(KR 제공 라벨) 일치 — 같은 발명의 KR·해외 라벨이 같은 비율
st["family_pkr_agree"] = con.sql(f"""select cc, count(*) n, avg((k_pkr=ksic_prov)::int) agree from {A} where m_pkr='family' group by rollup(cc) order by 1""").df().to_dict("records")
json.dump(st, open(os.path.join(HERE, "c1_stats.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
pd.set_option("display.width", 250)
print(T.round(3).to_string())
print(json.dumps({k: v for k, v in st.items() if not k.startswith("by_method")}, ensure_ascii=False, default=str)[:3000])
log("end")
