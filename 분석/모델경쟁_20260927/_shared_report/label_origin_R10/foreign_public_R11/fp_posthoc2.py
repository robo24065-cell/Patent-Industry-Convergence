# -*- coding: utf-8 -*-
r"""R11 foreign_public — [사후2] fp_posthoc 결과를 본 뒤 추가: 시간 분리판(표는 KR 출원≤2014 문서로만, 적용은 KR 출원 2015–19 문서). 판정에 쓰지 않는다.
질문: 같은 FREE-IPC 규칙(해설서 텍스트 단독 TXT-FREE → IPC sg→mg→sc 최빈표)을 KR 문서에 적용하면 근거 A(KR 라벨 − TXT-FREE)의 격차는 얼마나 남나.
주의: 표는 KR 출원≤2019 문서로 만들었으므로 KR 문서에는 표본 내 적용이다(자기 문서가 최빈 계산에 들어감 → 공개 쪽에 약간 유리).
읽기만: c5_R10/alt_assign/alt_KR.parquet, T3 rec_conc, label_origin_R10/cache. 쓰기: res_posthoc2.csv"""
import os, sys, json, time
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, duckdb
HERE = os.path.dirname(os.path.abspath(__file__)); MC = os.path.abspath(os.path.join(HERE, r"..\..\.."))
f = lambda *p: os.path.join(MC, *p).replace("\\", "/")
LO = os.path.join(MC, r"_shared_report\label_origin_R10\cache")
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
MODE = "arg_min(k, -n*100000 + cast(k as int))"
con.execute(f"""create table d as select a.appl_no, a.appl_year, a.ksic_prov, a.k_free k, r.sg, split_part(r.sg,'/',1) mg, left(r.sg,4) sc
   from read_parquet('{f("_shared_report","fusion_advantage","c5_R10","alt_assign","alt_KR.parquet")}') a left join read_parquet('{f("T3_기술셀부상예측","ESSENTIAL_R7","rec_conc.parquet")}') r
   on r.cc='KR' and r.appl_no=a.appl_no""")
for lvl in ("sg", "mg", "sc"):
    con.execute(f"""create table t_{lvl} as with g as (select {lvl} key_, k, count(*) n from d where appl_year<=2014 and {lvl} is not null and {lvl}<>'' group by all),
       tot as (select key_ from g group by all having sum(n)>=3) select key_, {MODE} k from g join tot using(key_) group by all""")
kr = con.sql("""select d.appl_no, d.appl_year ay, d.ksic_prov, d.k k_txtfree, coalesce(a.k, b.k, c.k) k_freeipc from d left join t_sg a on a.key_=d.sg
   left join t_mg b on b.key_=d.mg left join t_sc c on c.key_=d.sc""").df()
kr["appl_no"] = kr.appl_no.astype(str)
fam = pd.read_parquet(os.path.join(LO, "fam_reg.parquet")); famset = set(fam.kr_appl_no.astype(str))
pf = pd.read_parquet(os.path.join(LO, "pat_firm.parquet")); reg = pd.read_parquet(os.path.join(LO, "firm_reg.parquet"))
KSIC50 = sorted(set(kr.ksic_prov.dropna()))
rng = np.random.default_rng(20261012); out = []
X = pf.merge(kr, on="appl_no")
X = X[X.ay.between(2015, 2019)]
for tag, Z0 in (("패밀리 보유 KR 2015-19(표 ≤2014)", X[X.appl_no.isin(famset)]), ("법인 KR 2015-19(표 ≤2014)", X)):
    for src in ("nps", "dart"):
        R = reg[["applicantCd", src]].dropna(); R = R[R[src].isin(KSIC50)]
        Z = Z0.merge(R.rename(columns={src: "r"}), on="applicantCd")
        A = pd.DataFrame({"K": Z.ksic_prov == Z.r, "I": Z.k_freeipc == Z.r, "T": Z.k_txtfree == Z.r}).astype(float); A["f"] = Z.applicantCd.values
        T = A.groupby("f").mean(); M = T.values; v = M.mean(0)
        B = M[rng.integers(0, len(T), (1000, len(T)))].mean(1)
        r = {"표본": tag, "기준": src.upper(), "기업": len(T), "특허": len(Z), "KR제공": round(v[0], 4), "KR_FREE-IPC": round(v[1], 4), "KR_TXT-FREE": round(v[2], 4)}
        for nm, i, j in (("제공-FREEIPC", 0, 1), ("제공-TXTFREE", 0, 2)):
            dd = B[:, i] - B[:, j]; r[nm] = round(v[i] - v[j], 4); r[nm + "_lo"] = round(float(np.percentile(dd, 2.5)), 4); r[nm + "_hi"] = round(float(np.percentile(dd, 97.5)), 4)
        out.append(r)
P = pd.DataFrame(out); P.to_csv(os.path.join(HERE, "res_posthoc2.csv"), index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250); print(P.to_string())
print("KR 문서 FREE-IPC = 제공 일치율(전체):", round(float((kr.k_freeipc == kr.ksic_prov).mean()), 4), " TXT-FREE:", round(float((kr.k_txtfree == kr.ksic_prov).mean()), 4))
