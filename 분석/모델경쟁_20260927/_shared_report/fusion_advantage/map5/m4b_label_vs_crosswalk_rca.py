# -*- coding: utf-8 -*-
r"""map5 4b: '제공 KSIC 라벨로 만든 한국 특허 포지션' vs '공식 IPC-KSIC 연계표로 만든 포지션' — 어느 쪽이 수출 경쟁력(RCA)과 더 정합하나.
- 연계표 배정: T3 ESSENTIAL_R7\rec_conc.parquet (5개국 레코드별 연계표 argmax 코드, 읽기만). 라벨공간 27개 그룹 밖 코드·미배정은 뺀다.
- 제공 라벨은 label_validity GROUP_PREFIX 규칙으로 같은 27개 그룹에 올린다 → 같은 해상도(연계표의 최선)에서 비교.
- 지표: 한국 특허 RTA(5개 관청 동일가중, 출원 2015–17, 발명특허), 한국 기원 해외 출원 RTA(US·EP·JP·CN 관청, 우선권 첫 국가=KR, 2017–19).
- 기준: 한국 수출 RCA(5개 보고국 동일가중, 2015–17), 그룹 합산. Spearman 차이의 그룹 부트스트랩 2000회 95% CI.
- 한계: 레코드 풀이 이미 제공 라벨로 고른 50 KSIC 범위다(연계표에 유리). 수출은 ksic_hs6_map의 HS6 균등 분할이라 일부 그룹 안 KSIC 수출이 같다.
"""
import os, json
import numpy as np, pandas as pd, duckdb
from scipy.stats import spearmanr
OUT = os.path.dirname(os.path.abspath(__file__))
MC = os.path.abspath(os.path.join(OUT, r"..\..\.."))
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
REC = os.path.join(MC, r"T3_기술셀부상예측\ESSENTIAL_R7\rec_conc.parquet").replace("\\", "/")
rng = np.random.default_rng(11)
GP = [("C1500", ["15"]), ("C2030", ["202"]), ("C2042", ["2041"]), ("C2043", ["2042"]), ("C2049", ["2049"]), ("C2100", ["21"]), ("C2300", ["23"]),
      ("C2410", ["241"]), ("C2420", ["242"]), ("C2590", ["259"]), ("C2610", ["261"]), ("C2620", ["262"]), ("C2630", ["263"]), ("C2640", ["264"]),
      ("C2650", ["265"]), ("C2710", ["271"]), ("C2720", ["272"]), ("C2730", ["273"]), ("C2810", ["281"]), ("C2850", ["285"]), ("C2890", ["289"]),
      ("C2918", ["2918"]), ("C2919", ["291"]), ("C2920", ["292"]), ("C3000", ["30"]), ("C3120", ["312"]), ("C3130", ["313"]), ("C3300", ["33"])]


def grp(c):
    for g, ps in GP:
        if any(str(c).startswith(p) for p in ps):
            return g


con = duckdb.connect()
con.sql(f"SET memory_limit='3GB'; SET threads=2; SET enable_progress_bar=false; SET temp_directory='{OUT.replace(chr(92), '/')}/_duck_tmp'")
inv = """case when m.cc='KR' then substr(m.appl_no,1,2)='10'
 when m.cc='CN' then length(m.appl_no) in (12,14) and substr(m.appl_no,5,1) in ('1','8')
 when m.cc='JP' then length(m.appl_no) in (8,17) else true end"""
d = con.sql(f"""select m.ksic, r.code, m.cc, m.appl_year, split_part(coalesce(m.prio_cc,''),'|',1)='KR' kr_origin, count(*) n
 from read_parquet('{META}') m join read_parquet('{REC}') r on m.cc=r.cc and m.appl_no=r.appl_no
 where ({inv}) and m.appl_year between 2015 and 2019 group by all""").df()
d["g_label"] = d.ksic.astype(str).map(grp)
SPACE = [g for g, _ in GP]
d["g_cw"] = d.code.where(d.code.isin(SPACE))
res = {"records_2015_2019": int(d.n.sum()), "cw_in_space_share": round(d[d.g_cw.notna()].n.sum() / d.n.sum(), 4),
       "label_cw_agree_share": round(d[d.g_cw == d.g_label].n.sum() / d.n.sum(), 4)}


def rta(col, y0, y1):
    x = d[d.appl_year.between(y0, y1) & d[col].notna()].groupby([col, "cc"]).n.sum().unstack().fillna(0)
    s = x / x.sum()
    return (s.div(s.mean(1), axis=0)).KR


def abroad(col, y0, y1):
    x = d[d.appl_year.between(y0, y1) & d[col].notna() & (d.cc != "KR")]
    k = x[x.kr_origin].groupby(col).n.sum(); a = x.groupby(col).n.sum()
    return (k / a) / (k.sum() / a.sum())


ek = pd.read_csv(os.path.join(OUT, "m4_exports_ksic_5rep.csv"), dtype={"k5": str})
ek["g"] = ek.k5.map(grp)
v = ek[ek.year.isin([2015, 2016, 2017])].groupby(["g", "rep"]).v.sum().unstack()
s = v / v.sum(); rca = s.div(s.mean(1), axis=0).KR
G = pd.DataFrame({"rca": rca, "rta_label": rta("g_label", 2015, 2017), "rta_cw": rta("g_cw", 2015, 2017),
                  "abroad_label": abroad("g_label", 2017, 2019), "abroad_cw": abroad("g_cw", 2017, 2019)}).dropna()
G.to_csv(os.path.join(OUT, "m4b_group_table.csv"), encoding="utf-8-sig")


def diff_ci(a, b, y, B=2000):
    x = G[[a, b, y]].values; n = len(x)
    r1 = spearmanr(x[:, 0], x[:, 2]).statistic; r2 = spearmanr(x[:, 1], x[:, 2]).statistic
    bs = []
    for _ in range(B):
        i = rng.integers(0, n, n)
        bs.append(spearmanr(x[i, 0], x[i, 2]).statistic - spearmanr(x[i, 1], x[i, 2]).statistic)
    return dict(rho_label=round(r1, 3), rho_crosswalk=round(r2, 3), diff=round(r1 - r2, 3),
                ci=[round(np.nanpercentile(bs, 2.5), 3), round(np.nanpercentile(bs, 97.5), 3)], n_groups=n)


res["patRTA_vs_exportRCA"] = diff_ci("rta_label", "rta_cw", "rca")
res["abroadRTA_vs_exportRCA"] = diff_ci("abroad_label", "abroad_cw", "rca")
res["rho_rta_label_vs_cw"] = round(spearmanr(G.rta_label, G.rta_cw).statistic, 3)
json.dump(res, open(os.path.join(OUT, "m4b_summary.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)); print(G.round(2).to_string())
