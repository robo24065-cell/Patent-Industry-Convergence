# verify_0: N7 perform_grp 매핑 민감도 독립 재계산 (duckdb 경로, 작업자 스크립트 미사용)
import sys, json, duckdb, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
RAW = (B + r"\_shared\ext\ntis_project_raw.parquet").replace("\\", "/")
LNK = (B + r"\_shared\ext\ntis_patent_link.parquet").replace("\\", "/")
T = B + r"\N7_특허기업_정부RnD첫수행"
OUTD = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C03\C03-T3\verify_0"
o = {}
con = duckdb.connect()
ct = con.sql(f"select perform_agent_cd cd, perform_agent txt, perform_grp grp, count(*) n from read_parquet('{RAW}') group by all order by n desc").df()
o["crosstab"] = ct[ct.n >= 20].astype(str).values.tolist()
# 공식 정의 재구성(SQL): 2010<=year<=2024, corp_rgst_no 숫자만, 첫 연도
fy = con.sql(f"""select regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, min(year::int) fy from read_parquet('{RAW}')
 where year::int between 2010 and 2024 and corp_rgst_no is not null group by 1""").df().set_index("bz").fy
FIRMW = "('04','05','06')"; FIRMC = "('04','05','08')"
def fy_firm(codes, ymax=2024):
    return con.sql(f"""select regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, min(year::int) fy from read_parquet('{RAW}')
 where year::int between 2010 and {ymax} and corp_rgst_no is not null and perform_agent_cd in {codes} group by 1""").df().set_index("bz").fy
P = pd.read_parquet(T + r"\final_test_pred_r5.parquet")
P["bz"] = P.bizno.astype(str).str.replace(r"[^0-9]", "", regex=True)
f = P.bz.map(fy)
yr = f.between(2023, 2024).astype(int)
o["final"] = dict(n=len(P), pos_frozen=int(P.y.sum()), pos_recon=int(yr.sum()), mismatch=int((yr != P.y).sum()), first_le2022=int((f <= 2022).sum()),
                  dup_bz=int(P.bz.duplicated().sum()))
from sklearn.metrics import roc_auc_score as A
def sc(y):
    y = np.asarray(y, int); r = {c: A(y, P[c]) for c in ["p_champ", "p_M0", "p_SB"]}
    r["sk_M0"] = (r["p_champ"] - r["p_M0"]) / (1 - r["p_M0"]); r["sk_SB"] = (r["p_champ"] - r["p_SB"]) / (1 - r["p_SB"])
    for k in (123, 614, 1229):
        r[f"top{k}"] = int(y[np.argsort(-P.p_champ.values, kind="mergesort")[:k]].sum())
        r[f"M0top{k}"] = int(y[np.argsort(-P.p_M0.values, kind="mergesort")[:k]].sum())
    r["pos"] = int(y.sum()); return {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()}
o["score_frozen"] = sc(P.y); o["score_recon"] = sc(yr)
fw = P.bz.map(fy_firm(FIRMW)); fc = P.bz.map(fy_firm(FIRMC))
yw = fw.between(2023, 2024).astype(int); yc = fc.between(2023, 2024).astype(int)
o["alt_wrong"] = sc(yw); o["alt_corr"] = sc(yc)
o["alt_symdiff_wrong_vs_official"] = int((yw != P.y).sum()); o["alt_symdiff_corr_vs_official"] = int((yc != P.y).sum())
# 양성의 첫 진입 연도 과제 코드
pos = P[P.y == 1]
q = con.sql(f"""select regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, year::int yr, perform_agent_cd cd from read_parquet('{RAW}')
 where year::int between 2023 and 2024 and corp_rgst_no is not null""").df()
q = q[q.bz.isin(set(pos.bz))]; q["fy"] = q.bz.map(fy); q = q[q.yr == q.fy]
mix = q.groupby("bz").cd.agg(lambda s: "|".join(sorted(set(map(str, s)))))
o["pos_entry_mix"] = mix.value_counts().to_dict()
# 공식 패널 dev 원점과 대조(dev 레이블: NTIS<=2022)
D = pd.read_parquet(T + r"\cache\final_panel.parquet", columns=["bizno", "t0", "y"])
D["bz"] = D.bizno.astype(str).str.replace(r"[^0-9]", "", regex=True)
fy22 = con.sql(f"""select regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, min(year::int) fy from read_parquet('{RAW}')
 where year::int between 2010 and 2022 and corp_rgst_no is not null group by 1""").df().set_index("bz").fy
dv = {}
for t0, g in D[D.t0 <= 2020].groupby("t0"):
    yy = g.bz.map(fy22).between(t0 + 1, t0 + 2).astype(float)
    dv[int(t0)] = dict(n=len(g), match=float((yy == g.y).mean()), rate=float(g.y.mean()))
o["dev"] = dv
# M2 g_part_firm 출처: ntis_patent_link의 PerformAgent_code 분포 및 perform_grp
lc = con.sql(f"select * from read_parquet('{LNK}') limit 1").df().columns.tolist()
o["link_cols"] = lc
if "PerformAgent_code" in lc:
    extra = [c for c in lc if "Agent" in c or "agent" in c]
    o["link_agent_cols"] = extra
    o["link_code_grp"] = con.sql(f"select PerformAgent_code, perform_grp, count(*) n from read_parquet('{LNK}') group by all order by n desc").df().astype(str).values.tolist()
json.dump(o, open(OUTD + r"\v_n7.json", "w", encoding="utf8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
