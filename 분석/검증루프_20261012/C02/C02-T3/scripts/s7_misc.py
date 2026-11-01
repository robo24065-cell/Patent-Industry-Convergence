# E-C01-07(T3 CN 피처연도 구성), E-C01-22/4.3(등록원부 단축창 건수), X3(키 평문 여부, 값 미출력), Part3(NPS 업종코드 변경 구성)
import sys, json, re, sqlite3, os, duckdb, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = P + r"\분석\모델경쟁_20260927"; OUT = P + r"\분석\검증루프_20261012\C02\C02-T3"
META = (MC + r"\_shared\meta\*.parquet").replace("\\", "/")
R = {}
con = duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2; SET enable_progress_bar=false")
# (1) CN: T3 피처연도 = least(pub_date, rgst_date) 연도. 실용신안 비중, main_ipc 결측 비중
q = f"""select try_cast(substr(least(coalesce(nullif(pub_date,''),'99999999'), coalesce(nullif(rgst_date,''),'99999999')),1,4) as int) y,
  count(*) n,
  avg((length(appl_no) in (12,14) and substr(appl_no,5,1) in ('2','9'))::int) um_share,
  avg((main_ipc is null or main_ipc='')::int) ipc_missing
  from read_parquet('{META}') where cc='CN' group by 1 having y between 2014 and 2022 order by 1"""
R["cn_featyear"] = [list(map(lambda v: round(v, 4) if isinstance(v, float) else v, r)) for r in con.sql(q).fetchall()]
print(R["cn_featyear"], flush=True)
# (2) 등록원부: 분쟁 예고등록 2025-01~2026-07, 최대 날짜
c = sqlite3.connect("file:" + (P + r"\수집파이프라인\registry_full.sqlite").replace("\\", "/") + "?mode=ro", uri=True)
R["dispute_2023_24"] = c.execute("select count(*) from right where rgstCsName in ('무효(심판)예고등록','권리범위(심판)예고등록') and rgstCsDate between '20230101' and '20241231'").fetchone()[0]
R["dispute_2025_01_2026_07"] = c.execute("select count(*) from right where rgstCsName in ('무효(심판)예고등록','권리범위(심판)예고등록') and rgstCsDate between '20250101' and '20260731'").fetchone()[0]
R["dispute_by_year"] = c.execute("select substr(rgstCsDate,1,4), count(*) from right where rgstCsName in ('무효(심판)예고등록','권리범위(심판)예고등록') and rgstCsDate>='2022' group by 1").fetchall()
R["right_max_date"] = c.execute("select max(rgstCsDate) from right where rgstCsDate<'2100'").fetchone()[0]
c.close()
print({k: v for k, v in R.items() if k != "cn_featyear"}, flush=True)
# (3) X3: collect_registry.py 18–30행에 키 형태 문자열 리터럴이 있는지(값은 출력하지 않음, 길이만)
fp = P + r"\수집파이프라인\collect_registry.py"
hits = []
if os.path.exists(fp):
    L = open(fp, encoding="utf-8", errors="replace").read().splitlines()
    for i, line in enumerate(L, 1):
        m = re.search(r"""(?i)(key|token|secret)\w*\s*=\s*["']([^"']{16,})["']""", line)
        if m: hits.append({"line": i, "var": m.group(1), "literal_len": len(m.group(2))})
        m2 = re.search(r"""["']([A-Za-z0-9%+/=]{30,})["']""", line)
        if m2 and not m: hits.append({"line": i, "var": "(long literal)", "literal_len": len(m2.group(1))})
    R["x3_lines"] = hits; R["x3_n_lines"] = len(L)
    R["x3_env_used"] = any("environ" in l or "getenv" in l for l in L)
print(R.get("x3_lines"), R.get("x3_env_used"), flush=True)
# (4) Part3: NPS 패널 업종코드 연속 스냅샷 변경 구성
pn = pd.read_parquet(MC + r"\_shared\ext\firm_nps_firm_panel.parquet")
R["nps_panel_cols"] = list(pn.columns)
json.dump(R, open(OUT + r"\s7_misc_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(pn.dtypes.to_string()); print(pn.head(3).to_string())
