# 1.4 표 3·데이터 출처 숫자 원천 재계산 (모두 읽기 전용)
import sys, json, sqlite3, os, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding="utf-8")
P = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = P + r"\분석\모델경쟁_20260927"; OUT = P + r"\분석\검증루프_20261012\C02\C02-T3"
R = {}
def ro(path):
    return sqlite3.connect("file:" + path.replace("\\", "/") + "?mode=ro", uri=True)
c = ro(P + r"\수집파이프라인\registry_full.sqlite")
tabs = [r[0] for r in c.execute("select name from sqlite_master where type='table'")]
R["registry_tables"] = {t: c.execute(f'select count(*) from "{t}"').fetchone()[0] for t in tabs}
R["registry_total_rows"] = sum(R["registry_tables"].values()); R["registry_n_tables"] = len(tabs)
c.close()
c = ro(P + r"\공모전데이터\외부데이터\DART\dart_company.sqlite")
R["dart_tables"] = {t: c.execute(f'select count(*) from "{t}"').fetchone()[0] for (t,) in c.execute("select name from sqlite_master where type='table'")}
c.close()
with open(P + r"\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt", "rb") as f:
    R["corp_applicant_lines"] = sum(1 for _ in f)
for nm in ["ntis_project_raw", "ntis_rpatent_raw"]:
    fp = MC + rf"\_shared\ext\{nm}.parquet"
    if os.path.exists(fp): R[nm + "_rows"] = pq.ParquetFile(fp).metadata.num_rows
c = ro(P + r"\수집파이프라인\business_rights.sqlite")
R["right_list_rows"] = c.execute("select count(*) from right_list").fetchone()[0]
c.close()
json.dump(R, open(OUT + r"\s6_table3_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
