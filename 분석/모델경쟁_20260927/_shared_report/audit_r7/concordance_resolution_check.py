# R7 공고문 부합성 감사 보조: 공식 IPC-KSIC 연계표(2013)의 해상도와 제공 KSIC 라벨 비교 (기술통계, 모형 없음)
# 읽기 전용: _shared/meta/*.parquet, 공모전데이터/외부데이터/20131002_IPC-KSIC/20131002_IPC-KSIC.xlsx
import os, json, re
import pandas as pd, duckdb
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
XL = os.path.join(BASE, r"공모전데이터\외부데이터\20131002_IPC-KSIC\20131002_IPC-KSIC.xlsx")
OUT = os.path.dirname(os.path.abspath(__file__))

cc = pd.read_excel(XL, "전체연계표", header=0)
cc.columns = ["code", "ipc", "n"]
cc = cc.dropna()
cc["ipc"] = cc["ipc"].astype(str).str.strip()
codes = sorted(cc["code"].unique())
con = duckdb.connect()
con.execute("PRAGMA threads=2; PRAGMA memory_limit='3GB'")
ks = [r[0] for r in con.execute(f"select distinct ksic from read_parquet('{META}') order by 1").fetchall()]
ks = [str(k) for k in ks]

# KSIC 10차(제공) -> 9차(2013 연계표) 차수 차이 수동 보정: 화학 202/204 재편, 291 일반목적기계
OVERRIDE = {"20202": "C2030", "20411": "C2042", "20423": "C2043"}
def map_code(k):
    if k in OVERRIDE:
        return OVERRIDE[k]
    if k.startswith("291"):
        return "C2919"
    for c in ("C" + k[:4], "C" + k[:3] + "0", "C" + k[:2] + "00", "C" + k[:2] + "10"):
        if c in codes:
            return c
    return None
kmap = {k: map_code(k) for k in ks}
grp = {}
for k, c in kmap.items():
    grp.setdefault(c, []).append(k)
res = {"n_provided_ksic": len(ks), "n_concordance_codes_total": len(codes),
       "n_concordance_codes_covering_50": len([c for c in grp if c]),
       "unmapped": [k for k, c in kmap.items() if c is None],
       "max_ksic_per_code": max(len(v) for v in grp.values()),
       "codes_with_2plus": {c: v for c, v in grp.items() if len(v) >= 2},
       "ksic_alone_in_code": sum(1 for v in grp.values() if len(v) == 1)}

# 확률 연계: IPC 서브그룹 -> 연계표 산업 분포(행 정규화), 제공 KR 출원의 main_ipc로 argmax 산업
cc["p"] = cc["n"] / cc.groupby("ipc")["n"].transform("sum")
top = cc.sort_values(["ipc", "p"], ascending=[True, False]).drop_duplicates("ipc")[["ipc", "code", "p"]]
ipc_ncode = cc.groupby("ipc")["code"].nunique().rename("ncode").reset_index()
con.register("top", top); con.register("ipcn", ipc_ncode)
km = pd.DataFrame({"ksic": list(kmap.keys()), "kcode": list(kmap.values())})
con.register("km", km)
q = f"""
with kr as (select distinct appl_no, cast(ksic as varchar) ksic, trim(main_ipc) ipc from read_parquet('{META}')
            where cc='KR' and main_ipc is not null and appl_year between 2005 and 2022)
select count(*) n,
  avg(case when t.ipc is not null then 1 else 0 end) matched_ipc,
  avg(case when t.code = km.kcode then 1.0 when t.ipc is null then null else 0 end) top_agree_at_concordance_level,
  avg(case when t.ipc is not null then t.p end) mean_top_p,
  avg(case when n.ncode is not null then (n.ncode>1)::int end) share_ipc_multi_industry
from kr left join top t on kr.ipc=t.ipc left join ipcn n on kr.ipc=n.ipc left join km on kr.ksic=km.ksic
"""
r = con.execute(q).fetchdf().iloc[0].to_dict()
res["kr_2005_2022"] = {k: (float(v) if v is not None else None) for k, v in r.items()}
# 같은 출원이 여러 제공 KSIC에 동시 등장하는 비율(제공 라벨의 다중 배정 정도)
r2 = con.execute(f"""select avg((nk>1)::int) share_multi_label, avg(nk) mean_labels from
   (select appl_no, count(distinct ksic) nk from read_parquet('{META}') where cc='KR' and appl_year between 2005 and 2022 group by 1)""").fetchdf().iloc[0].to_dict()
res["kr_provided_multilabel"] = {k: float(v) for k, v in r2.items()}
json.dump(res, open(os.path.join(OUT, "concordance_resolution_check.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1))
