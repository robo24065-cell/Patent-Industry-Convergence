# verify_2 v3: 회복률 정의 점검 — 납부완료연도 기준(원 주장) vs 소멸등록 이벤트 기준
import sqlite3, os, json
import pandas as pd
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\verify_2")
con = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + "?mode=ro", uri=True)
rt = pd.read_sql("select rgst_no, seq, rgstCsDate, rgstCsName, rgstCsReason from right where rgstCsName like '%소멸%' or rgstCsName like '%회복%'", con)
con.close()
base = pd.read_parquet(os.path.join(OUT, "v1_base.parquet"))
rt["d"] = pd.to_datetime(rt.rgstCsDate, format="%Y%m%d", errors="coerce")
le = rt[(rt.rgstCsName == "소멸등록") & rt.rgstCsReason.isin(["등록료불납", "연차료불납"])][["rgst_no", "seq", "d"]]
re_ = rt[rt.rgstCsName.str.startswith("회복")][["rgst_no", "seq", "d"]]
m = le.merge(re_, on="rgst_no", how="left", suffixes=("", "_r"))
m["rest_after"] = m.d_r.notna() & (m.d_r >= m.d)
ev = m.groupby(["rgst_no", "seq", "d"]).rest_after.any().reset_index()
r = {}
r["lapse_events"] = int(len(ev)); r["lapse_events_restored"] = int(ev.rest_after.sum())
r["event_restore_rate_by_lapyear"] = {int(y): round(float(g.rest_after.mean()), 4) for y, g in ev.groupby(ev.d.dt.year) if y >= 2012}
# 회복된 특허의 납부완료연도 분포(원 주장 분모에서 어디로 갔는가)
rs = base[base.rest.notna()]
r["restored_patents_paidthru_year"] = {int(k): int(v) for k, v in rs.paid_thru.dt.year.value_counts().sort_index().items()}
# 회복 이벤트와 같은 날 소멸 이벤트 존재 비율 (모든 회복 이벤트 기준)
same = re_.merge(le, on="rgst_no", suffixes=("", "_l"))
same["sameday"] = same.d == same.d_l
r["restore_events_with_sameday_lapse"] = round(float(same.groupby(["rgst_no", "seq"]).sameday.any().mean()), 4)
# seq 순서: 같은 날일 때 소멸 seq < 회복 seq ?
s2 = same[same.sameday]
r["sameday_lapse_seq_before_restore"] = round(float((s2.seq_l < s2.seq).mean()), 4) if len(s2) else None
json.dump(r, open(os.path.join(OUT, "v3_restore.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(r)
