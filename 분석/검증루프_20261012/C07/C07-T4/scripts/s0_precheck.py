# C07-T4 사전 점검 (1)(2): owner·right 구조, 과거 행 보존, 이전 사유 분포(개발 2015–2022 연도별, 잠금 2025는 사유별 연 합계 1줄)
import sqlite3, sys, json, time
import pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
DB = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T4"
con = sqlite3.connect('file:' + DB + '?mode=ro', uri=True)
t = time.time(); R = {}
o = pd.read_sql("select rgst_no, seq, finalOwnerYn f, ownerRgstCsDate d, ownerRgstCsName nm, ownerRgstCsReason rs from owner", con)
p = pd.read_sql("select rgst_no, rgstDate from patent", con)
R['owner_rows'] = len(o); R['patents'] = len(p)
R['owner_date_null'] = int(o.d.isna().sum()); R['owner_date_bad'] = int((o.d.fillna('').str.len() != 8).sum())
R['owner_max_date'] = o.d.max(); R['owner_min_date'] = o.d.min()
n = o.groupby('rgst_no').size()
R['patents_with_owner'] = int(n.size); R['patents_no_owner'] = int(len(p) - n.size)
R['owner_rows_per_patent'] = n.value_counts().sort_index().head(10).to_dict()
s0 = o[o.seq == 0].merge(p, on='rgst_no')
R['seq0_date_eq_rgstDate'] = float((s0.d == s0.rgstDate).mean())
R['seq0_reason_null'] = float(s0.nm.isna().mean())
R['final_Y_per_patent'] = o[o.f == 'Y'].groupby('rgst_no').size().value_counts().head(5).to_dict()
# seq 단조성: seq가 클수록 날짜가 같거나 늦은가
o2 = o.sort_values(['rgst_no', 'seq'])
dd = o2.groupby('rgst_no').d.apply(lambda s: bool((s.fillna('0').values[1:] >= s.fillna('0').values[:-1]).all()))
R['seq_date_monotone_share'] = float(dd.mean())
o['y'] = o.d.str[:4]
ev = o[o.seq > 0]
dev = ev[(ev.y >= '2015') & (ev.y <= '2022')]
tab = dev.groupby(['nm', 'rs'], dropna=False).size().sort_values(ascending=False)
R['dev_reason_table_top'] = {f'{a}|{b}': int(v) for (a, b), v in tab.head(30).items()}
R['dev_reason_by_year_yangdo_full'] = dev[(dev.nm == '양도') & (dev.rs == '권리의 전부이전등록')].groupby('y').size().to_dict()
lk = ev[ev.y == '2025']
R['lock2025_reason_total_only'] = {f'{a}|{b}': int(v) for (a, b), v in lk.groupby(['nm', 'rs'], dropna=False).size().sort_values(ascending=False).head(15).items()}
R['rows_after_2025'] = int((ev.y >= '2026').sum())
r = pd.read_sql("select rgst_no, rgstCsDate d, rgstCsName nm, rgstCsReason rs from right", con)
R['right_rows'] = len(r)
R['right_name_top'] = r.nm.value_counts().head(25).to_dict()
R['right_disp_change_names'] = r[r.nm.fillna('').str.contains('표시변경')].nm.value_counts().head(10).to_dict()
R['right_disp_change_by_year'] = r[r.nm.fillna('').str.contains('표시변경')].d.str[:4].value_counts().sort_index().to_dict()
R['right_extinct_names'] = r[r.nm.fillna('').str.contains('소멸|말소')].nm.value_counts().head(10).to_dict()
R['right_max_date'] = r.d.max()
R['sec'] = round(time.time() - t, 1)
json.dump(R, open(W + r'\cache\s0_precheck.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
