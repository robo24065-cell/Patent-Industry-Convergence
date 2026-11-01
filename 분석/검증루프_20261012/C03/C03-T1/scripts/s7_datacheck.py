# C03-T1 s7: 복원원(법인 권리목록 business_rights.sqlite, 2026 스냅샷) 검증 (a)~(d) 보조 계산 — 읽기 전용
import sys, re, json, time, sqlite3, pathlib
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
ROOT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전")
T1 = ROOT/'분석/모델경쟁_20260927/T1_산업대기실'; TC = T1/'cache'
W = ROOT/'분석/검증루프_20261012/C03/C03-T1'; WC = W/'cache'
def ro(p): return sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
out = {}
c = ro(ROOT/'수집파이프라인/business_rights.sqlite')
out['sqlite_rows_10'] = c.execute("select count(*) from right_list where substr(rgst_no,1,2)='10'").fetchone()[0]
out['rl_pat_rows'] = int(pd.read_parquet(TC/'rl_pat.parquet', columns=['rgst_no']).shape[0])
aq = pd.read_sql('select applicant_cd, status, total_count, fetched, truncated, ts from applicant_query', c)
out['query_status'] = aq.status.value_counts().to_dict()
ok = aq[aq.status == 'OK']; out['ok_truncated'] = int((ok.truncated == 1).sum()); out['ok_fetched_lt_total'] = int((ok.fetched < ok.total_count).sum())
out['query_ts_range'] = [str(aq.ts.min()), str(aq.ts.max())]
out['max_rgst_date'] = c.execute("select max(rgst_date) from right_list where substr(rgst_no,1,2)='10'").fetchone()[0]
c.close()
# 보정 재현율을 법인 유형·규모별로(출원 2020–23 등록번호 있는 범위안 등록특허, 등록원부 원출원인 코드가 T1 법인)
kr = pd.read_parquet(ROOT/'공모전데이터/작업데이터/kr_registered.parquet', columns=['출원번호', '출원연도', '등록번호']); kr.columns = ['appl_no', 'ay', 'rgst_no']; kr['rgst_no'] = kr.rgst_no.astype(str)
ins = pd.read_parquet(TC/'ins_firm.parquet', columns=['rgst_no', 'applicant_cd', 'ay']); ins = ins[(ins.ay >= 2020) & (ins.ay <= 2023)].merge(kr[['rgst_no', 'appl_no']], on='rgst_no')
rl = pd.read_parquet(TC/'rl_pat.parquet', columns=['applicant_cd', 'appl_no', 'pert', 'rgst_date']); rl = rl[rl.pert == '권리자란']
S = set(zip(rl.applicant_cd, rl.appl_no)); ins['in_rl'] = [(a, b) in S for a, b in zip(ins.applicant_cd, ins.appl_no)]
fm = pd.read_parquet(TC/'firm_meta.parquet', columns=['applicant_cd', 'tp']).set_index('applicant_cd').tp
fsz = pd.read_parquet(TC/'ins_firm.parquet', columns=['applicant_cd', 'ay']).query('ay <= 2019').groupby('applicant_cd').size()
ins['natl'] = ins.applicant_cd.map(fm).fillna('결측'); ins['size'] = pd.cut(ins.applicant_cd.map(fsz).fillna(0), [-1, 2, 20, 200, 1e9], labels=['≤2건', '3–20', '21–200', '200+']).astype(str)
out['recall_in_rl_by_natl'] = ins.groupby('natl').in_rl.mean().round(4).to_dict(); out['recall_in_rl_by_size'] = ins.groupby('size').in_rl.mean().round(4).to_dict()
out['recall_in_rl_by_ay'] = ins.groupby('ay').in_rl.mean().round(4).to_dict(); out['recall_in_rl_overall'] = float(ins.in_rl.mean())
# 등록일 정합: 등록번호 있는 건의 권리목록 등록일 = 제공 등록일 (C02 s2: 월 불일치 0) — 여기서 일 단위 재확인
kd = pd.read_parquet(ROOT/'공모전데이터/작업데이터/kr_registered.parquet', columns=['출원번호', '등록일자']); kd.columns = ['appl_no', 'rd']; kd['rd'] = kd.rd.astype(str)
x = rl.drop_duplicates('appl_no').merge(kd, on='appl_no'); out['rgst_date_equal_share'] = float((x.rgst_date == x.rd).mean()); out['rgst_date_compared'] = int(len(x))
# 공란 등록 중복·다출원인
bp = pd.read_parquet(WC/'blank_pairs_2009_2024.parquet'); w = bp[(bp.ay >= 2020) & (bp.ay <= 2023)]
out['blank_window_pairs'] = int(len(w)); out['blank_window_patents'] = int(w.appl_no.nunique()); out['blank_multi_firm_patents'] = int((w.groupby('appl_no').size() > 1).sum())
out['blank_rgst_month'] = w.drop_duplicates('appl_no').rgst_date.str[:6].value_counts().sort_index().to_dict()
json.dump(out, open(WC/'s7_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)); print('done', round(time.time() - t0))
