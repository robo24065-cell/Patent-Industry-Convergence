# -*- coding: utf-8 -*-
"""C18-T1 verify v1: pandas 없이 순수 sqlite + pyarrow + numpy 로 (a) 월별 RF·KR·RLin·RLin−RF·합집합 집계, (b) T1 Δ(AUC·skill)
독립 재계산. 원천은 mode=ro ATTACH. 메모리 DB만 쓴다(디스크 쓰기는 C18-T1/verify/v1_out.json 뿐)."""
import sys, pathlib, json, sqlite3, time, csv
import numpy as np, pyarrow.parquet as pq
sys.dont_write_bytecode = True
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
H = B/'분석/검증루프_20261012/C18/C18-T1'
END = '20250822'
con = sqlite3.connect('file::memory:', uri=True)
con.execute(f"attach database '{(B/'수집파이프라인/registry_full.sqlite').as_uri()}?mode=ro' as rf")
con.execute(f"attach database '{(B/'수집파이프라인/business_rights.sqlite').as_uri()}?mode=ro' as br")
kp = pq.read_table(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '등록번호', '출원연도']).to_pydict()
con.execute('create temp table P(appl_no text primary key, blank int, ay int)')
con.executemany('insert or ignore into P values (?,?,?)', ((a.strip(), 0 if (r or '').strip().isdigit() and len((r or '').strip()) == 13 else 1, int(y))
                                                            for a, r, y in zip(kp['출원번호'], kp['등록번호'], kp['출원연도'])))
kr = pq.read_table(B/'공모전데이터/작업데이터/kr_registered.parquet', columns=['등록번호', '등록일자']).to_pydict()
con.execute('create temp table KR(rgst_no text primary key, rd text)')
con.executemany('insert or ignore into KR values (?,?)', ((r.strip(), d.strip()) for r, d in zip(kr['등록번호'], kr['등록일자']) if r.strip()[:2] == '10'))
con.execute("create temp table RF as select rgst_no, rgstDate rd from rf.patent where substr(rgst_no,1,2)='10' and rgstDate<=?", (END,))
con.execute('create index t_rf on RF(rgst_no)')
con.execute("""create temp table RL as select rgst_no, min(rgst_date) rd, min(appl_no) appl_no from br.right_list
               where substr(rgst_no,1,2)='10' and rgst_date<=? and appl_no in (select appl_no from P) group by rgst_no""", (END,))
con.execute('create index t_rl on RL(rgst_no)')
q = lambda s: dict(con.execute(s).fetchall())
rfm = q("select substr(rd,1,6), count(*) from RF group by 1")
krm = q(f"select substr(rd,1,6), count(*) from KR where rd<='{END}' group by 1")
rlm = q("select substr(rd,1,6), count(*) from RL group by 1")
mis = q("select substr(rd,1,6), count(*) from RL where rgst_no not in (select rgst_no from RF) and rgst_no not in (select rgst_no from KR) group by 1")
misb = q("select substr(RL.rd,1,6), count(*) from RL join P on RL.appl_no=P.appl_no where P.blank=1 and RL.rgst_no not in (select rgst_no from RF) group by 1")
# 합집합(월 우선순위 RF>KR>RL; RF=KR이므로 RF ∪ (RL−RF))
uni = {k: rfm.get(k, 0) + mis.get(k, 0) for k in set(rfm) | set(mis)}
o = {'kr_minus_rf': con.execute("select count(*) from KR where rgst_no not in (select rgst_no from rf.patent)").fetchone()[0],
     'rf_minus_kr': con.execute("select count(*) from rf.patent where substr(rgst_no,1,2)='10' and rgst_no not in (select rgst_no from KR)").fetchone()[0]}
# 비교 대상: monthly_capture.csv(s1, pandas 경로)
rows = list(csv.DictReader(open(H/'monthly_capture.csv', encoding='utf-8-sig')))
diffs = []
for r in rows:
    ym = r['ym']
    a = np.array([float(r['RF']), float(r['KR']), float(r['RLin']), float(r['RLin_not_RF']), float(r['union']), float(r['miss_kp_blank'])])
    b = np.array([rfm.get(ym, 0), krm.get(ym, 0), rlm.get(ym, 0), mis.get(ym, 0), uni.get(ym, 0), misb.get(ym, 0)], dtype=float)
    if np.any(a != b): diffs.append([ym, a.tolist(), b.tolist()])
o['monthly_months_compared'] = len(rows); o['monthly_months_differ'] = len(diffs); o['monthly_diffs_head'] = diffs[:10]
cm = {r['ym']: (rfm.get(r['ym'], 0) / uni[r['ym']] if uni.get(r['ym']) else None) for r in rows}
o['C_m_recomputed_gap'] = {k: round(v, 6) for k, v in cm.items() if v is not None and v < 0.8}
# T1 창 누락 문서(출원 2020–23)
o['T1_window_missing_docs'] = con.execute("""select count(distinct P.appl_no) from RL join P on RL.appl_no=P.appl_no
     where P.blank=1 and P.ay between 2020 and 2023 and RL.rgst_no not in (select rgst_no from RF)""").fetchone()[0]
# 2024 비율·전년비(공식 월별: s1_official_month.csv)
off = {r['ym']: int(r['O_pat']) for r in csv.DictReader(open(H/'cache/s1_official_month.csv', encoding='utf-8-sig'))}
def yr(y, d): return sum(v for k, v in d.items() if k and k.startswith(y))
rat = {y: yr(y, rfm) / yr(y, off) for y in ['2019', '2020', '2021', '2022', '2023', '2024']}
o['ratio_by_year'] = rat; o['ratio_2024_vs_mean1923'] = rat['2024'] / np.mean([rat[y] for y in ['2019', '2020', '2021', '2022', '2023']]) - 1
o['raw_yoy_2024'] = yr('2024', rfm) / yr('2023', rfm) - 1; o['official_yoy_2024'] = yr('2024', off) / yr('2023', off) - 1
o['ratio_2025_jan_jul_vs_ref'] = (sum(rfm.get(f'2025{m:02d}', 0) for m in range(1, 8)) / sum(off[f'2025{m:02d}'] for m in range(1, 8))) / 0.2600397085620072 - 1
con.close()
# ---------- (b) T1 Δ: 순수 numpy 순위 AUC(동점 중간순위) ----------
ev = pq.read_table(H/'cache/s5_eval.parquet').to_pydict(); L = np.load(H/'cache/s5_labels.npz')
def auc(y, s):
    o_ = np.argsort(s, kind='mergesort'); ss = s[o_]; r = np.empty(len(s)); i = 0
    # 동점 중간순위
    _, idx, cnt = np.unique(ss, return_index=True, return_counts=True)
    mid = idx + (cnt - 1) / 2.0 + 1; r[o_] = np.repeat(mid, cnt)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
s = np.asarray(ev['s']); BC = [k for k in ev if k.startswith('b_')]
res = {}
for nm in ['y0', 'S1_R_self_G', 'S2_R_any_G', 'S1b_R_self_all']:
    y = L[nm].astype(np.int64); m = auc(y, s); bb = {k: auc(y, np.asarray(ev[k])) for k in BC}; st = max(bb.values())
    res[nm] = dict(pos=int(y.sum()), auc=m, strongest=st, skill=(m - st) / (1 - st), top1000=int(y[np.argsort(-s, kind='stable')[:1000]].sum()))
for nm in ['S1_R_self_G', 'S2_R_any_G', 'S1b_R_self_all']:
    res[nm]['d_auc'] = res[nm]['auc'] - res['y0']['auc']; res[nm]['d_skill'] = res[nm]['skill'] - res['y0']['skill']
o['t1'] = res; o['secs'] = round(time.time() - t0)
json.dump(o, open(H/'verify/v1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: o[k] for k in o if k not in ('monthly_diffs_head',)}, ensure_ascii=False)[:3000]); print('done', o['secs'])
