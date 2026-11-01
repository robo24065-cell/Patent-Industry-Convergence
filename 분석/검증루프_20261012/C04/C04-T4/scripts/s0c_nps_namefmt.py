# s0c: 상호 표기 형식 변화 점검(형식 수준, n_emp 미사용). 전국 법인 유형코드 행 전체 기준.
# 연속 스냅샷 쌍에서, 뒤 스냅샷 행 중 앞 스냅샷에 (a) 원 상호+biz6 (b) 정규화 상호+biz6 (c) biz6+적용일(ba) 으로 있던 비율.
# (c)는 있는데 (a)(b)가 없으면 같은 사업장의 상호 표기가 바뀐 것이다.
import sys, re, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
EXT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext'
OUT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C04\C04-T4\cache'
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
SN = ['201512', '201612', '201712', '201812', '201912', '202012', '202112', '202212', '202312', '202412', '202512', '202608']
def load(s):
    d = pd.read_parquet(EXT + rf'\firm_nps_wkpl_{s}.parquet', columns=['name', 'biz6', 'adpt_dt'])
    d = d[d.biz6.astype(str).str[3:5].isin([str(x) for x in range(81, 89)])]
    d['raw'] = d.biz6 + '|' + d.name.astype(str)
    d['nk'] = d.biz6 + '|' + d.name.map(norm)
    d['ba'] = d.biz6 + '|' + d.adpt_dt.fillna('').astype(str)
    d['nm'] = d.name.astype(str)
    return d
rows = []
prev = load(SN[0])
for s in SN[1:]:
    cur = load(s)
    r = dict(prev=SN[SN.index(s) - 1], cur=s)
    r['rows_cur'] = len(cur)
    a = cur.raw.isin(set(prev.raw)); b = cur.nk.isin(set(prev.nk)); c = cur.ba.isin(set(prev.ba))
    r['in_prev_raw'] = round(float(a.mean()), 4); r['in_prev_norm'] = round(float(b.mean()), 4); r['in_prev_ba'] = round(float(c.mean()), 4)
    r['ba_but_not_norm'] = round(float((c & ~b).mean()), 4); r['ba_but_not_raw'] = round(float((c & ~a).mean()), 4)
    # 상호 표기 변화 예시 유형(접미 표기)
    x = cur[c & ~a].head(200000)
    pm = prev.drop_duplicates('ba').set_index('ba')['nm']
    pairs = pd.DataFrame({'old': x.ba.map(pm).values, 'new': x['nm'].values})
    r['suffix_pattern_top'] = json.dumps(pd.Series([('(주)' in o, '주식회사' in o, '(주)' in n, '주식회사' in n) for o, n in zip(pairs.old.astype(str), pairs.new.astype(str))]).astype(str).value_counts().head(4).to_dict(), ensure_ascii=False)
    rows.append(r); print(r, flush=True)
    prev = cur
pd.DataFrame(rows).to_csv(OUT + r'\s0c_nps_namefmt.csv', index=False, encoding='utf-8-sig')
