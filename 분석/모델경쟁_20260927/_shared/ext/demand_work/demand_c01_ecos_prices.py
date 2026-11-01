# ECOS 품목별 물가지수(생산자·수입·수출) 연간 수집. 2021년까지만(최종시험 봉인). 키는 파일에서 읽음.
import requests, time, json, sys, pandas as pd, pathlib
import demand_keys
K = demand_keys.ecos_key()
OUT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext")
Y0, Y1 = sys.argv[1] if len(sys.argv) > 1 else '1995', sys.argv[2] if len(sys.argv) > 2 else '2021'
TABLES = {'404Y016': 'ppi_item', '404Y014': 'ppi_basic', '401Y017': 'imp_item', '402Y016': 'exp_item'}
frames = []
for t, lab in TABLES.items():
    start = 1
    while True:
        url = f'https://ecos.bok.or.kr/api/StatisticSearch/{K}/json/kr/{start}/{start+99999}/{t}/A/{Y0}/{Y1}'
        for att in range(4):
            try:
                r = requests.get(url, timeout=180); r.raise_for_status(); j = r.json(); break
            except Exception as e:
                print('retry', t, att, type(e).__name__); time.sleep(5 * 2 ** att)
        else:
            raise SystemExit('ECOS 실패 — 중단')
        if 'StatisticSearch' not in j:
            print('응답 이상', t, str(j)[:200]); raise SystemExit(1)
        s = j['StatisticSearch']; rows = s['row']
        df = pd.DataFrame(rows); df['table'] = t; df['series'] = lab; frames.append(df)
        print(t, lab, 'total', s['list_total_count'], 'got', len(rows))
        if start + 99999 >= s['list_total_count']: break
        start += 100000; time.sleep(2)
    time.sleep(2)
d = pd.concat(frames, ignore_index=True)
d = d[['table', 'series', 'ITEM_CODE1', 'ITEM_NAME1', 'ITEM_CODE2', 'ITEM_NAME2', 'UNIT_NAME', 'WGT', 'TIME', 'DATA_VALUE']]
d.columns = ['table', 'series', 'item_code', 'item_name', 'cur_code', 'cur_name', 'unit', 'wgt', 'year', 'value']
d['year'] = d['year'].astype(int); d['value'] = pd.to_numeric(d['value'], errors='coerce'); d['wgt'] = pd.to_numeric(d['wgt'], errors='coerce')
fn = OUT / f'demand_ecos_price_item_annual_{Y0}_{Y1}.parquet'
d.to_parquet(fn, index=False); print('saved', fn, d.shape)
