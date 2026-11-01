# -*- coding: utf-8 -*-
"""R&D 투자 통계 수집 (science_ 트랙, 2026-09-28)
1) OECD ANBERD (ISIC Rev.4 산업별 기업 R&D, SDMX 공개 API, 키 불필요)
   -> science_oecd_anberd_isic4.parquet
2) ECOS 기업경영분석 '연구개발비대매출액'(%) 업종별(KSIC 2~3자리), 기업규모 종합·대·중소
   501Y006(11차 KSIC, 2009~), 501Y026(2007~2010 표본), 501Y027(1998~2007 표본)
   -> science_ecos_rd_intensity.parquet
키는 API키_기록.txt에서 읽음(값 미기록). 요청 간 1초.
"""
import io, os, re, json, time, urllib.request
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(OUT, '..', '..', '..', '..'))


def anberd():
    url = ('https://sdmx.oecd.org/public/rest/data/OECD.STI.STP,DSD_ANBERD@DF_ANBERDi4,1.0/'
           'KOR+USA+JPN+CHN+DEU+FRA+GBR.A.......?startPeriod=1995')
    req = urllib.request.Request(url, headers={'Accept': 'application/vnd.sdmx.data+csv;version=1.0.0',
                                               'User-Agent': 'patent-contest-research/1.0'})
    raw = urllib.request.urlopen(req, timeout=180).read().decode('utf-8')
    d = pd.read_csv(io.StringIO(raw))
    d = d[['REF_AREA', 'CRITERIA', 'ACTIVITY', 'UNIT_MEASURE', 'PRICE_BASE', 'MEASURE', 'TIME_PERIOD',
           'OBS_VALUE', 'OBS_STATUS', 'UNIT_MULT']]
    d.to_parquet(os.path.join(OUT, 'science_oecd_anberd_isic4.parquet'), index=False)
    print('anberd', d.shape)


def ecos():
    t = open(os.path.join(ROOT, 'API키_기록.txt'), encoding='utf-8').read()
    key = re.search(r'키:\s*(\S+)', t.split('[한국은행 ECOS')[1]).group(1)
    rows = []
    for code, y0, y1 in [('501Y006', 2009, 2024), ('501Y026', 2007, 2010), ('501Y027', 1998, 2007)]:
        for size in ['A', 'L', 'M']:
            url = f'https://ecos.bok.or.kr/api/StatisticSearch/{key}/json/kr/1/20000/{code}/A/{y0}/{y1}/?/{size}/615'
            d = json.load(urllib.request.urlopen(url, timeout=120))
            k = list(d)[0]
            r = d[k].get('row', [])
            print(code, size, len(r), d[k].get('list_total_count', d[k]) if not r else '')
            for x in r:
                rows.append({'stat_code': code, 'ind_code': x['ITEM_CODE1'], 'ind_name': x['ITEM_NAME1'],
                             'size': x['ITEM_CODE2'], 'year': int(x['TIME']), 'rd_to_sales_pct': float(x['DATA_VALUE']) if x['DATA_VALUE'] not in (None, '', '-') else None})
            time.sleep(1)
    df = pd.DataFrame(rows)
    df.to_parquet(os.path.join(OUT, 'science_ecos_rd_intensity.parquet'), index=False)
    print('ecos', df.shape)


if __name__ == '__main__':
    anberd()
    ecos()
