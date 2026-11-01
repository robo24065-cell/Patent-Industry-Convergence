# -*- coding: utf-8 -*-
"""OpenAlex 논문 수 수집 (science_ 트랙, 2026-09-28)
KSIC 50개 각각에 대한 일반 영어 키워드(KSIC 공식 영문명·해설 기반, 제공 데이터 내용 미사용)로
title_and_abstract 검색 → 국가(저자 소속국)별·연도별 논문 수 (group_by=publication_year).
키 불필요, mailto 등 개인정보 미포함. 요청 간 1.2초, 429/5xx 지수 백오프, 403 즉시 중단.
출력: science_openalex_ksic_country_year.parquet, science_openalex_country_total_year.parquet
"""
import json, time, sys, urllib.request, urllib.parse, os
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
KW = {
 '15220': ['shoe sole', 'outsole', 'insole', 'footwear'],
 '20202': ['synthetic resin', 'thermoplastic resin', 'polyolefin', 'engineering plastic'],
 '20411': ['coating composition', 'waterborne coating', 'anticorrosive coating', 'paint coating'],
 '20423': ['cosmetic', 'skin care', 'sunscreen', 'perfume'],
 '20491': ['photoresist', 'photosensitive material', 'photographic film'],
 '21220': ['herbal medicine', 'traditional Chinese medicine', 'herbal extract'],
 '23992': ['abrasive', 'grinding wheel', 'polishing pad'],
 '24191': ['coated steel', 'galvanized steel', 'steel sheet coating'],
 '24219': ['nonferrous metallurgy', 'hydrometallurgy', 'rare earth extraction', 'smelting'],
 '25914': ['metal stamping', 'sheet metal forming', 'press forming'],
 '25922': ['electroplating', 'electroless plating'],
 '25924': ['laser cutting', 'laser engraving', 'waterjet cutting'],
 '25932': ['door hinge', 'door lock', 'furniture hardware'],
 '26111': ['nonvolatile memory', 'DRAM', 'NAND flash', 'resistive memory'],
 '26112': ['integrated circuit', 'system on chip', 'CMOS image sensor', 'microprocessor'],
 '26121': ['light emitting diode', 'GaN LED', 'micro LED'],
 '26129': ['power semiconductor', 'MOSFET', 'IGBT', 'photodiode'],
 '26211': ['liquid crystal display', 'LCD panel', 'backlight unit'],
 '26212': ['organic light emitting diode', 'OLED display'],
 '26223': ['printed circuit board', 'flexible circuit board'],
 '26224': ['surface mount technology', 'solder joint', 'PCB assembly'],
 '26291': ['capacitor', 'supercapacitor', 'multilayer ceramic capacitor'],
 '26322': ['computer monitor', 'display monitor'],
 '26323': ['printer', 'inkjet printing', 'laser printer'],
 '26422': ['mobile phone', 'smartphone', 'handset'],
 '26429': ['wireless communication', 'antenna', 'base station'],
 '26519': ['video camera', 'television receiver', 'projector'],
 '26529': ['loudspeaker', 'microphone', 'earphone', 'headphone'],
 '27199': ['medical device', 'surgical instrument', 'orthopedic implant'],
 '27211': ['radar', 'navigation system', 'GNSS'],
 '27212': ['electrical measurement', 'oscilloscope', 'current sensor', 'voltage measurement'],
 '27301': ['optical lens', 'lens design', 'diffractive optical element'],
 '28111': ['electric motor', 'permanent magnet motor', 'synchronous motor', 'electric generator'],
 '28122': ['circuit breaker', 'switchgear', 'electrical connector'],
 '28511': ['rice cooker', 'microwave oven', 'refrigerator', 'kitchen appliance'],
 '28902': ['electrical insulator', 'carbon brush', 'graphite electrode'],
 '29131': ['centrifugal pump', 'liquid pump'],
 '29132': ['compressor', 'vacuum pump'],
 '29133': ['control valve', 'check valve', 'solenoid valve'],
 '29150': ['industrial furnace', 'burner', 'kiln'],
 '29162': ['crane', 'elevator', 'hoist'],
 '29210': ['agricultural machinery', 'tractor', 'harvester'],
 '29229': ['machine tool', 'CNC machining'],
 '29241': ['excavator', 'construction machinery', 'mining machinery'],
 '29280': ['industrial robot', 'robotic arm', 'robot manipulator'],
 '30393': ['vehicle seat', 'car seat', 'automotive seat'],
 '31202': ['railway vehicle', 'rolling stock', 'bogie'],
 '31312': ['unmanned aerial vehicle', 'drone', 'UAV'],
 '33301': ['fitness equipment', 'exercise machine', 'treadmill'],
 '33309': ['golf club', 'sports equipment', 'tennis racket', 'golf ball'],
}
COUNTRIES = ['ALL', 'KR', 'US', 'JP', 'CN', 'DE']
BASE = 'https://api.openalex.org/works'
UA = {'User-Agent': 'patent-contest-research/1.0'}


def call(params):
    url = BASE + '?' + urllib.parse.urlencode(params)
    wait = 5
    for attempt in range(6):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                rem = r.headers.get('X-RateLimit-Remaining')
                return json.load(r), rem
        except urllib.error.HTTPError as e:
            if e.code == 403:
                print('403 -> stop', file=sys.stderr); sys.exit(2)
            if e.code in (429, 500, 502, 503, 504):
                print('retry', e.code, wait, file=sys.stderr); time.sleep(wait); wait *= 2; continue
            raise
        except Exception as e:
            print('err', e, wait, file=sys.stderr); time.sleep(wait); wait *= 2
    raise RuntimeError('failed ' + url)


def main():
    part = os.path.join(OUT, 'science_openalex_ksic_country_year.parquet')
    done = pd.read_parquet(part) if os.path.exists(part) else pd.DataFrame(columns=['ksic', 'country', 'year', 'n_works', 'query'])
    have = set(zip(done.ksic, done.country))
    rows = done.to_dict('records')
    for k, kws in KW.items():
        q = '(' + ' OR '.join('"%s"' % w for w in kws) + ')'
        for c in COUNTRIES:
            if (k, c) in have:
                continue
            f = 'title_and_abstract.search:%s,publication_year:1995-2025' % q
            if c != 'ALL':
                f += ',authorships.countries:%s' % c
            d, rem = call({'filter': f, 'group_by': 'publication_year'})
            got = {int(g['key']): g['count'] for g in d['group_by']}
            for y in range(1995, 2026):
                rows.append({'ksic': k, 'country': c, 'year': y, 'n_works': got.get(y, 0), 'query': q})
            print(k, c, d['meta']['count'], 'remaining', rem, flush=True)
            time.sleep(1.2)
        pd.DataFrame(rows).to_parquet(part, index=False)
        if rem is not None and int(rem) < 150:
            print('budget guard stop', flush=True); break
    # 국가별 전체 논문 수(정규화용)
    tot = []
    for c in COUNTRIES:
        f = 'publication_year:1995-2025' + ('' if c == 'ALL' else ',authorships.countries:%s' % c)
        d, rem = call({'filter': f, 'group_by': 'publication_year'})
        for g in d['group_by']:
            tot.append({'country': c, 'year': int(g['key']), 'n_works_total': g['count']})
        time.sleep(1.2)
    pd.DataFrame(tot).to_parquet(os.path.join(OUT, 'science_openalex_country_total_year.parquet'), index=False)
    print('done', len(rows))


if __name__ == '__main__':
    main()
