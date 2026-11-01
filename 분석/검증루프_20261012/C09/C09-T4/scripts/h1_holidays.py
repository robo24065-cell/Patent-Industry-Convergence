# C09-T4 h1: 대한민국 공휴일 1999–2026 구성 + 출처 대조.
#  출처1 = vacanza/holidays v0.83 south_korea.py·chinese.py(MIT, GitHub 공개) — 특별공휴일(선거·임시) 목록·음력표를 '텍스트로만' 파싱(코드 실행 안 함)
#  출처2 = 한국천문연구원 음력→양력 변환(astro.kasi.re.kr/life/lunc, 공개) — 설날·부처님오신날·추석 양력일(우선 사용)
#  출처3 = holidays.hyunbin.page/basic.json(우주항공청·천문연 월력요항 기반, 2018~) — 연도별 목록 대조
#  구성 규칙은 「관공서의 공휴일에 관한 규정」 연혁을 이 스크립트에 직접 구현한다.
import os, re, json, sys
import numpy as np, pandas as pd
from datetime import date, timedelta
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
from c9lib import *
DL = os.path.join(W, 'download')
MON = dict(JAN=1, FEB=2, MAR=3, APR=4, MAY=5, JUN=6, JUL=7, AUG=8, SEP=9, OCT=10, NOV=11, DEC=12)
txt = open(os.path.join(DL, 'vacanza', 'chinese.py'), encoding='utf-8').read()


def table(name):
    m = re.search(name + r' = \{(.*?)\n    \}', txt, re.S)
    return {int(y): date(int(y), MON[mo], int(d)) for y, mo, d in re.findall(r'(\d{4}): \((\w{3}), (\d+)\)', m.group(1))}


V = {}
for k in ['LUNAR_NEW_YEAR', 'BUDDHA_BIRTHDAY', 'MID_AUTUMN']:
    t = table(k + '_DATES'); t.update(table('KOREAN_CALENDAR_' + k + '_DATES')); V[k] = t
K = json.load(open(os.path.join(C, 'kasi_lunar.json')))
KK = {'LUNAR_NEW_YEAR': '01-01', 'BUDDHA_BIRTHDAY': '04-08', 'MID_AUTUMN': '08-15'}
cmp_rows = []
for k, md in KK.items():
    for y in range(1999, 2027):
        kd = date.fromisoformat(K[f'{y}-{md}']); vd = V[k].get(y)
        cmp_rows.append(dict(holiday=k, year=y, kasi=str(kd), vacanza=str(vd), same=kd == vd))
cmp_l = pd.DataFrame(cmp_rows)
print('음력 공휴일 KASI vs vacanza 불일치', int((~cmp_l.same).sum()), '/', len(cmp_l))
LUN = {k: {y: date.fromisoformat(K[f'{y}-{md}']) for y in range(1999, 2027)} for k, md in KK.items()}   # KASI(공식) 우선

# 특별공휴일(선거·임시) — vacanza StaticHolidays 목록 파싱
st = open(os.path.join(DL, 'vacanza', 'south_korea.py'), encoding='utf-8').read()
blk = st[st.index('special_public_holidays = {'):st.index('# Pre-2014 Alternate Holidays')]
special = []
for y, body in re.findall(r'\n        (\d{4}): (.*?)(?=\n        \d{4}:|\n    \})', blk, re.S):
    y = int(y)
    for mo, d in re.findall(r'\((\w{3}), (\d+), ', body):
        if y >= 1999:
            special.append(date(y, MON[mo], int(d)))


def nth_wed(y, m, n):
    d = date(y, m, 1); d += timedelta((2 - d.weekday()) % 7)
    return d + timedelta(7 * (n - 1))


def elections(y):
    out = []
    if y == 2020: out.append(date(2020, 4, 15))
    elif y >= 2007 and (y - 2008) % 4 == 0: out.append(nth_wed(y, 4, 2))
    if 2007 <= y <= 2024 and (y - 2007) % 5 == 0:
        if y <= 2012: out.append(nth_wed(y, 12, 3))
        elif y >= 2022: out.append(nth_wed(y, 3, 2))
    if y >= 2007 and (y - 2010) % 4 == 0:
        j6 = date(y, 6, 6)
        out.append(nth_wed(y, 6, 2) if j6.weekday() in (1, 2, 3) else nth_wed(y, 6, 1))
    return out


def year_holidays(y):
    H = {}
    def add(d, nm): H.setdefault(d, []).append(nm)
    add(date(y, 1, 1), '신정')
    three = {}
    for k, nm in [('LUNAR_NEW_YEAR', '설날'), ('MID_AUTUMN', '추석')]:
        c0 = LUN[k][y]
        for dd in (-1, 0, 1):
            add(c0 + timedelta(dd), nm); three[c0 + timedelta(dd)] = nm
    obs = []   # (날짜, 이름, 대체 적용 시작연도)
    add(date(y, 3, 1), '삼일절'); obs.append((date(y, 3, 1), '삼일절', 2022))
    if y <= 2005: add(date(y, 4, 5), '식목일')
    add(LUN['BUDDHA_BIRTHDAY'][y], '부처님오신날'); obs.append((LUN['BUDDHA_BIRTHDAY'][y], '부처님오신날', 2023))
    add(date(y, 5, 5), '어린이날'); obs.append((date(y, 5, 5), '어린이날', 2014))
    add(date(y, 6, 6), '현충일')
    if y <= 2007 or y >= 2026: add(date(y, 7, 17), '제헌절')   # 2026 재지정(월력요항 2026)
    add(date(y, 8, 15), '광복절'); obs.append((date(y, 8, 15), '광복절', 2021))
    add(date(y, 10, 3), '개천절'); obs.append((date(y, 10, 3), '개천절', 2021))
    if y >= 2013:
        add(date(y, 10, 9), '한글날'); obs.append((date(y, 10, 9), '한글날', 2021))
    add(date(y, 12, 25), '성탄절'); obs.append((date(y, 12, 25), '성탄절', 2023))
    for d in elections(y): add(d, '선거일')
    for d in special:
        if d.year == y and d not in H: add(d, '임시·선거(특별)')
    if y >= 2014:
        base = set(H)
        cand = []
        for d, nm in sorted(three.items()):
            if d.weekday() == 6 or len(H[d]) > 1: cand.append((d, nm))
        for d, nm, since in obs:
            if y >= since and (d.weekday() >= 5 or len(H[d]) > 1) and d not in three: cand.append((d, nm))
        extra = set()
        seen = set()
        for d, nm in sorted(cand):
            if d in seen: continue        # 같은 날 겹친 공휴일은 대체공휴일 1일(2025-05-05 어린이날·부처님오신날 → 05-06만)
            seen.add(d)
            e = d + timedelta(1)
            while e.weekday() >= 5 or e in base or e in extra: e += timedelta(1)
            extra.add(e); add(e, f'대체공휴일({nm})')
    add(date(y, 5, 1), '근로자의 날')
    return H


rows = []
for y in range(1999, 2027):
    for d, nms in year_holidays(y).items():
        rows.append(dict(date=str(d), name='|'.join(nms), weekday=d.weekday()))
hol = pd.DataFrame(rows).groupby('date').agg(name=('name', lambda s: '|'.join(s)), weekday=('weekday', 'first')).reset_index()
hol = hol[(hol.date >= '1999-01-01') & (hol.date <= '2026-12-31')].reset_index(drop=True)
hol.to_csv(os.path.join(C, 'holidays_kr_2000_2026.csv'), index=False, encoding='utf-8-sig')
# 출처3 대조(2018~2026): 근로자의 날은 관공서 공휴일이 아니므로 제외하고 비교
hb = json.load(open(os.path.join(DL, 'hyunbin', 'basic.json'), encoding='utf-8'))
hbset = {d: (v if isinstance(v, list) else [v]) for y, dd in hb.items() for d, v in dd.items()}
mine = {}
for d, n in zip(hol.date, hol.name):
    n2 = '|'.join(x for x in n.split('|') if x != '근로자의 날')
    if n2 and d[:4] in hb: mine[d] = n2
only_mine = sorted((d, mine[d]) for d in mine if d not in hbset)
only_hb = sorted((d, hbset[d]) for d in hbset if d[:4] <= '2026' and d not in mine)
print('hyunbin 연도', min(hb), '~', max(hb), '| 내 목록에만', only_mine, '| hyunbin에만', only_hb)
hol['y'] = hol.date.str[:4].astype(int)
cnt = hol[hol.y.between(1999, 2026)].groupby('y').size()
cnt_wd = hol[hol.y.between(1999, 2026) & (hol.weekday < 5)].groupby('y').size()
hbc = pd.Series({int(y): len(v) for y, v in hb.items()})
out = dict(n_lunar_mismatch=int((~cmp_l.same).sum()), lunar_mismatch=cmp_l[~cmp_l.same].to_dict('records'),
           special_parsed=[str(d) for d in special], only_mine_vs_hyunbin=only_mine, only_hyunbin=only_hb,
           count_by_year=cnt.to_dict(), weekday_count_by_year=cnt_wd.to_dict(), hyunbin_count_by_year=hbc.to_dict())
json.dump(out, open(os.path.join(C, 'h1_holidays.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
cmp_l.to_csv(os.path.join(C, 'h1_lunar_compare.csv'), index=False)
print(pd.DataFrame({'all(근로자의날 포함)': cnt, 'weekday': cnt_wd, 'hyunbin(관공서)': hbc}).to_string())
print('special', out['special_parsed'])
