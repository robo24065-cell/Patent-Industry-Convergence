# -*- coding: utf-8 -*-
"""[2단계 감사 수정] 제공 융합데이터(meta) 해외 패밀리의 '시점 보정' 토큰표.
1단계 kr_family.parquet 는 날짜 없는 스냅샷(2025 수집)이라 기준일 L 이후 생긴 패밀리(미국 계속출원·후발 국가 진입 등)가 섞인다.
여기서는 패밀리 토큰(공개·등록번호)마다 공개연도를 추정하고, 피처 계산 시 '추정 공개연도 <= L연도-1' 인 토큰만 센다.
연도 추정 규칙(보수적으로 범위의 끝 연도 사용):
 - 번호에 연도가 박힌 형식(US2009172929, WO2009075436, JP2009142637, KR20210095648, TW202215998, MX2008013729 등): 그 연도
 - JP 2자리 연호형(JP21142637): 1988+yy
 - EP/US등록/JP등록/KR등록/CN 공개번호: 공개 연도별 번호 앵커로 선형보간 후 올림
 - 그 밖(연도 추정 불가): 제외(보수적)
출력: cache/s2_fam_tokens.parquet (applNo, cc, yr, tok)
"""
import os, re, time, numpy as np, pandas as pd, duckdb
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(os.path.dirname(HERE), '_shared', 'meta', '*.parquet')
duckdb.sql("SET threads=2; SET memory_limit='3GB'")
t = time.time()
d = duckdb.sql(f"select appl_no, any_value(family) AS fam from read_parquet('{M}') where cc='KR' and family is not null and family<>'' group by appl_no").df()
print('KR apps with family', len(d), round(time.time() - t, 1), flush=True)

# 번호 -> 연도 앵커 (번호, 연도말 근사). 보간 후 ceil.
ANCH = {
    'EP': [(1000000, 2000.0), (1500000, 2005.0), (2000000, 2008.9), (2500000, 2012.7), (3000000, 2016.1), (3500000, 2019.5), (4000000, 2022.4), (4500000, 2025.5)],
    'USG': [(6000000, 1999.9), (7000000, 2006.1), (8000000, 2011.6), (9000000, 2015.3), (10000000, 2018.5), (11000000, 2021.4), (12000000, 2024.5)],
    'JPG': [(3000000, 2000.0), (4000000, 2007.9), (5000000, 2012.5), (6000000, 2016.8), (7000000, 2022.3), (7600000, 2025.0)],
    'KRG': [(500000, 2005.4), (800000, 2008.1), (1000000, 2010.9), (1500000, 2015.2), (2000000, 2019.5), (2500000, 2023.2), (2800000, 2025.6)],
}
CN_PFX = {101: 2011, 102: 2013, 103: 2014, 104: 2015, 105: 2016, 106: 2017, 107: 2018, 108: 2018, 109: 2019, 110: 2020, 111: 2020,
          112: 2021, 113: 2021, 114: 2022, 115: 2023, 116: 2023, 117: 2024, 118: 2025, 119: 2025,
          201: 2010, 202: 2012, 203: 2014, 204: 2015, 205: 2016, 206: 2017, 207: 2018, 208: 2019, 209: 2019, 210: 2020, 211: 2020,
          212: 2021, 213: 2021, 214: 2021, 215: 2022, 216: 2022, 217: 2022, 218: 2023, 219: 2023, 220: 2024}


def interp(kind, n):
    a = ANCH[kind]
    xs = [x for x, _ in a]; ys = [y for _, y in a]
    if n < xs[0]: return int(np.ceil(ys[0]))
    return int(np.ceil(np.interp(n, xs, ys, right=ys[-1] + 1)))


def est_year(tok):
    cc, rest = tok[:2], tok[2:]
    if not rest.isdigit():
        return None
    L = len(rest)
    if L >= 9:
        y = int(rest[:4])
        if 1985 <= y <= 2026 and not (cc == 'CN' or (cc == 'KR' and rest.startswith('10') and L == 9)):
            return y
    if cc == 'KR' and L == 13 and rest[:2] in ('10', '20'):   # 출원번호형 10YYYY...
        y = int(rest[2:6]); return y if 1985 <= y <= 2026 else None
    if cc == 'JP' and L == 8 and not rest.startswith('0'):
        yy = int(rest[:2]); return 1988 + yy if 1 <= yy <= 38 else None
    if cc == 'JP' and L == 8 and rest.startswith('0'):
        return interp('JPG', int(rest))
    if cc == 'JP' and L == 7:
        return interp('JPG', int(rest))
    if cc == 'EP' and L in (7, 8):
        return interp('EP', int(rest))
    if cc == 'US' and L in (7, 8):
        return interp('USG', int(rest))
    if cc == 'KR' and L == 9 and rest.startswith('10'):
        return interp('KRG', int(rest[2:]))
    if cc == 'CN':
        if L == 9: return CN_PFX.get(int(rest[:3]))
        if L == 7: return 2007
    return None


rows = []
unk = 0; tot = 0
for a, f in zip(d.appl_no.values, d.fam.values):
    for tok in set(f.split('|')):
        tok = tok.strip()
        if len(tok) < 4: continue
        tot += 1
        y = est_year(tok)
        if y is None:
            unk += 1; continue
        rows.append((a, tok[:2], y, tok))
out = pd.DataFrame(rows, columns=['applNo', 'cc', 'yr', 'tok'])
out['yr'] = out.yr.astype('int16')
out.to_parquet(os.path.join(HERE, 'cache', 's2_fam_tokens.parquet'), index=False)
print(f'tokens {tot:,} dated {len(out):,} unknown {unk:,} ({unk / tot:.3f})', round(time.time() - t, 1))
print(out.groupby('cc').yr.describe().sort_values('count', ascending=False).head(12).to_string())
