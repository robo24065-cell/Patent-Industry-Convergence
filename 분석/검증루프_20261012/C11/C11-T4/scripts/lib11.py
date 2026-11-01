# C11-T4 공용
import re, os, sys, json
import numpy as np, pandas as pd

B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
MC = B + r'\분석\모델경쟁_20260927'
META = (MC + r'\_shared\meta\*.parquet').replace('\\', '/')
TXTFREE = (MC + r'\T1_산업대기실\TXT_R9\cache\kr_txtfree.parquet').replace('\\', '/')
W = B + r'\분석\검증루프_20261012\C11\C11-T4'
C = W + r'\cache'
CUT = 20250101           # 캐시 상한(이 날짜 이전 공개만)
TRAIN_Y = [2017, 2018, 2019]
EVAL_Y = [2020, 2021, 2022, 2023, 2024]
ALL_Y = TRAIN_Y + EVAL_Y
SEED = 20261013

SUF = [
    '股份有限公司', '有限责任公司', '有限公司', '股份公司', '株式会社', '有限会社',
    '주식회사', '유한책임회사', '유한회사', '합자회사', '합명회사', '농업회사법인', '영농조합법인',
    '(주)', '㈜', '（주）', '(유)', '(株)',
]
SUF_RE = re.compile('|'.join(re.escape(s) for s in SUF))
LAT_RE = re.compile(r'\b(CO|COMPANY|LTD|LIMITED|INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C|GMBH|AG|SA|SAS|SARL|BV|B\.V|NV|N\.V|AB|OY|OYJ|SPA|S\.P\.A|PLC|KK|KABUSHIKI|KAISHA|KG|SE|AS|A/S)\b\.?')
NONALNUM = re.compile(r'[^0-9A-Z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7a3]')
PAREN = re.compile(r'[\(（][^\)）]*[\)）]')
GEN = re.compile(r'\b(GROUP|HOLDINGS|HOLDING|THE)\b|集团|控股|股份')
KR_CORP = re.compile(r'주식회사|\(주\)|㈜|유한회사|유한책임회사|합자회사|합명회사|농업회사법인|영농조합법인')


def norm_key(s):
    s = SUF_RE.sub(' ', str(s).upper())
    s = LAT_RE.sub(' ', s)
    return NONALNUM.sub('', s)


def aggr_key(s):
    s = PAREN.sub(' ', str(s).upper())
    s = SUF_RE.sub(' ', s)
    s = LAT_RE.sub(' ', s)
    s = GEN.sub(' ', s)
    return NONALNUM.sub('', s)


def split_app(s):
    if s is None:
        return []
    return [x.strip() for x in re.split(r'[|;]', str(s)) if x.strip() and x.strip().lower() not in ('nan', 'none', '-', 'null')]


def ymd(y, m=1, d=1):
    return y * 10000 + m * 100 + d


def win(y):
    """결과 창 [Y0101, Y0630]"""
    return ymd(y, 1, 1), ymd(y, 6, 30)


def con_duck():
    import duckdb
    con = duckdb.connect()
    con.execute('set threads=2'); con.execute("set memory_limit='3GB'"); con.execute('set enable_progress_bar=false')
    return con


def out(*a):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    print(*a, flush=True)


# ---------- 공개 배정 표 (KR TXT-FREE 라벨의 IPC 최빈표) ----------
def build_pub_tables(kr, t0):
    """kr: DataFrame(pub, sc, mg, sg, kt) KR 문서(kt = TXT-FREE). t0 이전 공개만으로 H-sg-3, H-sc-1 표."""
    d = kr[(kr.pub < t0) & kr.kt.notna()]
    tabs = {}
    for lev in ['sg', 'mg', 'sc']:
        g = d.groupby([lev, 'kt']).size().rename('n').reset_index()
        tot = g.groupby(lev).n.sum()
        g = g.sort_values([lev, 'n', 'kt'], ascending=[True, False, True]).drop_duplicates(lev)
        g['tot'] = g[lev].map(tot)
        tabs[lev] = g.set_index(lev)
    return tabs


def apply_pub(df, tabs, rule):
    """df: sc, mg, sg 열. rule 'H-sg-3' or 'H-sc-1' → 라벨 Series(없으면 NaN)."""
    if rule == 'H-sc-1':
        t = tabs['sc']
        return df.sc.map(t.kt.where(t.tot >= 1))
    res = pd.Series(np.nan, index=df.index, dtype=object)
    for lev in ['sg', 'mg', 'sc']:
        t = tabs[lev]
        m = df[lev].map(t.kt.where(t.tot >= 3))
        res = res.where(res.notna(), m)
    return res
