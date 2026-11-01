# C13-T4 공용. norm_key는 C12 lib12와 같은 식(복사). 조직 정규식은 PREREG 변경 1 그대로.
import re, sys
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
MC = B + r'\분석\모델경쟁_20260927'
META = (MC + r'\_shared\meta\*.parquet').replace('\\', '/')
W = B + r'\분석\검증루프_20261012\C13\C13-T4'
C = W + r'\cache'
CUT = 20250101
SEED = 20261017

SUF = ['股份有限公司', '有限责任公司', '有限公司', '股份公司', '株式会社', '有限会社',
       '주식회사', '유한책임회사', '유한회사', '합자회사', '합명회사', '농업회사법인', '영농조합법인',
       '(주)', '㈜', '（주）', '(유)', '(株)']
SUF_RE = re.compile('|'.join(re.escape(s) for s in SUF))
LAT_RE = re.compile(r'\b(CO|COMPANY|LTD|LIMITED|INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C|GMBH|AG|SA|SAS|SARL|BV|B\.V|NV|N\.V|AB|OY|OYJ|SPA|S\.P\.A|PLC|KK|KABUSHIKI|KAISHA|KG|SE|AS|A/S)\b\.?')
NONALNUM = re.compile(r'[^0-9A-Z\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7a3]')

ORG_CN = re.compile(r'公司|大学|学院|研究|院|所|厂|集团|中心|局|会|株式会社|LTD|INC|CORP', re.I)
ORG_EP = re.compile(r'\b(LTD|LIMITED|INC|CORP|GMBH|AG|SA|SAS|BV|NV|AB|OY|SPA|PLC|KK|KAISHA|CO|COMPANY|LLC)\b|UNIV|INSTITUT|INSTITUTE|CENTRE|CENTER|FOUND|UNIVERSIT|ECOLE|FRAUNHOFER|CNRS|COMMISSARIAT|S\.A|S\.P\.A')


def norm_key(s):
    s = SUF_RE.sub(' ', str(s).upper())
    s = LAT_RE.sub(' ', s)
    return NONALNUM.sub('', s)


def is_org(raw, cc):
    u = str(raw).upper()
    return bool(ORG_CN.search(u)) if cc == 'CN' else bool(ORG_EP.search(u))


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


# ---- 공개 배정(lib12와 같은 식) ----
import numpy as np, pandas as pd
TXTFREE = (MC + r'\T1_산업대기실\TXT_R9\cache\kr_txtfree.parquet').replace(chr(92), '/')


def parse_ipc(series):
    ip = series.fillna('').str.upper().str.replace(r'\s+', ' ', regex=True).str.strip()
    ex = ip.str.extract(r'^([A-H]\d\d[A-Z])\s?0*(\d+)(?:/(\d+))?', expand=True)
    sc = ex[0]
    mg = (ex[0] + ' ' + ex[1]).where(ex[0].notna() & ex[1].notna())
    sg = (mg + '/' + ex[2]).where(mg.notna() & ex[2].notna())
    return sc, mg, sg


def build_pub_tables(kr, t0):
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
    if rule == 'H-sc-1':
        t = tabs['sc']
        return df.sc.map(t.kt.where(t.tot >= 1))
    res = pd.Series(np.nan, index=df.index, dtype=object)
    for lev in ['sg', 'mg', 'sc']:
        t = tabs[lev]
        m = df[lev].map(t.kt.where(t.tot >= 3))
        res = res.where(res.notna(), m)
    return res
