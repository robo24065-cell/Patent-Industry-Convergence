# C04-T4 공용 적재·연결 함수. 연결은 as-of: 문서(기준일까지 보인 것) 출원인명 → CORP_APPLICANT(유일 정규화 상호) → biz6|상호 = S0 스냅샷 기업키.
import re
import numpy as np, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
MC = B + r'\분석\모델경쟁_20260927'
EXT = MC + r'\_shared\ext'
W = B + r'\분석\검증루프_20261012\C04\C04-T4'
C = W + r'\cache'
DEV_A = ['201512', '201612', '201712', '201812', '201912', '202012', '202112']
S1_OF = {'201512': '201712', '201612': '201812', '201712': '201912', '201812': '202012', '201912': '202112',
         '202012': '202212', '202112': '202312'}
NONOVER_A = ['201512', '201712', '201912', '202112']
DEV_B = ['201512', '201612', '201712', '201812', '201912', '202012', '202112', '202212', '202312']
LOCK_S0 = ['202312', '202412']
REF = {'201512': '20151130', '201612': '20161130', '201712': '20171130', '201812': '20181130', '201912': '20191130',
       '202012': '20201231', '202112': '20211231', '202212': '20221231', '202312': '20231231', '202412': '20241231'}
LOCKED_S1 = {'202512', '202608'}
CORP_TC = {str(x) for x in range(81, 89)}
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


def add_months(ymd, m):
    y, mo, d = int(ymd[:4]), int(ymd[4:6]), int(ymd[6:])
    mo += m
    y += (mo - 1) // 12; mo = (mo - 1) % 12 + 1
    import calendar
    d = min(d, calendar.monthrange(y, mo)[1])
    return f'{y:04d}{mo:02d}{d:02d}'


def load_ca():
    rows = []
    with open(B + r'\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt', encoding='utf-8') as f:
        next(f)
        for line in f:
            p = line.rstrip('\r\n').split('¶')
            if len(p) >= 5:
                rows.append((p[0].strip(), p[1].strip(), p[3].strip(), p[4].strip()))
    ca = pd.DataFrame(rows, columns=['applicantCd', 'aname', 'corpno', 'bizno'])
    ca['bizno'] = ca.bizno.str.replace(r'\D', '', regex=True)
    ca['nname'] = ca.aname.map(norm)
    ca['has_biz'] = ca.bizno.str.len() == 10
    ca['biz6'] = np.where(ca.has_biz, ca.bizno.str[:6], None)
    ca['corp_tc'] = ca.has_biz & ca.bizno.str[3:5].isin(CORP_TC)
    return ca


def name_map(ca):
    """정규화 상호 → biz6 (그 상호의 CA 행이 모두 사업자번호를 갖고 biz6가 하나일 때만). 상호 길이 ≥ 2."""
    g = ca[ca.nname.str.len() >= 2].groupby('nname')
    st = g.agg(n=('applicantCd', 'size'), nb=('has_biz', 'sum'), nb6=('biz6', 'nunique'), b6=('biz6', 'first'), ctc=('corp_tc', 'max'))
    ok = st[(st.nb == st.n) & (st.nb6 == 1) & st.ctc]
    return ok.b6.to_dict(), st


def load_docs():
    k = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet',
                        columns=['출원번호', '출원연도', '공개일자', '등록일자', '출원인명', '세세분류코드', '우선권국가'])
    k.columns = ['appl_no', 'ay', 'pub', 'rg', 'aname', 'ksic', 'prio']
    k['pub'] = k.pub.fillna('').astype(str).str.strip(); k['rg'] = k.rg.fillna('').astype(str).str.strip()
    k.loc[k.pub.str.len() != 8, 'pub'] = None; k.loc[k.rg.str.len() != 8, 'rg'] = None
    k['vis'] = k.pub
    m = k.rg.notna() & (k.vis.isna() | (k.rg.fillna('99999999') < k.vis.fillna('99999999')))
    k.loc[m, 'vis'] = k.loc[m, 'rg']
    k['nname'] = k.aname.map(norm)
    k['ksic'] = k.ksic.astype(str)
    p = k.prio.fillna('')
    k['dom_prio'] = (p == '') | p.str.split('|').map(lambda L: all(x == 'KR' for x in L if x))
    return k[['appl_no', 'ay', 'pub', 'rg', 'vis', 'nname', 'ksic', 'dom_prio']]


def load_nps(s, cols=('name', 'biz6', 'stat', 'ldong', 'indcd', 'adpt_dt', 'n_emp', 'amt')):
    assert s not in LOCKED_S1, 'locked S1 snapshot must not be loaded here'
    d = pd.read_parquet(EXT + rf'\firm_nps_wkpl_{s}.parquet', columns=list(cols))
    d = d[(d.stat.astype(str) == '1') & d.biz6.astype(str).str[3:5].isin(CORP_TC)].copy()
    d['key'] = d.biz6 + '|' + d.name.map(norm)
    d['ba'] = d.biz6 + '|' + d.adpt_dt.fillna('').astype(str)
    return d.reset_index(drop=True)


def firm_table(d):
    d = d.sort_values('n_emp', ascending=False)
    g = d.groupby('key')
    f = g.agg(E=('n_emp', 'sum'), amt=('amt', 'sum'), n_wkpl=('ba', 'size'), indcd=('indcd', 'first'), ldong=('ldong', 'first'))
    f['sido'] = f.ldong.astype(str).str[:2]
    ic = f.indcd.fillna('').astype(str).str.strip()
    ic = ic.where(~ic.isin(['', '000000']), None)
    f['ind2'] = ic.map(lambda x: x.zfill(6)[:2] if isinstance(x, str) and x.isdigit() else None)
    return f


def uniq_ba(d):
    c = d.ba.value_counts()
    return set(c[c == 1].index)


def outcome_E1(units_keys, d0, d1):
    """units_keys: S0 기업키 집합. d0: S0 활성 행, d1: S1 활성 행.
    S1 행 배정: 기업키 일치 우선. 아니면 ba(biz6|적용일)가 S0 파일 전체와 S1 파일 전체에서 각각 유일할 때만 같은 사업장으로 본다."""
    K = set(units_keys)
    u0, u1 = uniq_ba(d0), uniq_ba(d1)
    ba2key = d0[d0.key.isin(K) & d0.ba.isin(u0)].set_index('ba').key
    d1 = d1.copy()
    d1['own'] = np.where(d1.key.isin(K), d1.key, None)
    m = d1.own.isna() & d1.ba.isin(u1)
    d1.loc[m, 'own'] = d1.loc[m, 'ba'].map(ba2key)
    d1 = d1[d1.own.notna()]
    e = d1.groupby('own').agg(E1=('n_emp', 'sum'), amt1=('amt', 'sum'))
    by_ba = d1[~d1.key.isin(K)].groupby('own').size()
    e['n_rows_by_ba'] = by_ba.reindex(e.index).fillna(0).astype(int)
    return e
