# C06-T4 공용 함수. C04 libA의 정규화·CA·문서 적재 규칙을 그대로 복사(읽기 전용 원천).
import re, calendar
import numpy as np, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
MC = B + r'\분석\모델경쟁_20260927'
EXT = MC + r'\_shared\ext'
W = B + r'\분석\검증루프_20261012\C06\C06-T4'
C = W + r'\cache'
CORP_TC = {str(x) for x in range(81, 89)}
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)

# C06-A 원점(6개월 창). 값 = (t0, 창 끝)
DEV_A = [('2023-04-30', '2023-10-31'), ('2023-10-31', '2024-04-30'), ('2024-04-30', '2024-10-31'),
         ('2024-10-31', '2025-04-30'), ('2025-04-30', '2025-10-31')]
LOCK_A = ('2025-10-31', '2026-04-30')
LABEL_MAX_A = '2025-10-31'


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


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
    g = ca[ca.nname.str.len() >= 2].groupby('nname')
    st = g.agg(n=('applicantCd', 'size'), nb=('has_biz', 'sum'), nb6=('biz6', 'nunique'), b6=('biz6', 'first'), ctc=('corp_tc', 'max'))
    ok = st[(st.nb == st.n) & (st.nb6 == 1) & st.ctc]
    return ok.b6.to_dict(), st


def load_docs():
    k = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet',
                        columns=['출원번호', '출원연도', '공개일자', '등록일자', '출원인명', '세세분류코드', '우선권국가', '메인IPC', '출원인주소'])
    k.columns = ['appl_no', 'ay', 'pub', 'rg', 'aname', 'ksic', 'prio', 'mipc', 'addr']
    k['sc'] = k.mipc.fillna('').astype(str).str.replace(' ', '').str[:4]
    k['pub'] = k.pub.fillna('').astype(str).str.strip(); k['rg'] = k.rg.fillna('').astype(str).str.strip()
    k.loc[k.pub.str.len() != 8, 'pub'] = None; k.loc[k.rg.str.len() != 8, 'rg'] = None
    k['vis'] = k.pub
    m = k.rg.notna() & (k.vis.isna() | (k.rg.fillna('99999999') < k.vis.fillna('99999999')))
    k.loc[m, 'vis'] = k.loc[m, 'rg']
    k['nname'] = k.aname.map(norm)
    k['ksic'] = k.ksic.astype(str)
    p = k.prio.fillna('')
    k['dom_prio'] = (p == '') | p.str.split('|').map(lambda L: all(x == 'KR' for x in L if x))
    return k[['appl_no', 'ay', 'pub', 'rg', 'vis', 'nname', 'ksic', 'dom_prio', 'sc', 'addr']]


def ymd(s):
    return s.replace('-', '')


SIDO = {'11': '서울', '26': '부산', '27': '대구', '28': '인천', '29': '광주', '30': '대전', '31': '울산', '36': '세종',
        '41': '경기', '42': '강원', '51': '강원', '43': '충북', '44': '충남', '45': '전북', '52': '전북', '46': '전남',
        '47': '경북', '48': '경남', '50': '제주'}
