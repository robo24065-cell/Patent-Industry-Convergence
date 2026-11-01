# C08-T4 공용
import re, os
import numpy as np, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
DB = B + r'\수집파이프라인\registry_full.sqlite'
W = B + r'\분석\검증루프_20261012\C08\C08-T4'
C = W + r'\cache'
CUT = '20250101'                     # 이 날짜 이후 사건은 추출하지 않는다
DEV = list(range(2015, 2023))
DEAD = ['소멸등록', '권리의 말소등록', '말소등록', '직권말소등록']
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


def name_type(name, natl=None):
    """T5 common.name_type 복사. 1 국내법인 2 대학 3 연구기관 4 국내개인 5 외국 7 국가·공공"""
    s = str(name) if name is not None else ''
    if isinstance(natl, str) and natl not in ('대한민국', ''):
        return 5
    if '대학' in s or '산학협력' in s:
        return 2
    if any(k in s for k in ['연구원', '연구소', '과학기술원', '연구재단']):
        return 3
    if any(k in s for k in ['대한민국', '광역시', '특별시', '도청', '시청', '군청', '구청', '공사', '공단', '진흥원', '재단']):
        return 7
    if any(k in s for k in ['주식회사', '(주)', '㈜', '회사', '유한', '코리아', '테크', '산업', '전자', '기업', '협동조합', '법인']):
        return 1
    if re.fullmatch(r'[가-힣]{2,4}', s.replace(' ', '')):
        return 4
    if re.search(r'[A-Za-z]', s):
        return 5
    return 1


def ok8(s):
    s = s.fillna('').astype(str).str.strip()
    return s.where(s.str.fullmatch(r'\d{8}'), None)


def alive_mask(pat, st, t0):
    """pat: rgst_no, rgstDate, exp(존속만료 예정). st: rgst_no, d, kind('dead'|'rec'). t0 'YYYYMMDD'."""
    base = (pat.rgstDate.notna()) & (pat.rgstDate < t0) & (pat.exp.isna() | (pat.exp >= t0))
    s = st[st.d < t0].sort_values(['rgst_no', 'd', 'kind'])     # 같은 날이면 dead < rec 순(문자열) → rec가 마지막
    last = s.drop_duplicates('rgst_no', keep='last').set_index('rgst_no').kind
    dead = pat.rgst_no.map(last).eq('dead').values
    return base.values & ~dead
