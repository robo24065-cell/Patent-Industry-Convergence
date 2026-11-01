# C07-T4 공용. 정규화는 C06-T4 lib6.norm(C04 libA)과 같은 규칙을 복사. 유형·그룹명은 T5 s20/s51 복사(읽기 전용 원천).
import re
import numpy as np, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
DB = B + r'\수집파이프라인\registry_full.sqlite'
KRP = B + r'\공모전데이터\작업데이터\kr_patents.parquet'
CUBE = B + r'\공모전데이터\작업데이터\country_year_cube.parquet'
MC = B + r'\분석\모델경쟁_20260927'
W = B + r'\분석\검증루프_20261012\C07\C07-T4'
C = W + r'\cache'
EVENT_CUT = '20250101'          # 이 날짜 이후 사건은 추출하지 않는다(잠금 레이블 보호)
DEV = list(range(2015, 2023))   # 개발 원점 t0 = Y-01-01, 창 = Y년
EVAL = list(range(2017, 2023))
LOCK = 2025
DEV_LABEL_MAX = '20221231'
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
GROUPS = ['삼성', '현대', '에스케이', 'SK', '엘지', 'LG', '롯데', '포스코', '지에스', 'GS', '한화', '케이티', 'KT', '한진', '두산', '씨제이', 'CJ',
          '엘에스', 'LS', '금호', '대림', '부영', '동부', '신세계', '효성', '대우', '영풍', '에쓰오일', '오씨아이', 'OCI', '코오롱', '케이씨씨', 'KCC',
          '한국타이어', '대성', '동국', '세아', '태광', '미래에셋', '한라', '이랜드', '한솔', '동양']
STATUS = ['설정등록', '국유특허(실용신안)설정등록', '국유설정등록', '소멸등록', '회복등록', '권리의 말소등록', '말소등록', '직권말소등록']
DEAD = ['소멸등록', '권리의 말소등록', '말소등록', '직권말소등록']


def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


def group_of(raw):
    s = str(raw or '').replace('주식회사', '').replace('(주)', '').replace('㈜', '').strip()
    for g in GROUPS:
        if s.startswith(g):
            return g
    return None


def name_type(name, natl):
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


def ts(y):
    return f'{y}0101'


def auc(y, s, w=None):
    """가중 AUC(동점 평균 순위). w = 행 가중치(부트스트랩 다중도)."""
    y = np.asarray(y, dtype=np.float64); s = np.asarray(s, dtype=np.float64)
    w = np.ones_like(y) if w is None else np.asarray(w, dtype=np.float64)
    o = np.argsort(s, kind='mergesort'); s, y, w = s[o], y[o], w[o]
    wp = w * y; wn = w * (1 - y)
    # 동점 묶음
    idx = np.r_[0, np.flatnonzero(np.diff(s)) + 1, len(s)]
    cp = np.add.reduceat(wp, idx[:-1]); cn = np.add.reduceat(wn, idx[:-1])
    cum_n = np.cumsum(cn) - cn
    num = (cp * (cum_n + 0.5 * cn)).sum()
    P, N = wp.sum(), wn.sum()
    return num / (P * N) if P > 0 and N > 0 else np.nan


def smin(df, cols):
    """YYYYMMDD 문자열 열들의 최소(결측·형식 불량 무시). 모두 없으면 None."""
    M = np.full(len(df), '99999999', dtype=object)
    for c in cols:
        v = df[c].fillna('').astype(str).str.strip().values
        ok = np.array([len(x) == 8 and x.isdigit() for x in v])
        M = np.where(ok & (v < M), v, M)
    return pd.Series(np.where(M == '99999999', None, M), index=df.index)
