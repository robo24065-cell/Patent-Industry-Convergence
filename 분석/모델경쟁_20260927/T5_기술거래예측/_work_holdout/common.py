import os, re, pandas as pd, numpy as np
# 3단계 홀드아웃 평가자만 T5_CUT·T5_LMAX 환경변수를 바꾼다(별도 복사본 폴더에서). 1·2단계 기본값: 2019-01-01 / 2015
HOLD_CUT = pd.Timestamp(os.environ.get('T5_CUT', '2019-01-01'))   # 홀드아웃 보호: 이 날짜 이후 사건은 캐시에 없음(추출 단계에서 제외)
LMAX = int(os.environ.get('T5_LMAX', 2015))
TRAIN_MAX_L = 2011   # 검증(L=2014,2015)용 학습: L+3y <= 2014-01-01 인 기준일만
VAL_L = [2014, 2015]


def classify(rs):
    """권리이전 사유 -> 클래스. 주 정의 MAIN='양도'류."""
    if rs is None or (isinstance(rs, float) and np.isnan(rs)):
        return None
    s = str(rs)
    if '상속' in s:
        return 'INH'
    if '합병' in s or '분할' in s:
        return 'CORP'
    if '전담조직' in s:
        return 'OTHER'
    if any(k in s for k in ['양도', '증여', '매각', '경매', '유질', '이전']):
        return 'MAIN'
    return 'OTHER'


_rm = re.compile(r'\s+|주식회사|\(주\)|㈜|유한회사|\(유\)|,|\.|co|ltd|inc|corporation|corp', re.I)


def norm(s):
    if s is None or isinstance(s, float):
        return ''
    return _rm.sub('', str(s)).lower()


def name_type(name, natl):
    """권리자 이름 -> 유형. 1 국내법인 2 대학 3 연구기관 4 국내개인 5 외국 7 국가·공공"""
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


def load_panel(years, cols=None):
    return pd.concat([pd.read_parquet(f'cache/panel_L{y}.parquet', columns=cols) for y in years], ignore_index=True)
