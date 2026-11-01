"""C16-T3 공용 경로(C09-T3 common.py 방식). 입력은 읽기 전용, 쓰기는 C16-T3 폴더만."""
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
WD = os.path.dirname(HERE)                       # C16-T3
CACHE = os.path.join(WD, 'cache'); os.makedirs(CACHE, exist_ok=True)
ROOT = os.path.abspath(os.path.join(WD, '..', '..', '..'))   # 분석
T3 = os.path.join(ROOT, '모델경쟁_20260927', 'T3_기술셀부상예측')
R3 = os.path.join(T3, 'R3')
CP_DEV = os.path.join(R3, 'cache_pub')
CP_FIN = os.path.join(R3, 'cache_pub_final')
C09 = os.path.join(ROOT, '검증루프_20261012', 'C09', 'C09-T3')

def ddb(path):
    return path.replace('\\', '/')

# CN 조직 판정: C13-T4 lib13.ORG_CN + norm_key 길이 ≥2 (복사)
SUF = ['股份有限公司', '有限责任公司', '有限公司', '股份公司', '株式会社', '有限会社',
       '주식회사', '유한책임회사', '유한회사', '합자회사', '합명회사', '농업회사법인', '영농조합법인',
       '(주)', '㈜', '（주）', '(유)', '(株)']
SUF_RE = re.compile('|'.join(re.escape(s) for s in SUF))
LAT_RE = re.compile(r'\b(CO|COMPANY|LTD|LIMITED|INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C|GMBH|AG|SA|SAS|SARL|BV|B\.V|NV|N\.V|AB|OY|OYJ|SPA|S\.P\.A|PLC|KK|KABUSHIKI|KAISHA|KG|SE|AS|A/S)\b\.?')
NONALNUM = re.compile(r'[^0-9A-Z一-鿿぀-ヿ가-힣]')
ORG_CN = re.compile(r'公司|大学|学院|研究|院|所|厂|集团|中心|局|会|株式会社|LTD|INC|CORP', re.I)

def norm_key(s):
    s = SUF_RE.sub(' ', str(s).upper()); s = LAT_RE.sub(' ', s)
    return NONALNUM.sub('', s)

def is_org_cn(nm):
    return bool(ORG_CN.search(str(nm).upper())) and len(norm_key(nm)) >= 2
