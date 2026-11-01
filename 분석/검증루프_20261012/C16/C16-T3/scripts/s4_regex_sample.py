# C16-T3 s4: 조직 정규식 감사 표본(공개연도 2015·2019·2024 각 300건, seed 20261017) + 독립 규칙 판정
# 사용: python -I s4_regex_sample.py <meta_glob> <cache_dir>
# 산출: cache/regex_sample.csv (appl_no, py, applicant 원문, C13 판정(문서·출원인1), 독립 규칙 판정, 불일치 표시)
import sys, os, re
import duckdb, pandas as pd, numpy as np
M, C = sys.argv[1].replace('\\', '/'), sys.argv[2]
sys.stdout.reconfigure(encoding='utf-8')
d = pd.read_parquet(os.path.join(C, 'cn_docs.parquet'), columns=['appl_no', 'py', 'pub', 'n_orgkey', 'app1_org', 'foreign', 'typ'])
d = d[d.pub.notna() & d.py.isin([2015, 2019, 2024])]
smp = pd.concat([g.sample(300, random_state=20261017) for _, g in d.groupby('py')])
con = duckdb.connect(); con.execute("set threads=2; set memory_limit='3GB'")
con.register('s', smp[['appl_no']])
ap = con.execute(f"select appl_no, any_value(applicant) applicant from read_parquet('{M}') m where cc='CN' and appl_no in (select appl_no from s) group by appl_no").fetchdf()
smp = smp.merge(ap, on='appl_no', how='left')
# 독립 규칙(정규식과 다른 어휘·구조 기준): 이름 단위로 '개인' 여부를 먼저 판정
ORG_LEX = ['公司', '大学', '学院', '学校', '中学', '小学', '研究', '院', '所', '厂', '集团', '中心', '局', '协会', '学会', '委员会', '基金会', '商会',
           '站', '部', '社', '处', '室', '队', '馆', '场', '店', '行', '基地', '银行', '事务所', '工作室', '合作社', '医院', '联合', '会社', '企业',
           '机构', '政府', '军', '台', '组', '坊', '园', '庄', '科技', '技术', '电子', '工业', '实业', '有限', '股份', '株式', '供电', '电网',
           '电力', '制造', '发展', '控股', '国际', '设备', '材料', '化工', '机械', '电器', '电气', '光电', '信息', '能源', '环保', '生物', '医药', '集团']
LAT_ORG = re.compile(r'\b(INC|CORP|CORPORATION|CO|COMPANY|LTD|LIMITED|LLC|GMBH|AG|SA|NV|BV|KK|AB|OY|SPA|PLC|SE|UNIV|UNIVERSITY|INSTITUTE|FOUNDATION)\b', re.I)
HAN = re.compile(r'^[一-鿿·・]+$')


def person(nm):
    nm = nm.strip()
    if not nm: return None
    if HAN.match(nm):
        core = nm.replace('·', '').replace('・', '')
        if any(w in nm for w in ORG_LEX if len(w) >= 2): return False
        if 2 <= len(core) <= 4: return True        # 중국인 개인 이름 길이
        if '·' in nm or '・' in nm: return True     # 음역 외국인 개인 이름(예: 约翰·史密斯)
        return False                              # 5자 이상 한자 = 기관명으로 봄
    if LAT_ORG.search(nm): return False
    if re.match(r'^[A-Za-z\.\-\' ]+$', nm) and 1 <= len(nm.split()) <= 4: return True
    return None                                   # 판단 보류(수작업)


def doc_rule(s):
    nms = [x.strip() for x in re.split(r'[|;]', s or '') if x.strip()]
    if not nms: return 'nonorg', 'nonorg', ''
    ps = [person(x) for x in nms]
    unk = any(p is None for p in ps)
    anyorg = any(p is False for p in ps)
    a1 = 'org' if ps[0] is False else ('nonorg' if ps[0] is True else 'unk')
    return ('org' if anyorg else ('unk' if unk else 'nonorg')), a1, '|'.join('P' if p else ('O' if p is False else '?') for p in ps)


r = smp.applicant.map(doc_rule)
smp['rule_doc'] = [x[0] for x in r]; smp['rule_app1'] = [x[1] for x in r]; smp['rule_pattern'] = [x[2] for x in r]
smp['c13_doc'] = np.where(smp.n_orgkey >= 1, 'org', 'nonorg')
smp['c13_app1'] = np.where(smp.app1_org, 'org', 'nonorg')
smp['disagree_doc'] = smp.rule_doc != smp.c13_doc
smp['disagree_app1'] = smp.rule_app1 != smp.c13_app1
smp.to_csv(os.path.join(C, 'regex_sample.csv'), index=False, encoding='utf-8-sig')
print(smp.groupby('py')[['disagree_doc', 'disagree_app1']].sum())
for _, x in smp[smp.disagree_doc | smp.disagree_app1].iterrows():
    print(x.py, x.appl_no, x.c13_doc, x.rule_doc, x.c13_app1, x.rule_app1, x.rule_pattern, '|', x.applicant)
