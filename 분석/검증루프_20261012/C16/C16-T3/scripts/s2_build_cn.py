# C16-T3 s2: 제공 메타 CN 발명 문서표(출원 단위) + 출원인 분류 → cache/cn_docs.parquet
# 사용: python -I s2_build_cn.py <meta_glob> <cache_dir>
# 조직 정규식·키 정규화는 C13-T4 lib13.py(ORG_CN, norm_key)와 같은 식(복사). 외국 판정은 PREREG §1(c).
import sys, os, re, json, time
import duckdb, pandas as pd, numpy as np
M, C = sys.argv[1].replace('\\', '/'), sys.argv[2]
sys.stdout.reconfigure(encoding='utf-8')
t = time.time()
SUF = ['股份有限公司', '有限责任公司', '有限公司', '股份公司', '株式会社', '有限会社',
       '주식회사', '유한책임회사', '유한회사', '합자회사', '합명회사', '농업회사법인', '영농조합법인',
       '(주)', '㈜', '（주）', '(유)', '(株)']
SUF_RE = re.compile('|'.join(re.escape(s) for s in SUF))
LAT_RE = re.compile(r'\b(CO|COMPANY|LTD|LIMITED|INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C|GMBH|AG|SA|SAS|SARL|BV|B\.V|NV|N\.V|AB|OY|OYJ|SPA|S\.P\.A|PLC|KK|KABUSHIKI|KAISHA|KG|SE|AS|A/S)\b\.?')
NONALNUM = re.compile(r'[^0-9A-Z一-鿿぀-ヿ가-힣]')
ORG_CN = re.compile(r'公司|大学|学院|研究|院|所|厂|集团|中心|局|会|株式会社|LTD|INC|CORP', re.I)
# 외국 출원인 표지(출원인1): 일본·한국 법인어, 가나·한글, 라틴 문자만으로 된 이름
FOR_RE = re.compile(r'株式会社|有限会社|株式會社|合同会社|[぀-ヿ가-힣]')
LATIN_ONLY = re.compile(r'^[\x00-\x7fÀ-ɏ\s]+$')
HMT = {'TW', 'HK', 'MO'}   # CNIPA는 港澳台를 국내 지역 표에 포함


def norm_key(s):
    s = SUF_RE.sub(' ', str(s).upper()); s = LAT_RE.sub(' ', s)
    return NONALNUM.sub('', s)


con = duckdb.connect(); con.execute("set threads=2; set memory_limit='3GB'; set preserve_insertion_order=false")
d = con.execute(f"""
  select appl_no, substr(appl_no,5,1) typ, min(appl_year)::int ay, min(case when length(pub_date)=8 then cast(pub_date as integer) end) pub,
         any_value(ksic) ksic, any_value(prio_cc) prio_cc, any_value(applicant) applicant, count(*) nrow
  from read_parquet('{M}')
  where cc='CN' and length(appl_no)=14 and substr(appl_no,5,1) in ('1','8')
    and (appl_year >= 2010 or pub_date >= '20100101')
  group by appl_no""").fetchdf()
print('rows', len(d), 'dup rows(appl_no x ksic>1)', int((d.nrow > 1).sum()), round(time.time() - t), flush=True)
d['ksic'] = d.ksic.astype(str)
names = d.applicant.fillna('').str.split(r'[|;]')
d['app1'] = names.str[0].str.strip()
# 고유 이름 분류
ex = pd.DataFrame({'i': np.repeat(np.arange(len(d)), names.str.len().values), 'nm': np.concatenate(names.values)})
ex['nm'] = ex.nm.str.strip()
ex = ex[(ex.nm != '') & (~ex.nm.str.lower().isin(['nan', 'none', '-', 'null']))]
u = pd.DataFrame({'nm': pd.unique(pd.concat([ex.nm, d.app1[d.app1 != '']]))})
u['key'] = u.nm.map(norm_key)
up = u.nm.str.upper()
u['org'] = up.map(lambda x: bool(ORG_CN.search(x)))
u['orgk'] = u.org & (u.key.str.len() >= 2)
u['for_mark'] = u.nm.map(lambda x: bool(FOR_RE.search(x)) or bool(LATIN_ONLY.match(x)))
print('uniq names', len(u), round(time.time() - t), flush=True)
ex = ex.merge(u[['nm', 'key', 'orgk']], on='nm', how='left')
og = ex[ex.orgk].groupby('i').key.nunique()
d['n_orgkey'] = 0; d.loc[og.index, 'n_orgkey'] = og.values
d['n_names'] = ex.groupby('i').size().reindex(range(len(d))).fillna(0).astype(int).values
um = u.set_index('nm')
d['app1_miss'] = d.app1 == ''
a1 = d.app1.where(~d.app1_miss)
d['app1_org'] = a1.map(um.orgk).fillna(False).astype(bool)
d['app1_formark'] = a1.map(um.for_mark).fillna(False).astype(bool)
pc = d.prio_cc.astype(str).fillna('')


def prio_foreign(s):
    if s in ('', 'None', 'nan'): return False
    return any(c not in ('CN',) and c not in HMT and c != '' for c in s.split('|'))


d['prio_for'] = pc.map(prio_foreign)
d['foreign'] = (d.typ == '8') | d.prio_for | d.app1_formark
d['py'] = (d.pub // 10000).astype('Int64')
d = d.drop(columns=['applicant'])
d.to_parquet(os.path.join(C, 'cn_docs.parquet'), index=False)
u.to_parquet(os.path.join(C, 'cn_names.parquet'), index=False)
log = dict(rows=len(d), typ=d.typ.value_counts().to_dict(), pub_na=float(d.pub.isna().mean()), uniq_names=len(u),
           org_name_share=float(u.orgk.mean()), foreign_share=float(d.foreign.mean()), secs=round(time.time() - t))
json.dump(log, open(os.path.join(C, 's2_log.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(log)
