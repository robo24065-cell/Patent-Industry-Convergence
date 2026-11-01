# -*- coding: utf-8 -*-
"""ETSI SEP 선언 특허 ↔ 제공 데이터(5개국) 로컬 매칭. 제공 데이터는 외부로 보내지 않음(공개 ETSI 파일을 내려받아 로컬 결합).
키: KR 출원번호('10'+연도+일련), US 등록번호·출원 일련(신형식), EP 출원(yy+6)·공개번호, JP 등록번호, CN 출원번호(체크디지트 제외)·공개번호."""
import re, pandas as pd, duckdb
EXT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext'
M = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\meta\*.parquet'
e = pd.read_parquet(EXT + r'\trade_etsi_sep_patents.parquet')
e = e[e.cc.isin(['KR','US','EP','JP','CN'])].copy()
e['app'] = e.PATT_APPLICATION_NUMBER.fillna('').str.replace(r'[^0-9A-Z]', '', regex=True)
def pubs(s):
    return [re.sub(r'\s+[A-Z]\d?$', '', p.strip()) for p in str(s).split('|') if p.strip() and p.strip() != '-' and p.strip() != 'None']
keys = []
for r in e.itertuples():
    cc, a = r.cc, r.app[2:] if r.app.startswith(r.cc) else r.app
    ks = []
    if cc == 'KR' and re.fullmatch(r'\d{11}', a): ks.append(('A', '10' + a))
    if cc == 'US' and re.fullmatch(r'\d{12}', a): ks.append(('A', a[4:]))
    if cc == 'EP' and re.fullmatch(r'\d{11}', a): ks.append(('A', a[2:4] + a[-6:]))
    if cc == 'CN' and re.fullmatch(r'\d{12}', a): ks.append(('A', a))
    for p in pubs(r.PUBL_NUMBER):
        if not p.startswith(cc): continue
        n = p[2:]
        if cc == 'KR' and re.fullmatch(r'10\d{7}', n): ks.append(('R', n + '0000'))
        if cc == 'US' and re.fullmatch(r'\d{7,8}', n): ks.append(('R', n))
        if cc == 'EP' and re.fullmatch(r'\d{7}', n): ks.append(('R', n.zfill(8)))
        if cc == 'JP' and re.fullmatch(r'\d{7}', n): ks.append(('R', n.zfill(8)))
        if cc == 'CN' and re.fullmatch(r'\d{9}', n): ks.append(('R', n))
    for t, k in ks: keys.append((r.Index, cc, t, k))
K = pd.DataFrame(keys, columns=['eid','cc','ktype','key'])
con = duckdb.connect(); con.sql("SET memory_limit='3GB'; SET threads=2")
con.register('K', K)
con.sql(f"""create temp table meta as select cc, appl_no, rgst_no, ksic, appl_year, main_ipc,
   case when cc='EP' then substr(split_part(appl_no,'.',1),1,8) when cc='CN' then split_part(appl_no,'.',1) else appl_no end as akey
   from read_parquet('{M}') where cc in ('KR','US','EP','JP','CN')""")
m = con.sql("""select K.eid, meta.cc, meta.appl_no, meta.ksic, meta.appl_year, meta.main_ipc from K join meta on K.cc=meta.cc and K.key=meta.akey where K.ktype='A'
 union select K.eid, meta.cc, meta.appl_no, meta.ksic, meta.appl_year, meta.main_ipc from K join meta on K.cc=meta.cc and K.key=meta.rgst_no where K.ktype='R'""").df()
m = m.merge(e[['first_decl','first_refl','company','g2','g3','g4','g5','n_std']], left_on='eid', right_index=True)
m.to_parquet(EXT + r'\trade_etsi_sep_link.parquet', index=False)
tot = e.groupby('cc').size(); hit = m.drop_duplicates('eid').groupby('cc').size()
keyed = K.drop_duplicates('eid').groupby('cc').size()
print(pd.DataFrame({'etsi_patents': tot, 'with_key': keyed, 'matched_to_provided': hit}).fillna(0).astype(int).to_string())
pk = m.drop_duplicates(['cc','appl_no','ksic']).groupby('ksic').size().sort_values(ascending=False)
print('KSIC별 SEP 선언 매칭 (cc,appl_no) 수'); print(pk.to_string()); print('SEP≥1 KSIC 수', (pk > 0).sum(), '≥100:', (pk >= 100).sum())
