# -*- coding: utf-8 -*-
"""ISO 표준(공개 Open Data) → ICS → KSIC 50 연결, KSIC×연도 표준 발행 건수 집계.
ICS↔KSIC 연결표는 공식 대조표가 없어 KSIC 품목 정의와 ICS 제목을 대조해 수작업으로 작성(아래 MAP, 근거=ICS 제목).
산출: trade_ics_ksic_map.csv, trade_iso_ksic_year.parquet"""
import pandas as pd, numpy as np
MAP = {
 '15220':['61.060'], '20202':['83.080'], '20411':['87.040','87.060'], '20423':['71.100.70'],
 '20491':['37.040.20','37.040.25','37.040.30','37.100.20'], '21220':['11.120.10','11.120.01'],
 '23992':['25.100.70'], '24191':['77.140.50','25.220.40'], '24219':['77.120'],
 '25914':['25.120.10','77.140.85'], '25922':['25.220.40','25.220.20'], '25924':['25.100','25.080'],
 '25932':['21.060','21.080','91.190'],
 '26111':['31.200','35.220'], '26112':['31.200'], '26121':['31.260'], '26129':['31.080','29.045'],
 '26211':['31.120'], '26212':['31.120'], '26223':['31.180'], '26224':['31.190','31.180'], '26291':['31.060'],
 '26322':['35.180','31.120'], '26323':['35.180','37.100.10'], '26422':['33.050','33.070'], '26429':['33.060'],
 '26519':['33.160.40','33.160.25','33.160.60'], '26529':['33.160.30','33.160.50','33.160.10','17.140'],
 '27199':['11.040','11.060','11.180'], '27211':['17.020','17.040','17.080','17.120','49.090','07.040','47.020.70'],
 '27212':['17.220','19.080'], '27301':['37.020','17.180'],
 '28111':['29.160'], '28122':['29.120','29.130'], '28511':['97.040','97.030'], '28902':['29.035','29.080'],
 '29131':['23.080'], '29132':['23.140','23.160'], '29133':['23.060'], '29150':['25.180','27.060'],
 '29162':['91.140.90','53.020'], '29210':['65.060'], '29229':['25.080','25.120','25.040.10','25.040.20'],
 '29241':['53.100','73.100','91.220'], '29280':['25.040.30'], '30393':['43.040.60','43.040.80'],
 '31202':['45.060','45.040'], '31312':['49.020'], '33301':['97.220.30'], '33309':['97.220.40','97.220.20','97.220.99','97.220.01'],
}
ics = pd.read_csv('trade_iso_ics_clean.csv', dtype=str)
rows = []
for k, ps in MAP.items():
    for p in ps:
        sub = ics[(ics.ics == p) | ics.ics.str.startswith(p + '.')]
        t = ics.loc[ics.ics == p, 'title']
        rows.append(dict(ksic=k, ics_prefix=p, ics_title=t.iloc[0] if len(t) else '', n_ics_codes=len(sub)))
mp = pd.DataFrame(rows); mp.to_csv('trade_ics_ksic_map.csv', index=False, encoding='utf-8-sig')
print('ICS 접두어 중 ICS 목록에 없는 것:', mp[mp.ics_title == ''].ics_prefix.tolist())
d = pd.read_parquet('trade_iso_deliverables_metadata_raw.parquet',
                    columns=['id','deliverableType','supplementType','reference','publicationDate','edition','icsCode','ownerCommittee','currentStage'])
d = d[d.publicationDate.notna() & d.icsCode.notna()]
d = d[d.deliverableType.isin(['IS','TS','TR','PAS','ISP','IWA'])]
d['year'] = pd.to_datetime(d.publicationDate.astype(str).str[:10], errors='coerce').dt.year; d = d[d.year.notna()]; d['year'] = d.year.astype(int)
d['kind'] = np.where(d.supplementType.notna(), 'supp', np.where(d.edition.fillna(1) <= 1, 'new', 'rev'))
e = d[['id','year','kind','icsCode','ownerCommittee']].explode('icsCode')
out = []
for k, ps in MAP.items():
    m = np.zeros(len(e), bool)
    for p in ps:
        m |= (e.icsCode.values == p) | e.icsCode.str.startswith(p + '.').values
    s = e[m].drop_duplicates('id')
    g = s.groupby(['year','kind']).size().unstack(fill_value=0).reindex(columns=['new','rev','supp'], fill_value=0)
    g['ksic'] = k; out.append(g.reset_index())
iso = pd.concat(out).fillna(0)
iso['total'] = iso[['new','rev','supp']].sum(axis=1)
iso.to_parquet('trade_iso_ksic_year.parquet', index=False)
t = iso[(iso.year >= 2000) & (iso.year <= 2016)].groupby('ksic')[['new','rev','total']].sum()
print(t.to_string()); print('2000-16 발행 1건 이상 KSIC:', (t.total > 0).sum(), '/ 50; 연 평균 5건 이상:', (t.total >= 85).sum())
