# verify_3: form=2 행이 파생 패널에 들어가는지(행 수만, n_emp 미열람)
import sys, json, re, os
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
BASE = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
EXT = BASE + r'\분석\모델경쟁_20260927\_shared\ext'
OUT = BASE + r'\분석\검증루프_20261012\C04\C04-T4\verify_3'
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper(); return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
d50 = json.load(open(BASE + r'\공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json', encoding='utf-8'))
nn = lambda s: re.sub(r"[\s·,및]", "", str(s)); kmap = {nn(x['codenmKr']): x['code'] for x in d50}
link = pd.read_parquet(EXT + r'\firm_nps_applicant_link.parquet')
hk = set((link.loc[link.nps_hit, 'biz6'] + '|' + link.loc[link.nps_hit, 'nname']))
kp = pd.read_parquet(EXT + r'\firm_nps_ksic50_panel.parquet'); print('ksic50_panel cols', list(kp.columns))
out = []
for s in ['202412', '202512', '202608']:
    d = pd.read_parquet(EXT + rf'\firm_nps_wkpl_{s}.parquet', columns=['name', 'biz6', 'stat', 'form', 'indnm'])
    d['k50'] = d.indnm.map(lambda x: kmap.get(nn(x)))
    d['hit'] = (d.biz6 + '|' + d.name.map(norm)).isin(hk)
    for f in ['1', '2']:
        x = d[d.form.astype(str) == f]
        out.append(dict(snap=s, form=f, rows=len(x), active=int((x.stat == '1').sum()), k50_direct_rows=int(x.k50.notna().sum()),
                        k50_direct_active=int((x.k50.notna() & (x.stat == '1')).sum()), patlink_rows=int(x.hit.sum()),
                        patlink_active=int((x.hit & (x.stat == '1')).sum())))
        print(out[-1], flush=True)
pd.DataFrame(out).to_csv(OUT + r'\v3_affects.csv', index=False, encoding='utf-8-sig')
