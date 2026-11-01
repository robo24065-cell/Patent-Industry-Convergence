# v6b: 위험집합 키 중 기준일 이전 보인 연결 문서 보유 여부를 '기업키(b6|상호)' 단위로 재확인.
import os, re, json, sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
WD = os.path.join(B, '분석', '검증루프_20261012', 'C05', 'C05-T4')
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
def norm(s):
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)
k = pd.read_parquet(os.path.join(B, '공모전데이터', '작업데이터', 'kr_patents.parquet'), columns=['공개일자', '등록일자', '출원인명'])
k.columns = ['pub', 'rg', 'aname']
for c in ['pub', 'rg']:
    k[c] = k[c].fillna('').astype(str).str.strip(); k.loc[k[c].str.len() != 8, c] = None
k['vis'] = np.where(k.rg.notna() & (k.pub.isna() | (k.rg.fillna('9') < k.pub.fillna('9'))), k.rg, k.pub)
k['nname'] = k.aname.map(norm)
rows = []
with open(os.path.join(B, '공모전데이터', '외부데이터', 'Corporate_20260720', 'CORP_APPLICANT.txt'), encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('¶')
        if len(p) >= 5: rows.append((p[0].strip(), p[1].strip(), re.sub(r'\D', '', p[4])))
ca = pd.DataFrame(rows, columns=['cd', 'aname', 'biz']); ca['nname'] = ca.aname.map(norm); ca['hb'] = ca.biz.str.len() == 10
ca['b6'] = np.where(ca.hb, ca.biz.str[:6], None); ca['tc'] = ca.hb & ca.biz.str[3:5].isin([str(x) for x in range(81, 89)])
ca = ca[ca.nname.str.len() >= 2]
g = ca.groupby('nname').agg(n=('cd', 'size'), nb=('hb', 'sum'), nb6=('b6', 'nunique'), b6=('b6', 'first'), tc=('tc', 'max'))
ok = g[(g.nb == g.n) & (g.nb6 == 1) & g.tc].b6
k['b6'] = k.nname.map(ok); lk = k[k.b6.notna() & k.vis.notna()]
fvk = (lk.b6 + '|' + lk.nname).to_frame('key').assign(vis=lk.vis.values).groupby('key').vis.min()
fz = pd.read_parquet(os.path.join(WD, 'frozen_scores.parquet'), columns=['key'])
lab = pd.read_parquet(os.path.join(WD, 'cache', 'lock_labels.parquet'))
fz = fz.merge(lab, on='key')
pv = fz.key.map(fvk)
o = dict(riskset_keys_with_prior_linked_doc_by_key=int((pv <= '20241231').sum()),
         riskset_keys_nname_in_ok_but_b6_differs=int((fz.key.str.split('|', n=1).str[1].isin(ok.index) & (fz.key.str[:6] != fz.key.str.split('|', n=1).str[1].map(ok))).sum()),
         pos_keys_first_vis_min=str(pv[fz.y == 1].min()), pos_keys_first_vis_max=str(pv[fz.y == 1].max()))
# 동일 상호에 위험집합 키가 여러 개(같은 상호, 다른 b6)인 경우
nn = fz.key.str.split('|', n=1).str[1]
o['riskset_dup_nname_keys'] = int(nn.duplicated(keep=False).sum())
json.dump(o, open(os.path.join(WD, 'verify_0', 'v6b_riskset_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(o)
