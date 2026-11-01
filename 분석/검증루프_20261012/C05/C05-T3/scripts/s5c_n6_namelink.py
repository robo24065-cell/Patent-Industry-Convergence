# -*- coding: utf-8 -*-
"""C05-T3 s5c: 자기 재구성에서만 위험집합 밖으로 빠진 183개 기업(이름 보조 연결) 연결 품질 점검"""
import pandas as pd, numpy as np, os, sys, io, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C05\C05-T3"); CA = os.path.join(OUT, "cache")
lab = pd.read_parquet(os.path.join(CA, 'n6_lab.parquet'))
lo = lab[lab.src == 'left_only'].bz
fl = pd.read_parquet(os.path.join(CA, 'n6_firm_link.parquet'))
fa = pd.read_parquet(os.path.join(CA, 'n6_firm_apps.parquet'))
rows = []
with open(os.path.join(ROOT, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8", errors="replace") as fh:
    next(fh)
    for line in fh:
        f = line.rstrip(chr(13) + chr(10)).split("¶")
        if len(f) >= 5: rows.append([x.strip() for x in f[:5]])
ca = pd.DataFrame(rows, columns=['cd', 'name', 'ename', 'crn', 'bizno'])
x = fl[fl.bz.isin(lo)].merge(ca, on='cd', how='left')
o = {'n_firms': int(lo.nunique()), 'route': x.route.value_counts().to_dict(),
     'ca_bizno_blank': int((x.bizno.fillna('').str.replace(r'\D', '', regex=True).str.len() != 10).sum()),
     'ca_bizno_other': int((x.bizno.fillna('').str.replace(r'\D', '', regex=True).str.len() == 10).sum()),
     'ca_crn_present': int((x.crn.fillna('').str.replace(r'\D', '', regex=True).str.len() == 13).sum())}
g = fa[fa.bz.isin(lo)]
o['n_apps'] = int(len(g)); o['apps_per_firm_median'] = float(g.groupby('bz').size().median())
o['first_known_year_dist'] = (g.groupby('bz').known.min() // 10000).value_counts().sort_index().astype(int).to_dict()
o['last_appl_year_dist'] = (g.groupby('bz').ad.max() // 10000).value_counts().sort_index().astype(int).to_dict()
o['code_prefix(1=법인,2=?,4/5=개인)'] = x.cd.str[0].value_counts().to_dict()
# 같은 코드가 다른 위험집합 기업에도 연결되는가(이름 충돌)
o['codes_shared_by_multiple_firms'] = int(fl.groupby('cd').bz.nunique().gt(1).sum())
json.dump(o, open(os.path.join(CA, 's5c_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
