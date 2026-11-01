# -*- coding: utf-8 -*-
"""평가 레이블(PREREG §3): 공식 평가 패널 전체 행 + Call 쌍의 공식 정의 y_u(2020–23 출원, 2025-08-22까지 등록·공개)를 동결 r3.reg_lab·pub_lab(가드 2023)으로 계산.
 → cache/labels.parquet(firm,k,y_reg,y_pub,y_u,src), cache/labels_check.json(C15 final_labels 대조)"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c16t1 import *
import pickle
import final_run_r3 as F, r3, r2
if (CC/'labels.parquet').exists(): print('exists'); sys.exit(0)
te = pd.read_parquet(FC/'prep_te.parquet', columns=['firm', 'k', 'cut', 'newc', 'prior_any'])
PP = pickle.load(open(CC/'build'/'pop.pkl', 'rb'))
cx = pd.DataFrame(PP['Call'], columns=['firm', 'k']); cx['cut'] = 2019; cx['newc'] = False; cx['prior_any'] = False
d = pd.concat([te.assign(src='te'), cx.assign(src='Call')], ignore_index=True)
r2.light_load()
with F.guard(2023):
    d['y_reg'] = r3.reg_lab(d, None); d['y_pub'] = r3.pub_lab(d, r3.END)
d['y_u'] = np.maximum(d.y_reg.values, d.y_pub.values).astype(np.int8)
d[['firm', 'k', 'src', 'prior_any', 'y_reg', 'y_pub', 'y_u']].to_parquet(CC/'labels.parquet', index=False)
L15 = pd.read_parquet(C15/'cache'/'t1'/'final_labels.parquet'); t = d[d.src == 'te'].reset_index(drop=True)
chk = dict(eq_C15_y_u=bool((L15.y_u.values == t.y_u.values).all() and (L15.firm.values == t.firm.values).all()),
           kept_rows=int((~t.prior_any).sum()), kept_pos=int(t.y_u[~t.prior_any].sum()), kept_pos_reg=int(t.y_reg[~t.prior_any].sum()),
           Call_rows=int((d.src == 'Call').sum()), Call_officialdef_pos=int(d.y_u[d.src == 'Call'].sum()))
jdump(chk, CC/'labels_check.json'); print(chk)
