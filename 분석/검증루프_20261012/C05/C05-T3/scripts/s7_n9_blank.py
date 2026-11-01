# -*- coding: utf-8 -*-
"""C05-T3 s7: N9 — 제공 KR 등록번호 공란(2023-11 이후 등록) 때문에 NTIS '등록' 성과 레코드가 연결 안 된 경우의 레이블 영향.
NTIS KR 등록 레코드 번호 -> 권리목록(right_list) 등록번호 -> 출원번호 -> 제공 meta(rgst_no 공란) -> appl_year 2022–23 -> 과제 t=2022 양성 추가?
"""
import duckdb, pandas as pd, numpy as np, os, sys, io, json, sqlite3
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
B = os.path.join(ROOT, "분석"); SH = os.path.join(B, r"모델경쟁_20260927\_shared")
N9 = os.path.join(B, r"모델경쟁_20260927\N9_정부RnD과제_특허성과예측")
OUT = os.path.join(B, r"검증루프_20261012\C05\C05-T3"); CA = os.path.join(OUT, "cache")
M = os.path.join(SH, "meta", "*.parquet").replace("\\", "/"); R = os.path.join(SH, "ext", "ntis_rpatent_raw.parquet").replace("\\", "/")
o = {}
kr = duckdb.sql(f"select appl_no, any_value(rgst_no) rg, any_value(appl_year)::int ay from read_parquet('{M}') where cc='KR' group by 1").df()
rp = duckdb.sql(f"select ResultID, ProjectID, RegistType, num_digits num from read_parquet('{R}') where is_kr and length(num_digits)=13").df()
la = pd.read_parquet(os.path.join(CA, 'n9_links_A.parquet'))
linked_res = set(la.ResultID)
un = rp[~rp.ResultID.isin(linked_res)]
o['unlinked_kr13_records'] = int(len(un)); o['unlinked_by_type'] = un.RegistType.value_counts().to_dict()
con = sqlite3.connect('file:' + os.path.join(ROOT, r"수집파이프라인\business_rights.sqlite") + '?mode=ro', uri=True)
rl = pd.read_sql("select distinct rgst_no, appl_no, rgst_date from right_list where substr(rgst_no,1,2)='10'", con); con.close()
x = un.merge(rl, left_on='num', right_on='rgst_no').merge(kr[kr.rg.isna()], on='appl_no')
o['recoverable_records'] = int(len(x)); o['recoverable_patents'] = int(x.appl_no.nunique())
o['recoverable_window_patents'] = int(x[x.ay.between(2022, 2023)].appl_no.nunique())
pr = pd.read_parquet(os.path.join(N9, "final_preds_r6.parquet"))
add = set(x[x.ay.between(2022, 2023)].ProjectID) & set(pr.pjt_no)
newpos = add - set(pr[pr.y == 1].pjt_no)
o['t2022_projects_touched'] = len(add); o['t2022_new_positives'] = len(newpos)
y2 = (pr.y.astype(bool) | pr.pjt_no.isin(newpos)).astype(int)
a = roc_auc_score(y2, pr.p_champ); b = roc_auc_score(y2, pr.p_M0_L2)
k = int(np.ceil(0.01 * len(pr))); top = np.argsort(-pr.p_champ.values, kind='stable')[:k]
o['rescore_with_recovered'] = {'pos': int(y2.sum()), 'auc': a, 'auc_M0L2': b, 'skill': (a - b) / (1 - b), 'top1': float(y2.values[top].mean()),
                               'dskill_vs_official': (a - b) / (1 - b) - 0.3917220538280016}
json.dump(o, open(os.path.join(CA, 's7_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
