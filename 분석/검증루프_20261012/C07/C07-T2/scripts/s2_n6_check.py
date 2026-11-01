# C07-T2 s2: N6 267·410행 산술 검사 + '공란'의 성격(등록 공란 vs 미등록) 측정. 원 작업자 스크립트 import 없음.
import os, sys, json, sqlite3
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = ROOT + r"\분석\모델경쟁_20260927"
C6 = ROOT + r"\분석\검증루프_20261012\C06\C06-T3"
OUT = ROOT + r"\분석\검증루프_20261012\C07\C07-T2"
def auc(y, p):
    y = np.asarray(y).astype(int); r = pd.Series(np.asarray(p, float)).rank(method='average').values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
sk = lambda a, b: (a - b) / (1 - b)
o = {}
fr = json.load(open(MC + r"\N6_정부RnD기업_첫특허전환\final_result_r5.json", encoding='utf-8'))
o['official_skill_json'] = fr['skill']
pr = pd.read_parquet(MC + r"\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet")
o['official_skill_recalc'] = sk(auc(pr.y, pr.p_champ), auc(pr.y, pr.p_m0)); o['n'] = len(pr); o['pos'] = int(pr.y.sum())
# C06-T3 시나리오 레이블(데이터 파일)로 skill 재계산
d = pd.read_csv(C6 + r"\n6_blank_full.csv")
for s in ['L', 'M', 'U']:
    m = d[d['in_risk_' + s] == 1]
    o['skill_' + s] = sk(auc(m['y_' + s], m.p_champ), auc(m['y_' + s], m.p_m0)); o['pos_' + s] = int(m['y_' + s].sum()); o['n_' + s] = len(m)
    o['flips_' + s] = int(((d.y == 0) & (d['y_' + s] == 1)).sum())
# 명세별 Δ(C06-T3 n6_rescore_blank.json·verify_0 출력의 값) → 공식 정확값에 더해 사사오입
spec = {'M': 0.009921411483358178, 'M_keep': 0.009040967044219583, 'M_strict': 0.00865641968781053,
        'M_strict_precision_adj': 0.007943141521247083, 'verify0_indep_M': 0.007738849373733064,
        'M_NTIS': 0.0068460883038844456, 'M_NTIS_strict': 0.0063932799115759}
base = o['official_skill_json']
o['spec_skill'] = {k: [round(v, 4), round(base + v, 4), f'{base + v:.3f}'] for k, v in spec.items()}
o['range_dskill_3dp'] = [f'{min(spec.values()):.3f}', f'{max(spec.values()):.3f}']
o['range_skill_3dp'] = [f'{base + min(spec.values()):.3f}', f'{base + max(spec.values()):.3f}']
# '공란'의 성격: 제공 KR 출원연도별 등록번호 공란 비율 + 결과창 공란 중 권리목록상 등록된 문서(등록 공란) 비율
KP = ROOT + r"\공모전데이터\작업데이터\kr_patents.parquet"
nm = pq.ParquetFile(KP).schema_arrow.names
k = pq.read_table(KP, columns=[nm[0], nm[1], nm[3]]).to_pandas(); k.columns = ['appl_no', 'ad', 'rg']
k['appl_no'] = k.appl_no.astype(str).str.replace(r'\D', '', regex=True)
k['blank'] = k.rg.fillna('').astype(str).str.strip() == ''
k['y'] = pd.to_numeric(k.ad, errors='coerce') // 10000
o['col_names_used'] = [nm[0], nm[1], nm[3]]
o['blank_share_by_year'] = {int(y): [int(g.blank.sum()), len(g), round(g.blank.mean(), 4)] for y, g in k[k.y.between(2014, 2024)].groupby('y')}
con = sqlite3.connect('file:' + ROOT + r"\수집파이프라인\business_rights.sqlite?mode=ro", uri=True)
rl = pd.read_sql("select distinct appl_no from right_list where substr(rgst_no,1,2)='10'", con); con.close()
rl = set(rl.appl_no.astype(str).str.replace(r'\D', '', regex=True))
w = k[(k.y.between(2022, 2023)) & k.blank]
o['window_blank'] = len(w); o['window_blank_in_rightlist_registered'] = int(w.appl_no.isin(rl).sum())
o['window_blank_registered_share_lowerbound'] = round(w.appl_no.isin(rl).mean(), 4)
json.dump(o, open(OUT + r"\cache\s2_n6_check.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(o, ensure_ascii=False, indent=1))
