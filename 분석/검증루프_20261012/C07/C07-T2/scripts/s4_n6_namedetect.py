# C07-T2 s4: E-C05T2-05 독립 재검 — N6 첫 특허를 '상호'로 따로 찾으면 공식 레이블이 몇 %를 놓치는가, 그 '놓침'이 동명 법인 문제인가.
# 원 작업자 스크립트 import 없음. 원천: N6 final_preds_r5, ntis_project_raw, kr_patents, CORP_APPLICANT.
import os, sys, json, re
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = ROOT + r"\분석\모델경쟁_20260927"; SH = MC + r"\_shared\ext"
OUT = ROOT + r"\분석\검증루프_20261012\C07\C07-T2"
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
nrm = lambda s: re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUF.sub("", str(s or "")).upper())
def auc(y, p):
    y = np.asarray(y).astype(int); r = pd.Series(np.asarray(p, float)).rank(method='average').values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
sk = lambda a, b: (a - b) / (1 - b)
o = {}
pr = pd.read_parquet(MC + r"\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet"); pr['bz'] = pr.bz.astype(str).str.zfill(10)
n = pd.read_parquet(SH + r"\ntis_project_raw.parquet", columns=['year', 'corp_rgst_no', 'research_agency'])
n['bz'] = n.corp_rgst_no.fillna('').str.replace(r'\D', '', regex=True)
n = n[(n.bz.str.len() == 10) & (n.year <= 2021)]
n['nm'] = n.research_agency.map(nrm)
modal = n[n.nm.str.len() >= 3].groupby(['bz', 'nm']).size().reset_index(name='c').sort_values(['bz', 'c'], ascending=[True, False]).drop_duplicates('bz')
# NTIS 기업 사이 유일 상호
cnt = modal.groupby('nm').bz.nunique(); modal = modal[modal.nm.map(cnt) == 1]
F = pr.merge(modal[['bz', 'nm']], on='bz', how='left')
o['risk_n'] = len(F); o['with_unique_name'] = int(F.nm.notna().sum())
KP = ROOT + r"\공모전데이터\작업데이터\kr_patents.parquet"; cols = pq.ParquetFile(KP).schema_arrow.names
k = pq.read_table(KP, columns=[cols[1], cols[8]]).to_pandas(); k.columns = ['ad', 'names']
k['ad'] = pd.to_numeric(k.ad, errors='coerce'); k = k[k.ad.between(20220101, 20231231)]
k['nm'] = k.names.map(nrm)
hit = set(k.nm)
F['det'] = F.nm.isin(hit).astype(int)
o['detected'] = int(F.det.sum()); o['detected_and_y1'] = int(((F.det == 1) & (F.y == 1)).sum())
o['capture_rate'] = round(o['detected_and_y1'] / max(o['detected'], 1), 4)
o['missed'] = int(((F.det == 1) & (F.y == 0)).sum())
# 동명 법인: CORP_APPLICANT에서 같은 정규화 상호가 2개 이상 출원인코드(사업자번호 서로 다름)
rows = []
with open(ROOT + r"\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt", encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('¶')
        if len(p) >= 5: rows.append((p[0].strip(), nrm(p[1]), re.sub(r'\D', '', p[4])))
ca = pd.DataFrame(rows, columns=['cd', 'nm', 'biz'])
ncode = ca.groupby('nm').cd.nunique()
own = ca[ca.biz.str.len() == 10].groupby('biz').cd.nunique()
F['n_codes_same_name'] = F.nm.map(ncode).fillna(0).astype(int)
F['has_own_code'] = F.bz.isin(set(own.index))
miss = F[(F.det == 1) & (F.y == 0)]; capt = F[(F.det == 1) & (F.y == 1)]
o['missed_same_name_ge2_share'] = round(float((miss.n_codes_same_name >= 2).mean()), 4)
o['captured_same_name_ge2_share'] = round(float((capt.n_codes_same_name >= 2).mean()), 4)
o['missed_no_own_CA_code_share'] = round(float((~miss.has_own_code).mean()), 4)
o['missed_name_absent_in_CA_share'] = round(float((miss.n_codes_same_name == 0).mean()), 4)
base = sk(auc(F.y, F.p_champ), auc(F.y, F.p_m0)); o['base_skill'] = base
ya = np.maximum(F.y.values, F.det.values); o['skill_y_or_det'] = sk(auc(ya, F.p_champ), auc(ya, F.p_m0))
# 동명(≥2코드) 탐지를 빼고 더한 판
d2 = F.det.values * (F.n_codes_same_name.values < 2)
yb = np.maximum(F.y.values, d2); o['skill_y_or_det_unique_in_CA'] = sk(auc(yb, F.p_champ), auc(yb, F.p_m0)); o['added_unique_in_CA'] = int(((d2 == 1) & (F.y == 0)).sum())
o['missed_score_pctile'] = round(float(pd.Series(F.p_champ).rank(pct=True)[(F.det == 1) & (F.y == 0)].mean()), 4)
o['official_pos_score_pctile'] = round(float(pd.Series(F.p_champ).rank(pct=True)[F.y == 1].mean()), 4)
json.dump(o, open(OUT + r"\cache\s4_n6_namedetect.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(o, ensure_ascii=False, indent=1, default=float))
