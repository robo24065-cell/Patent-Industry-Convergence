# -*- coding: utf-8 -*-
"""verify_0 -- 반박 시도(작업자 코드 import 없음, 독립 구현). 쓰기는 verify_0 폴더 안에만.
 v1 비자사 판정: 다른 방법(applicantInfo 이름 → 출원인코드 사전 매핑 + 별칭 포함)으로 다시 판정해 작업자 s02 결과와 대조, 무작위 40행 원문 표본 저장
 v2 최종창 재채점: o2019 산출(FROZEN·S0–S5)에서 numpy 순위 AUC(Mann–Whitney)로 직접 계산, 순위 혼합도 직접 구현
 v3 공식 총량: sqlite에서 SQL로 직접 센 연도별 디자인·실용신안 건수 대 KOSIS json 직접 파싱
 python -I v_indep.py <C10-T1 폴더>"""
import sys, json, re, sqlite3, pathlib
import numpy as np, pandas as pd
H = pathlib.Path(sys.argv[1]); V = H/'verify_0'
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; R3 = T1/'R3'
OUT = {}
# ---------------- v1
con = sqlite3.connect((B/'수집파이프라인/business_rights.sqlite').as_uri() + '?mode=ro', uri=True)
rl = pd.read_sql("select applicant_cd, rgst_no, rgst_date, json_extract(raw,'$.applicantInfo') a, json_extract(raw,'$.rightHolderInfo') h, json_extract(raw,'$.pertDiv') pd_ from right_list where substr(rgst_no,1,2) in ('30','20')", con)
rl = rl.drop_duplicates(['applicant_cd', 'rgst_no'])
aq = pd.read_sql("select applicant_cd, applicant_name, status from applicant_query", con)
app = pd.read_parquet(T1/'cache/ins_applicant.parquet', columns=['applicantCd', 'applicantName'])
def nz(s):  # 독립 정규화: 법인격 표기·괄호·공백·구두점 제거, 소문자
    s = str(s).lower()
    for w in ('주식회사', '(주)', '㈜', '유한회사', '유한책임회사', '재단법인', '사단법인', '학교법인', '가부시키가이샤', '가부시끼가이샤', '가부시기가이샤', '가부시키카이샤', '코포레이션', '인코포레이티드', '리미티드', '컴퍼니'):
        s = s.replace(w, '')
    return re.sub(r"[\s\.\,\-\(\)\'\"·&（）]", '', s)
names = pd.concat([aq[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'n'}), app.rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'n'})]).dropna()
names['k'] = names.n.map(nz); names = names[names.k.str.len() >= 2]
code_of = names.groupby('k').applicant_cd.agg(lambda s: set(s)).to_dict()
own = names[names.applicant_cd.isin(aq[aq.status == 'OK'].applicant_cd)].groupby('applicant_cd').k.agg(set).to_dict()
def selfv(cd, a):
    parts = [nz(p) for p in str(a).split(',')]
    for p in parts:
        if cd in code_of.get(p, set()): return True
    ks = own.get(cd, set()); aa = nz(a)
    return any((len(k) >= 3 and k in aa) for k in ks)
rl['self_v'] = [selfv(c, a) for c, a in zip(rl.applicant_cd.values, rl.a.values)]
w = pd.read_parquet(H/'cache/dsum_flags.parquet', columns=['applicant_cd', 'rgst_no', 'self']).drop_duplicates(['applicant_cd', 'rgst_no'])
m = rl.merge(w, on=['applicant_cd', 'rgst_no'])
for kind, pre in (('design', '30'), ('um', '20')):
    x = m[m.rgst_no.str[:2] == pre]
    OUT[f'v1_{kind}'] = dict(rows=int(len(x)), nonself_worker=float((~x.self).mean()), nonself_verifier=float((~x.self_v).mean()),
                             agree=float((x.self == x.self_v).mean()), worker_self_verifier_not=int((x.self & ~x.self_v).sum()), worker_not_verifier_self=int((~x.self & x.self_v).sum()))
smp = m[(~m.self)].sample(40, random_state=7)[['applicant_cd', 'rgst_no', 'rgst_date', 'a', 'h', 'self', 'self_v']]
smp.to_csv(V/'v1_nonself_sample40.csv', index=False, encoding='utf-8-sig')
# ---------------- v2
def auc(y, s):
    y = np.asarray(y).astype(bool); r = pd.Series(s).rank(method='average').values
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
def pct(v): return pd.Series(v).rank(pct=True, method='average').values
L = pd.read_parquet(H/'cache/final_labels.parquet'); keep = ~L.prior_any.values; y = L.y_u.values[keep]; nc = L.newc.values[keep].astype(bool)
bs = pd.read_parquet(R3/'cache_final/o2019/baseline_scores.parquet').loc[keep]
b_all = max(auc(y, bs[c].values) for c in bs.columns); b_new = max(auc(y[nc], bs[c].values[nc]) for c in bs.columns)
OUT['v2_base'] = dict(n=int(len(y)), pos=int(y.sum()), strongest_auc=b_all, strongest_auc_new=b_new)
for nm, p in [('FROZEN', R3/'runs/FROZEN_final/o2019.parquet')] + [(v, H/'runs'/v/'o2019.parquet') for v in ('S0', 'S1', 'S2', 'S3', 'S4', 'S5')]:
    if not p.exists(): continue
    o = pd.read_parquet(p, columns=['px', 'pl']).loc[keep]
    s = 0.9 * pct(o.px.values) + 0.1 * pct(o.pl.values)
    a = auc(y, s); an = auc(y[nc], s[nc])
    OUT[f'v2_{nm}'] = dict(auc=a, skill=(a - b_all) / (1 - b_all), auc_new=an, skill_new=(an - b_new) / (1 - b_new), top1000=int(y[np.argsort(-s)[:1000]].sum()))
# ---------------- v3
q = """select substr(rgst_no,1,2) p, substr(rgst_date,1,4) y, count(distinct case when substr(rgst_no,1,2)='30' then substr(rgst_no,1,9) else rgst_no end) n
       from right_list where substr(rgst_no,1,2) in ('30','20') and json_extract(raw,'$.pertDiv')='권리자란' group by 1,2"""
c = pd.read_sql(q, con)
KD = H/'download/kosis'
a40 = pd.DataFrame(json.load(open(KD/'data_TX_13801_A040_1_2010_2025.json', encoding='utf-8'))); a40['DT'] = pd.to_numeric(a40.DT)
a45 = pd.DataFrame(json.load(open(KD/'data_TX_13801_A045_1990_2016.json', encoding='utf-8'))); a45['DT'] = pd.to_numeric(a45.DT)
v3 = {}
for p, c40, c45 in (('30', 'A0003', '15138AG1ADAI'), ('20', 'A003', '15138AG1AB')):
    for yr in ('2005', '2006', '2015', '2017', '2019'):
        mine = int(c[(c.p == p) & (c.y == yr)].n.sum())
        if yr >= '2015':
            k = float(a40[(a40.C1 == c40) & (a40.C2 == 'B003') & (a40.C3 == 'C001') & (a40.PRD_DE == yr)].DT.sum())
        else:
            k = float(a45[(a45.C1 == c45) & (a45.PRD_DE == yr) & a45.C2.isin(['15138AI4ABAB', '15138AI4ADAB'])].DT.sum())
        v3[f'{p}_{yr}'] = dict(rl=mine, kosis_corp=k, ratio=mine / k)
OUT['v3'] = v3
json.dump(OUT, open(V/'v_indep_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
