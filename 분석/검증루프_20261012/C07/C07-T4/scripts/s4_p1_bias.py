# P1(a)(b)·P3 수치, 깔때기, 사후 변형(판정 밖), 편향 점검 행(bias_checks.csv 원자료), r_post(P2(b) 대체 정의, 사실 기록)
import sys, time, json
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T4\scripts')
from lib7 import *
t = time.time(); R = {}; BR = []
def smd(a, b):
    a = pd.Series(a, dtype=float).dropna(); b = pd.Series(b, dtype=float).dropna()
    sd = np.sqrt((a.var() + b.var()) / 2); return float((a.mean() - b.mean()) / sd) if sd > 0 else np.nan
rows = []
for Y in DEV:
    b = pd.read_parquet(C + rf'\base_{Y}.parquet'); l = pd.read_parquet(C + rf'\lab_prov_{Y}.parquet')
    assert (b.rgst_no.values == l.rgst_no.values).all()
    d = pd.concat([b, l.drop(columns=['rgst_no', 't0', 'ksic'])], axis=1); d['t0'] = Y
    rows.append(d)
D = pd.concat(rows, ignore_index=True)
# P1
per = D.groupby('t0').agg(n=('y', 'size'), pos=('y', 'sum'), ev=('has_ev', 'sum'), pos_mode=('y_mode', 'sum'), pos_A=('y_A', 'sum'), pos_AB=('y_AB', 'sum'))
per['rate'] = per.pos / per.n
uniq = D[D.y == 1].rgst_no.nunique()
lock_n = len(pd.read_parquet(C + rf'\base_{LOCK}.parquet', columns=['rgst_no']))
base_rate = per.loc[[2021, 2022], 'rate'].mean()
R['P1'] = {'dev_unique_pos': int(uniq), 'dev_pos_sum': int(per.pos.sum()), 'per_origin': per.reset_index().to_dict('records'),
           'lock_risk_n': lock_n, 'base_rate_2021_2022': float(base_rate), 'lock_expected': float(lock_n * base_rate),
           'pass_a': bool(uniq >= 1000), 'pass_b': bool(lock_n * base_rate >= 1000)}
# 깔때기(개발 합)
e = D[D.has_ev == 1]
indiv = e.buyer_type.isin([4, 6])
R['funnel'] = {'risk_rows': len(D), 'accepted_transfer': len(e), 'nonindividual_buyer': int((~indiv).sum()),
               'nonindividual_nonempty': int(((~indiv) & (e.buyer_nonempty == 1)).sum()), 'cross_industry_pos': int(e.y.sum()),
               'nonindividual_empty': int(((~indiv) & (e.buyer_nonempty == 0)).sum())}
# 사후 변형(판정 밖)
lit = ((~indiv) & ((e.buyer_nonempty == 0) | (e.y == 1)))       # 문언 그대로: 빈 포트폴리오 법인·기관 양수인도 '산업 간'
e2 = e.assign(lit=lit.astype(int))
pl = e2.groupby('t0').lit.sum() / per.n
pm = per.pos_mode / per.n
R['posthoc'] = {
    'Y_mode_lock_expected': float(lock_n * pm.loc[[2021, 2022]].mean()), 'Y_mode_dev_unique': int(D[D.y_mode == 1].rgst_no.nunique()),
    'literal_incl_empty_lock_expected': float(lock_n * pl.loc[[2021, 2022]].mean()), 'literal_dev_pos': int(lit.sum()),
    'any_accepted_lock_expected': float(lock_n * (per.ev / per.n).loc[[2021, 2022]].mean()),
    'main_18m_window_scaled': float(lock_n * base_rate * 1.5),
    'main_all_dev_rate': float(lock_n * per.pos.sum() / per.n.sum())}
# P3
R['P3'] = {'origins_ge100': int((per.pos >= 100).sum()), 'origins_ge200': int((per.pos >= 200).sum()), 'min_pos': int(per.pos.min())}
# ---- 편향 점검
for Y, g in e.groupby('t0'):
    tp = g.buyer_type.value_counts(normalize=True)
    for k, nm in [(1, '국내법인'), (2, '대학'), (3, '연구기관'), (4, '국내개인'), (5, '외국법인'), (6, '외국개인'), (7, '국가공공')]:
        BR.append(('양수인 유형 비율', Y, nm, round(float(tp.get(k, 0)), 4), len(g), ''))
    ni = g[~g.buyer_type.isin([4, 6])]
    BR.append(('포트폴리오 연결률(비개인 양수인 중 비어 있지 않음)', Y, '전체', round(float(ni.buyer_nonempty.mean()), 4), len(ni), ''))
    for k, nm in [(1, '국내법인'), (2, '대학'), (3, '연구기관'), (5, '외국법인'), (7, '국가공공')]:
        x = ni[ni.buyer_type == k]
        if len(x): BR.append(('포트폴리오 연결률', Y, nm, round(float(x.buyer_nonempty.mean()), 4), len(x), ''))
ni = e[~e.buyer_type.isin([4, 6])].copy()
ni['orig_corp'] = (ni.orig_type == 1).astype(int); ni['bulk10'] = (ni.bundle >= 10).astype(int)
for c in ['age', 'log_claims', 'n_inv', 'orig_corp', 'bulk10', 'log_own_port', 'n_ipc']:
    a = ni[ni.buyer_nonempty == 1][c]; b = ni[ni.buyer_nonempty == 0][c]
    BR.append(('연결 vs 비연결 이전 SMD(비개인 양수인, 개발 합)', 'dev', c, round(smd(a, b), 3), f'{len(a)}/{len(b)}', f'평균 {a.mean():.3f} vs {b.mean():.3f}'))
# 양성 vs 같은 산업 이전(연결된 비개인) SMD
xx = ni[ni.buyer_nonempty == 1]
for c in ['age', 'log_claims', 'n_inv', 'orig_corp', 'bulk10']:
    a = xx[xx.y == 1][c]; b = xx[xx.y == 0][c]
    BR.append(('양성(산업 간) vs 같은 산업 이전 SMD', 'dev', c, round(smd(a, b), 3), f'{len(a)}/{len(b)}', ''))
# 경로 민감도: 레이블 뒤집힘
for lab in ['y_A', 'y_AB']:
    f = ((D[lab] != D.y) & ((D[lab] == 1) | (D.y == 1))).sum() / max(1, ((D[lab] == 1) | (D.y == 1)).sum())
    BR.append(('포트폴리오 경로 민감도: 양성 뒤집힘률(합집합 기준)', 'dev', lab + ' vs A∪B∪C', round(float(f), 4), int(D[lab].sum()), f'주 정의 양성 {int(D.y.sum())}'))
# 산업·연도 분포, 집중
pos = D[D.y == 1]
kd = pos.ksic.value_counts(normalize=True)
BR.append(('양성 KSIC 상위 5 점유', 'dev', ','.join(kd.index[:5]), round(float(kd.iloc[:5].sum()), 4), len(pos), f'위험집합 같은 5개 점유 {D.ksic.isin(kd.index[:5]).mean():.4f}'))
BR.append(('양성 KSIC 종류 수', 'dev', '', int(kd.size), len(pos), ''))
tb = pos.buyer.value_counts(normalize=True)
BR.append(('양성 상위 10 양수인 점유', 'dev', '|'.join(tb.index[:3]) + '…', round(float(tb.iloc[:10].sum()), 4), int(tb.size), '양수인 수'))
BR.append(('양성 중 일괄(같은 날 같은 양수인 ≥10건) 비중', 'dev', '', round(float((pos.bundle >= 10).mean()), 4), len(pos), f'양성 중 ≥51건 {(pos.bundle >= 51).mean():.4f}'))
for Y in DEV:
    p = pos[pos.t0 == Y]
    BR.append(('연도별 양성·일괄 비중', Y, '', int(len(p)), round(float((p.bundle >= 10).mean()), 4) if len(p) else 0, f'양수인 {p.buyer.nunique()}'))
# r_post(P2(b) 대체 정의): 이전 뒤에도 옛 이름(제공 출원인명 ≠ registry 이름)으로 출원한 양수인
docs = pd.read_parquet(C + r'\docs.parquet', columns=['appl_no', 'adate', 'nn'])
pat = pd.read_parquet(C + r'\pat.parquet', columns=['rgst_no', 'applNo'])
app = pd.read_parquet(C + r'\app.parquet', columns=['rgst_no', 'seq', 'applicantName'])
app['rn'] = app.applicantName.map(norm)
a = app.merge(pat, on='rgst_no').merge(docs, left_on='applNo', right_on='appl_no')
a0 = a[a.seq == a.groupby('rgst_no').seq.transform('min')]
R['registry_name_ne_provided_share'] = float((a0.rn != a0.nn).mean())
old = a[(a.rn != a.nn)].groupby('rn').adate.max()      # registry 이름별 '옛 이름' 출원 최신일
pp = pos.copy(); pp['old_last'] = pp.buyer.map(old)
pp['post'] = pp.old_last.notna() & (pp.old_last > pp.ev_date.dt.strftime('%Y%m%d'))
R['r_post'] = float(pp.post.mean()); R['r_post_n'] = int(pp.post.sum())
R['r_post_any_rename_evidence'] = float(pp.old_last.notna().mean())
BR.append(('사후 명칭 r_post(이전 뒤 옛 이름 출원 존재)', 'dev', '양성', round(R['r_post'], 4), len(pp), f"이름 변경 흔적 있는 양수인 비율 {R['r_post_any_rename_evidence']:.4f}"))
BR.append(('registry 이름 ≠ 제공 출원인명(등록특허 첫 출원인)', 'all', '', round(R['registry_name_ne_provided_share'], 4), len(a0), 'T5 감사 7.2%와 비교'))
# 등록 1년 이내 특허의 양성률(공란 영향 추정용)
D['young'] = D.age < 1.0
yr = D.groupby('young').y.mean()
R['young_rate'] = float(yr.get(True, np.nan)); R['old_rate'] = float(yr.get(False, np.nan))
R['sec'] = round(time.time() - t, 1)
pd.DataFrame(BR, columns=['check', 'origin', 'group', 'value', 'n', 'note']).to_csv(C + r'\s4_bias_rows.csv', index=False, encoding='utf-8-sig')
json.dump(R, open(C + r'\s4_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
