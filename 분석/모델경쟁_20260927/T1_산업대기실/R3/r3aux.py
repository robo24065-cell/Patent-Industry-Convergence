# T1 R3 — 보조 원점 c=2015(진입 2016–2019) 결측 설계 확인: 국민연금(NPS, 2015-12 스냅샷~)·DART 재무(FY2013~)
# 개발 원점(c≤2012)과 그 학습 원점에는 두 자료가 없다 → 주 모형에 넣으면 학습 행 전부 결측. 대안 설계 = '2단 보정층':
#   주 모형 점수(외부데이터 없음) + 기업 NPS/DART 피처 → 로지스틱 보정층. 보정층은 결과창이 ≤2019로 끝나는 원점(c=2015)에서 학습하고,
#   최종시험(c=2019)에서는 그 보정층을 2019-12 스냅샷에 적용한다. 여기서는 c=2015 안에서 법인 해시 2-fold 교차적합으로 증분을 잰다.
#   python R3/r3aux.py --exp R3E00
import sys, json, argparse, pathlib
R3 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R3))
from r3 import *
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
EXTD = B/'분석/모델경쟁_20260927/_shared/ext'

ap = argparse.ArgumentParser(); ap.add_argument('--exp', default='R3E00'); ap.add_argument('--c', type=int, default=2015); a = ap.parse_args(); c = a.c
assert c + H <= GUARD_DEV
o = pd.read_parquet(R3/'runs'/a.exp/f'o{c}.parquet'); o = o[~o.prior_any.values].reset_index(drop=True)
s = 0.9 * fr.rank(o.px.values) + 0.1 * fr.rank(o.pl.values); o['s'] = s
o['ls'] = np.log(np.clip(s, 1e-6, 1 - 1e-6) / (1 - np.clip(s, 1e-6, 1 - 1e-6)))

# NPS: 원점 연말 스냅샷(2015-12: 자료생성 2015-11)만. 이전 스냅샷 없음 → 증감 없음
snap = f'{c}-12'
P = pd.read_parquet(EXTD/'firm_nps_firm_panel.parquet'); P = P[P.snap == snap].set_index('applicantCd')
L = pd.read_parquet(EXTD/'firm_nps_applicant_link.parquet').drop_duplicates('applicantCd').set_index('applicantCd')
f = o.firm
o['nps_link'] = f.isin(L.index).astype(float)            # 사업자번호 확보 여부(공개 권리정보)
o['nps_in'] = f.isin(P.index).astype(float)
o['nps_lemp'] = np.log1p(f.map(P.n_emp).fillna(0).values)
emp = f.map(P.n_emp).fillna(0).values
o['nps_hire'] = np.where(emp > 0, f.map(P.n_new).fillna(0).values / np.maximum(emp, 1), 0)
o['nps_loss'] = np.where(emp > 0, f.map(P.n_loss).fillna(0).values / np.maximum(emp, 1), 0)
o['nps_age'] = np.where(o.nps_in > 0, c - pd.to_numeric(f.map(P.first_adpt).astype(str).str[:4], errors='coerce').fillna(c).values, 0)
NPS = ['nps_link', 'nps_in', 'nps_lemp', 'nps_hire', 'nps_loss', 'nps_age']

# DART 재무: 공시 가용일 ≤ 원점 연말인 최신 회계연도
D = pd.read_parquet(EXTD/'firm_dart_fin_panel.parquet'); D = D[D.avail_date <= pd.Timestamp(f'{c}-12-31')]
D = D.sort_values(['jurir_no', 'fy']).drop_duplicates('jurir_no', keep='last').set_index('jurir_no')
fm = pd.read_parquet(CACHE/'firm_meta.parquet').set_index('applicant_cd').corp_rgstno.astype(str).str.replace('-', '')
j = f.map(fm)
o['dt_has'] = j.isin(D.index).astype(float)
o['dt_lrev'] = np.log1p(np.maximum(j.map(D.rev).fillna(0).values, 0)); o['dt_grev'] = j.map(D.g_rev).fillna(0).clip(-1, 3).values
o['dt_opm'] = j.map(D.opm).fillna(0).clip(-1, 1).values; o['dt_lev'] = j.map(D.lev).fillna(0).clip(0, 3).values
DT = ['dt_has', 'dt_lrev', 'dt_grev', 'dt_opm', 'dt_lev']

fold = (pd.util.hash_array(o.firm.values.astype(object)) % 2).astype(int)
def cf(cols):
    out = np.zeros(len(o))
    for fo in (0, 1):
        trm = fold != fo; X = o[cols].values.astype(float); sc = StandardScaler().fit(X[trm])
        m = LogisticRegression(max_iter=2000).fit(sc.transform(X[trm]), o.y_u.values[trm]); out[fold == fo] = m.predict_proba(sc.transform(X[fold == fo]))[:, 1]
    return out
b = baselines(c)['strongest']['auc_all']
res = dict(origin=c, exp=a.exp, n=len(o), pos=int(o.y_u.sum()), base_strongest=b,
           link=dict(nps_link=float(o.nps_link.mean()), nps_in=float(o.nps_in.mean()), dart_has=float(o.dt_has.mean()),
                     firms_nps_in=float(o.drop_duplicates('firm').nps_in.mean()), firms_dart=float(o.drop_duplicates('firm').dt_has.mean())))
rs = {}
for nm, cols in (('score_only', ['ls']), ('score+nps', ['ls'] + NPS), ('score+dart', ['ls'] + DT), ('score+nps+dart', ['ls'] + NPS + DT)):
    p = cf(cols); au = float(roc_auc_score(o.y_u.values, p)); rs[nm] = dict(auc=au, skill=skill(au, b))
res['model_raw_auc'] = float(roc_auc_score(o.y_u.values, o.s.values)); res['model_raw_skill'] = skill(res['model_raw_auc'], b)
res['crossfit'] = rs
# 법인 군집 부트스트랩: nps 보정 증분
rng = np.random.RandomState(0); firms = o.firm.unique(); fidx = o.groupby('firm').indices
p0 = cf(['ls']); p1 = cf(['ls'] + NPS); dl = []
for r in range(100):
    ff = rng.choice(firms, len(firms)); ii = np.concatenate([fidx[x] for x in ff])
    dl.append(roc_auc_score(o.y_u.values[ii], p1[ii]) - roc_auc_score(o.y_u.values[ii], p0[ii]))
res['nps_delta_auc_ci'] = [float(np.percentile(dl, 2.5)), float(np.mean(dl)), float(np.percentile(dl, 97.5))]
# 단변량(주 모형 점수 통제 전) 참고
res['uni_auc'] = {x: float(roc_auc_score(o.y_u.values, o[x].values)) for x in NPS + DT}
json.dump(res, open(R3/f'aux_nps_dart_{a.exp}_o{c}.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(res, ensure_ascii=False, indent=1))
