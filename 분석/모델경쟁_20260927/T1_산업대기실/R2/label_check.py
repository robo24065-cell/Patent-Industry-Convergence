# 진입 판정 방식 결정용 진단 (개발 구간 정보만: 출원연도 ≤ 2017)
#  (1) 등록 지연: 2012–2014 출원 코호트의 등록분이 '출원 후 d일 이내' 등록된 비율 → 최종시험 창(2018–22 출원, 관측 끝 2025-08-22)에서
#      등록 기준 레이블이 놓칠 비율을 추정(최종 구간 결과 자체는 보지 않음)
#  (2) 개발 원점 c=2009 평가 패널: 등록 기준 양성 vs 공개(등록∪공개출원) 기준 양성 차이, 위험집합 안 '공개만 선행출원' 쌍 비율
import sys, pathlib, json
R2 = pathlib.Path(__file__).resolve().parent; T1 = R2.parent; sys.path.insert(0, str(T1))
import numpy as np, pandas as pd
from common import CACHE
out = {}
ins = pd.read_parquet(CACHE/'ins_firm.parquet', columns=['rgst_no', 'applicant_cd', 'ksic', 'ay', 'appl_date', 'rgst_date'])
ins = ins[(ins.ay >= 2000) & (ins.ay <= 2017)]
fm = pd.read_parquet(CACHE/'firm_meta.parquet').set_index('applicant_cd').tp
coh = ins[(ins.ay >= 2012) & (ins.ay <= 2014)].drop_duplicates('rgst_no')
lag = (pd.to_datetime(coh.rgst_date) - pd.to_datetime(coh.appl_date)).dt.days
fr = coh.applicant_cd.map(fm).eq('외국법인')
end = pd.Timestamp('2025-08-22')
res = {}
for y in range(2018, 2023):   # 최종 창 출원연도별 관측 가능 기간(연 중앙 출원 가정)
    d = (end - pd.Timestamp(f'{y}-07-01')).days
    res[str(y)] = dict(obs_days=int(d), frac_reg_observed_all=float((lag <= d).mean()), frac_domestic=float((lag[~fr] <= d).mean()), frac_foreign=float((lag[fr] <= d).mean()))
out['registration_lag_2012_14_cohort'] = res
# 개발 원점(c=2009) 평가 패널 기준
te = pd.read_parquet(R2/'cache/o2009/feat_c2009.parquet', columns=['firm', 'k', 'y', 'newc'])
a = pd.read_parquet(CACHE/'app_pub.parquet'); a = a[(a.ay >= 2000) & (a.ay <= 2017)]
K = sorted(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['ksic']).ksic.unique()); kidx = {k: i for i, k in enumerate(K)}
a['k'] = a.ksic.astype(str).map(kidx); a = a.dropna(subset=['k']); a['k'] = a.k.astype(int)
c = 2009
w = a[(a.ay > c) & (a.ay <= c + 5)].drop_duplicates(['applicant_cd', 'k'])[['applicant_cd', 'k']].assign(zp=1)
pre = a[a.ay <= c].drop_duplicates(['applicant_cd', 'k'])[['applicant_cd', 'k']].assign(zpre=1)
x = te.merge(w, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left').drop(columns='applicant_cd')
x = x.merge(pre, left_on=['firm', 'k'], right_on=['applicant_cd', 'k'], how='left').drop(columns='applicant_cd').fillna({'zp': 0, 'zpre': 0})
yu = np.maximum(x.y.values, x.zp.values)
out['c2009'] = dict(rows=len(x), pos_reg=int(x.y.sum()), pos_union=int(yu.sum()), pub_only_pos=int(((x.zp == 1) & (x.y == 0)).sum()),
                    reg_pos_not_in_pub=int(((x.y == 1) & (x.zp == 0)).sum()),
                    pairs_with_prior_pub_app=int(x.zpre.sum()), pos_union_among_prior_pub=int(yu[x.zpre.values == 1].sum()),
                    newc_pos_reg=int(x.y[x.newc].sum()), newc_pos_union=int(yu[x.newc.values].sum()))
json.dump(out, open(R2/'label_check.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print(json.dumps(out, indent=1, ensure_ascii=False))
