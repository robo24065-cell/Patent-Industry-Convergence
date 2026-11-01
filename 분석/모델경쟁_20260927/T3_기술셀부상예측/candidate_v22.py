"""최선 버전 v22 재현: 검증 t0=2011/2013/2015 롤링 원점. 사용: python candidate_v22.py"""
from models import *
D=enrich(H.data())
R,P=H.rolling(best_fp(),D)
print(H.summarize(R)); print(R.round(4).to_string())
P.to_parquet(os.path.join(L.HERE,'preds','candidate_v22_rerun.parquet'))
