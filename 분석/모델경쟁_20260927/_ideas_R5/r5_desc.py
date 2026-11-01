# -*- coding: utf-8 -*-
"""기술통계·중요도(사후, 채택 판단에 쓰지 않음). 개발 원점만."""
import json, numpy as np, pandas as pd
from r5_common import *
from r5_eval_spec import SPEC
out = []
# J2: 집단별 2년 분쟁률(개발 원점 2012–2020 합산)
D = pd.read_parquet(f"{CACHE}/J2_panel.parquet"); D = D[D.t0.between(2012, 2020)]
def rate(name, m):
    out.append(dict(idea="J2", group=name, n=int(m.sum()), pos=int(D.y[m].sum()), rate_permil=1000 * D.y[m].mean()))
rate("전체", D.y >= 0)
rate("과거 분쟁 있음", D.own_disp > 0); rate("과거 분쟁 없음", D.own_disp == 0)
rate("권리이전 있음", D.n_trf > 0); rate("권리이전 없음", D.n_trf == 0)
rate("최근 2년 이전", D.trf_2y > 0)
rate("실시권 설정 있음", D.lic > 0)
rate("기술평가 있음", D.tech_eval > 0)
rate("해외 패밀리(시점정합) 있음", D.fam_offices > 0); rate("해외 패밀리 없음", D.fam_offices == 0)
rate("미국 등록 패밀리(RoS)", D.us_granted > 0)
rate("ETSI 표준 선언", D.etsi > 0)
rate("정부R&D 기원(NTIS)", D.gov > 0); rate("정부R&D 기원 아님", D.gov == 0)
rate("권리자 최근 5년 분쟁 있음", D.owner_disp5 > 0)
rate("외국 출원인", D.foreign == 1)
rate("청구항 ≥15", D.claims >= 15); rate("청구항 <5", D.claims < 5)
# J2 중요도: 챔피언 피처로 t0 ≤ 2018 학습, gain
from xgboost import XGBClassifier
ch = json.load(open(f"{CACHE}/champ_J2_xgb.json", encoding="utf-8"))["feats"]
rng = np.random.default_rng(0)
A = pd.read_parquet(f"{CACHE}/J2_panel.parquet"); A = A[A.t0 <= 2018]
A = A[(A.y == 1) | (rng.random(len(A)) < 0.05)]
m = XGBClassifier(random_state=0, **XGB_PARAMS).fit(A[ch].astype(float), A.y, sample_weight=np.where(A.y == 1, 1, 20))
imp = pd.Series(m.get_booster().get_score(importance_type="gain")).reindex(ch).fillna(0); imp = imp / imp.sum()
for f, v in imp.sort_values(ascending=False).items():
    out.append(dict(idea="J2", group="gain:" + f, rate_permil=v))
# J1: 집단별 성과율
J = pd.read_parquet(f"{CACHE}/J1_panel.parquet"); J = J[J.t0.between(2014, 2020)]
for g, s in J.groupby(J.perform_grp.astype(str)):
    out.append(dict(idea="J1", group="수행주체:" + g, n=len(s), pos=int(s.y.sum()), rate_permil=1000 * s.y.mean()))
for nm_, m_ in [("KSIC50 b2 배정", J.b2_dict_assigned == 1), ("b2 미배정", J.b2_dict_assigned == 0), ("b1 배정", J.b1_tfidf_assigned == 1), ("b1 미배정", J.b1_tfidf_assigned == 0)]:
    out.append(dict(idea="J1", group=nm_, n=int(m_.sum()), pos=int(J.y[m_].sum()), rate_permil=1000 * J.y[m_].mean()))
ch1 = json.load(open(f"{CACHE}/champ_J1_xgb.json", encoding="utf-8"))["feats"]
B = pd.read_parquet(f"{CACHE}/J1_panel.parquet"); B = B[B.t0 <= 2018]
m = XGBClassifier(random_state=0, **XGB_PARAMS).fit(B[ch1].astype(float), B.y)
imp = pd.Series(m.get_booster().get_score(importance_type="gain")).reindex(ch1).fillna(0); imp = imp / imp.sum()
for f, v in imp.sort_values(ascending=False).head(12).items():
    out.append(dict(idea="J1", group="gain:" + f, rate_permil=v))
# J3: 합병 기업과 국세청 폐업 겹침, I2b 폐업 레이블 중 합병 비중
K = pd.read_parquet(f"{CACHE}/J3_panel.parquet")
for t0, s in K.groupby("t0"):
    out.append(dict(idea="J3", group=f"t0={t0} 폐업(국세청, 창 안) 중 합병 비중", n=int(s.closed_win.sum()), pos=int(((s.y == 1) & (s.closed_win == 1)).sum()),
                    rate_permil=1000 * ((s.y == 1) & (s.closed_win == 1)).sum() / max(1, s.closed_win.sum())))
R = pd.DataFrame(out); R.to_csv(f"{OUT}/R5_desc.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 200); print(R.round(4).to_string())
