# 2단계 개선 버전 실행기: python run2.py <ver>   (학습 L2006-2011 → 검증 L2014·2015)
import sys
sys.stdout.reconfigure(encoding='utf-8')
from v2lib import *
V2 = {
 'v29': dict(cfg=dict(recency=0.85), ref='v28',
   change='v28 + 권리자 모형 학습 가중에 최근성 γ^(2011-L), γ=0.85 (sqrt 보유건수 가중과 곱)',
   hyp='권리자 매각 동학은 비정상(학습기 내 0.9 vs 시간외 0.72) → 최근 기준일 가중을 높이면 시간외 권리자 순위가 개선'),
 'v30': dict(cfg=dict(extra=['mkt'], extra_level=MKT_OWN), ref='v28',
   change='v28 + 권리자 매각성향(최근 5년 양도유출/(보유+유출)을 전체평균으로 수축 a=20, 매각 경험 여부, 마지막 매각 후 경과연수)을 권리자 모형에 투입',
   hyp='원시 매각 건수(v16)는 번들 암기였지만, 규모 보정·수축된 성향은 권리자 1행 표본에서 일반화된다'),
 'v31': dict(cfg=dict(extra=['fam'], extra_pat=FAM, extra_mean=['n_fam_cc', 'fam_US', 'fam_CN']), ref='v28',
   change='v28 + 제공 5개국 메타 family 기반 해외 패밀리(US·EP·JP·CN·WO 여부, 국가 수, 문헌 수; 출원 2.5년 미경과는 결측)를 특허 모형과 권리자 평균에 투입',
   hyp='해외 패밀리가 있는 특허는 국제적 가치가 있어 거래 대상(특히 해외기업 간 이관·국내→해외 매각)이 되기 쉽다 — 5개국 데이터 융합'),
 'v32': dict(cfg=dict(extra=['mkt'], extra_mean=MKT_MKT, extra_pat=MKT_MKT), ref='v28',
   change='v28 + 기술시장 깊이(같은 KSIC·IPC4 최근 3년 서로 다른 양수인·양도인 수/생존 1천건, 양수인 중 외국·IP관리법인·개인 비중)를 특허 모형과 권리자 평균에 투입',
   hyp='양수인이 많은(깊은) 기술시장일수록 매각이 쉽다 — 유동성 자체를 피처로'),
  'v33': dict(cfg=dict(recency=0.85, extra=['ind'], extra_pat=IND, extra_level=IND), ref='v29',
   change='v29 + 개인 발명가의 법인화 신호(개인 권리자 이름이 최근 5년 출원·L 이전 등록된 국내법인 특허의 발명자로 등장한 건수, 같은 KSIC 한정 건수)를 특허·권리자 모형에 투입',
   hyp='개인→국내법인 양도가 2003년 45건→2018년 776건으로 급증: 개인 발명가가 법인을 세우거나 합류하면(법인 특허 발명자로 등장) 보유 특허를 그 법인에 넘긴다'),
  'v34': dict(cfg=dict(recency=0.85, extra=['cost'], extra_level=COST), ref='v29',
   change='v29 + 포트폴리오 유지비 압박(권리자 연간 유지비 합, 최근 출원 대비 유지비, 9년+ 노후 특허 비중, 3년- 신규 비중)을 권리자 모형에 투입',
   hyp='유지비 부담이 R&D 활동 대비 커진 권리자는 포트폴리오를 정리(매각)한다'),
  'v35': dict(cfg=dict(recency=0.85, extra=['mkt'], extra_level=MKT_OWN), ref='v29',
   change='v29 + v30의 수축 매각성향 3개(권리자 모형)',
   hyp='v30은 AUC 불변이나 PR-AUC·p@1%를 올렸음 → 최근성 가중과 결합 시 상보적인지 확인'),
  'v36': dict(cfg=dict(recency=0.85, pat_recency=0.85), ref='v29',
   change='v29 + 특허 단위 모형(p_pat)에도 최근성 가중 γ=0.85',
   hyp='권리자 내 상대비를 만드는 특허 점수도 최근 기준일 패턴에 맞추면 결합 점수가 개선'),
  'v37': dict(cfg=dict(recency=0.85, hurdle=True), ref='v29',
   change='v29 권리자 모형을 허들 구조로: P(3년 안 1건 이상 매각) XGB 분류 × E[매각비율|매각] XGB 회귀(매각 권리자만 학습)',
   hyp='권리자 매각률 목표는 0과 번들(대량) 비율의 이봉 분포 → 매각 여부와 매각 규모를 분리하면 대형 권리자의 순위가 안정된다'),
}
if __name__ == '__main__':
    v = sys.argv[1]; s = V2[v]; kept = sys.argv[2] if len(sys.argv) > 2 else '?'
    ap = fit_apply(s['cfg'], range(2006, 2012), VAL_L)
    evaluate(v, ap, s['change'], s['hyp'], kept, ref=s['ref'])
