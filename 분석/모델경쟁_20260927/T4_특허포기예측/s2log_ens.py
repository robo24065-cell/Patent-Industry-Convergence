# -*- coding: utf-8 -*-
"""2단계 앙상블 버전 기록: final_models/run_<tag>/dryrun_result.json -> leaderboard.csv"""
import sys, json
from run import log_lb, base_auc
V = {
 'v44': ('[감사 수정 후 재측정] v37 4모형 앙상블 + 시점보정 패밀리 + dspst 수정 (final_run.py 파이프라인, run_v44)', '누수 제거 후 v37의 정직한 성능', 'Y', 'v37 0.6966 -> 누수 제거 후. 이후 버전의 기준'),
 'v45': ('v44 + L 시점 권리자 불납률·생존비율 + KOSIS 경기지표 — 4개 구성요소 모두 (run_v45)', '분류형에서 확인된 권리자·경기 효과가 앙상블에서도 유지', 'Y', ''),
 'v51': ('v45를 홀드아웃 계산예산 규칙과 같은 추출비율(분류 0.507, 위험 0.368, HGB 38.4만행)로 학습 (run_v51)', '홀드아웃(C=2019) 행 추출이 성능을 떨어뜨리는가 — 민감도', 'R', '진단: 추출해도 하락 없음 → 계산예산 규칙 유지'),
 'v53': ('★최종 동결: v52 구성 + 계산예산 상한 100만행(분류·위험·HGB; 로지스틱 80만) + 청크 공급 QuantileDMatrix — final_spec.json/final_run.py --mode dryrun (run_dryrun)', '홀드아웃·미래 재학습에서도 4GB 이하로 같은 규칙 적용(위험모형은 검증에서도 0.843 추출)', 'Y', '★최종. 프로세스 피크 3.16GB(위험 구성원)'),
 'v52': ('v45 + XGB 구성요소(분류·위험) 3개 배깅(seed·열순서 다르게, 로짓평균) (run_v52)', '열 표집 무작위성에 따른 분산(분류형 단독 0.677~0.685 흔들림) 감소', 'Y', ''),
}
for v in sys.argv[1:]:
    r = json.load(open(f'final_models/run_{"dryrun" if v == "v53" else v}/dryrun_result.json', encoding='utf-8'))
    ch, hy, kept, note = V[v]
    ci = r['bootstrap_95ci']
    note2 = (note + f" | 구성요소 {{{', '.join(f'{k}:{x:.4f}' for k, x in r['components'].items())}}} | AUC 95%CI {ci['auc']} skill CI {ci['skill']} (출원인 군집 부트스트랩 B={r['bootstrap_B']}) | L별AUC { {L: round(x['auc'], 4) for L, x in r['per_L'].items()} } | 산업집계 Spearman { {L: x['spearman_model'] for L, x in r['industry_level_validity'].items()} }")
    log_lb(v, ch, hy, r['model'], base_auc(), kept, note2)
