# T1 R2 — 개발 원점별 분류기·패널 구축 (1라운드 final_run 함수 그대로 재사용, 원점별 cap)
#   python R2/build_origin.py --c 2010      (종료코드 3이면 같은 명령 재실행 → 캐시에서 이어감)
# 개발 원점 c ≤ 2012 (5년 진입창 끝 ≤2017 < F=2018). 학습 원점 c-8..c-5, 레이블·분류기 cap = c-12-31.
import sys, pathlib, argparse
R2 = pathlib.Path(__file__).resolve().parent; T1 = R2.parent
sys.path.insert(0, str(T1))
import final_run as fr   # T1_STRICT=1, T1_VIS=rgst 설정됨
import lib

ap = argparse.ArgumentParser(); ap.add_argument('--c', type=int, required=True); ap.add_argument('--final', action='store_true')
a = ap.parse_args(); c = a.c
if a.final:
    # 🔒 최종시험 전용(c=2017): 평가 진입창 2018–2022 → 가드 2022. 개발 단계에서는 실행하지 않는다.
    assert c == 2017; lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = 2022; FC = R2/'cache_final'/f'o{c}'
else:
    assert c <= 2012, '🔒 개발 원점은 c ≤ 2012'
    lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = 2017; FC = R2/'cache'/f'o{c}'
FC.mkdir(parents=True, exist_ok=True)
cfg = dict(eval_cut=c, train_cuts=tuple(range(c - 8, c - 4)), cap=f'{c}1231')
print('[build]', cfg, flush=True)
P = fr.clf_base(cfg, FC); P51, PO = fr.clf_os51(cfg, FC); Pins, ins_idx = fr.clf_ins(cfg, FC); ep = fr.ep_match(cfg, FC); cdf = fr.lag_cdf(cfg)
for t in cfg['train_cuts']:
    if not (FC/f'feat_c{t}.parquet').exists(): fr.cut_features(t, cfg['cap'], cfg, FC, P, P51, PO, Pins, ins_idx, ep, cdf)
if not (FC/f'feat_c{c}.parquet').exists(): fr.cut_features(c, None, cfg, FC, P, P51, PO, Pins, ins_idx, ep, cdf)
del P, P51, PO, Pins
for t in cfg['train_cuts']: fr.base_app_panel(t, cfg['cap'], FC)
fr.base_app_panel(c, None, FC)
print('[build] done', c, round(fr.elapsed()), flush=True)
