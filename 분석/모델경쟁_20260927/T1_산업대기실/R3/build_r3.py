# T1 R3 — 원점별 패널 보충 구축 (1라운드 final_run 함수 재사용, 진입창 h=4로 런타임 패치)
#   python R3/build_r3.py --c 2010     개발 원점: R2 캐시의 분류기(cap=c-12-31)를 읽어 학습 원점 c−4 패널만 새로 만든다(R3/cache/o<c>/)
#   python R3/build_r3.py --c 2015     보조 원점: 분류기부터 전부 R3/cache/o2015/에 새로 만든다(진입 2016–2019)
# 종료코드 3이면 같은 명령 재실행(캐시에서 이어감). 🔒 모든 레이블 진입창 끝 ≤2019.
import sys, pathlib, argparse
R3 = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(R3))
from r3 import *

ap = argparse.ArgumentParser(); ap.add_argument('--c', type=int, required=True); a = ap.parse_args(); c = a.c
cf = cfg_of(c)
lib.HOLDOUT_GUARD_MAX_ENTRY_YEAR = GUARD_DEV
_bp = lib.build_panel
fr.build_panel = lambda cut, h=5, cap=None, **kw: _bp(cut, H, cap=cap, **kw)   # 진입창 5→4(레이블 열만 영향, 피처 동일)
FC3 = fc_r3(c)
src = fc_r2(c) if (fc_r2(c)/'P_base.npy').exists() else FC3
print('[build r3]', cf, 'classifier src', src, flush=True)
P = fr.clf_base(cf, src); P51, PO = fr.clf_os51(cf, src); Pins, ins_idx = fr.clf_ins(cf, src); ep = fr.ep_match(cf, src); cdf = fr.lag_cdf(cf)
for t in list(cf['train_cuts']) + [c]:
    if feat_path(c, t) is None:
        fr.cut_features(t, None if t == c else cf['cap'], cf, FC3, P, P51, PO, Pins, ins_idx, ep, cdf)
del P, P51, PO, Pins
for t in list(cf['train_cuts']) + [c]:
    if feat_path(c, t, 'base_app') is None: fr.base_app_panel(t, None if t == c else cf['cap'], FC3)
print('[build r3] done', c, round(fr.elapsed()), f'peak {peak_gb():.2f}GB', flush=True)
