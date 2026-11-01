# 2단계 감사: (0) v2lib 로 원래 v23 재현(동등성 확인) → (a) 출원활동 수정만 (b) 권리자속성 수정만 (c) 전부 = v28
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, v2lib, owner_model_lib as oml, run
from common import *
from v2lib import *


def old_attr(d):
    oa = pd.read_parquet('cache/owner_attr.parquet').rename(columns={'nm': 'cur', 'natl_grp': 'o_natl', 'chaebol': 'o_chaebol', 'holding': 'o_holding', 'ip_entity': 'o_ip'})
    return d.merge(oa, on='cur', how='left')
EXTRA_FN['old_attr'] = old_attr
OLD_ATTR = ['o_natl', 'o_chaebol', 'o_holding', 'o_ip']
which = sys.argv[1:] or ['equiv', 'a', 'b', 'c']
tr = load(range(2006, 2012), dict(BASE, extra=['old_attr'])); va = load(VAL_L, dict(BASE, extra=['old_attr']))
if 'equiv' in which:
    cfg = dict(pat_feats=run.F12, own_mean=oml.AGG_MEAN, own_level=oml.OWN_LEVEL, own_attr=OLD_ATTR)
    ap = fit_apply(cfg, range(2006, 2012), VAL_L, tr, va); evaluate('a_equiv', ap, write=False)
if 'a' in which:
    cfg = dict(own_attr=OLD_ATTR)
    ap = fit_apply(cfg, range(2006, 2012), VAL_L, tr, va); evaluate('a_actfix', ap, ref='v23')
if 'b' in which:
    cfg = dict(pat_feats=run.F12, own_mean=oml.AGG_MEAN, own_level=oml.OWN_LEVEL)
    ap = fit_apply(cfg, range(2006, 2012), VAL_L, tr, va); evaluate('a_attrfix', ap, ref='v23')
if 'c' in which:
    ap = fit_apply({}, range(2006, 2012), VAL_L, tr, va)
    evaluate('v28', ap, ref='v23', kept='Y',
             change='[2단계 감사 수정] v23 재적합: (F1·F2) 권리자 출원활동을 kr_patents 전체출원(L 이전 공개분)·현재명칭 이름공간으로 재구성해 registry-등록분 기반 own_app*/own_years 와 이름불일치 k_* 를 교체, (F3) 권리자속성에서 검증기 양도인 기반 사후 추가 그룹명(제일모직·노루 등)과 스냅샷 의존 holding 플래그 제거',
             hyp='감사: 등록분만 세는 출원활동은 L 이후 등록 여부(미래)로 조건화되고, registry 이름은 2026 현재명칭이라 kr_patents 이름과 어긋나며, 그룹 목록에 검증 양성 상위 양도인이 들어가 있었다 → 제거 후 정직한 검증 수치 재측정')
