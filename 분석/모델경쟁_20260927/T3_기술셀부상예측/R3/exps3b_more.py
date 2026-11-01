# exps3b.py 가 exec 로 읽는 추가 실험 정의
ROS2 = ['ros_sci', 'ros_ppp']
MORE = {
 'E38': ('RoS 셀 과학 의존도 5피처 → 음이항 성분에만', feats(ROS, comps=('nb',)), 'E33 분해'),
 'E39': ('RoS 셀 과학 의존도 5피처 → XGB 분류 성분에만', feats(ROS, comps=('xgbc',)), 'E33 분해'),
 'E40': ('RoS 핵심 2피처(논문인용 특허 비중, 특허당 논문 인용) → 트리 2성분', feats(ROS2, comps=('xgbc', 'nb')), '단변량 편상관 최강 2개(+0.086 9/9, +0.057 8/9)'),
 'E41': ('RoS 5피처 + 과학×모멘텀 상호작용 2피처(과학편차×해외모멘텀, 과학편차×KR모멘텀) → 트리 2성분', feats(ROS + ['ix_sci_fx', 'ix_sci_kr'], comps=('xgbc', 'nb')), '과학 의존 셀에서 해외 신호가 더 잘 전달되는가'),
 'E42': ('조합 A(과학→특허): RoS 5피처 + OpenAlex 셀 4피처 → 트리 2성분', feats(ROS + OAC, comps=('xgbc', 'nb')), '실측 인용 + 셀 논문 성장'),
}

# ---- E40 채택 후 (챔피언 E40 = E19 + RoS 2피처 → 트리 2성분) ----
import json as _json
_E19 = _json.load(open(os.path.join(HERE, 'champion_A.json'), encoding='utf-8'))['cfg']
def from_e19(f):
    return lambda cfg: f(copy.deepcopy(_E19))
def seeds2(cfg):
    c2 = copy.deepcopy(cfg)
    for c in c2['components']:
        if c['type'] in ('xgbc', 'nb'): c['seeds'] = 2
    return c2
MORE.update({
 'E41': ('과학×모멘텀 상호작용 2피처(RoS 과학편차×해외모멘텀, ×KR모멘텀) → 트리 2성분 (E40 위)', feats(['ix_sci_fx', 'ix_sci_kr'], comps=('xgbc', 'nb')), ''),
 'E42': ('조합 A 확장: OpenAlex 셀 논문 4피처 → 트리 2성분 (E40 위, 실측 인용 + 셀 논문 성장)', feats(OAC, comps=('xgbc', 'nb')), ''),
 'E43': ('RoS 핵심 2피처를 로지스틱에도 추가 (E40 위)', feats(ROS2, comps=('logit',)), ''),
 'E44': ('KOSIS 품목편 생산 2년 성장·생산능력 3년 성장 2피처(단변량 9/9 양) → 트리 2성분 (E40 위)', feats(['ki_gprod2', 'cu_cap_g3'], comps=('xgbc', 'nb')), 'E36·E37 축소판'),
 'R19s2': ('강건성 INFO: E19 설정을 트리 성분 2시드 배깅으로', from_e19(seeds2), 'E40 채택이 시드 우연인지 확인(짝 R40s2)'),
 'R40s2': ('강건성 INFO: E40 설정을 트리 성분 2시드 배깅으로', seeds2, 'R19s2 와 비교'),
})
