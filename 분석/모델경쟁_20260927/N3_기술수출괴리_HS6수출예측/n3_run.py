# -*- coding: utf-8 -*-
"""N3 실험 실행기. 사용: python n3_run.py X|M exp_id [exp_id ...]
각 실험 = 현재 챔피언에 변경 1개. 판정은 n3_lib.record (PROTOCOL_R2 §3)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from n3_lib import *

FLOW = sys.argv[1]
D = load_panel(FLOW)
B1 = get_b1(D, FLOW)
own = OWN[FLOW]

X5_H = ["x5_KR", "x5_CN", "x5_JP", "x5_US", "x5_DE"]
X5_CHG = ["x5c_KR", "x5c_CN", "x5_tot_m"]
GAP = ["gap_kr", "gap_kr_lr", "gap_cn", "gap_jp", "gap_kr5", "gap_kr_h"]
XK = ["xk_KR", "xk_CN", "xk_JP", "xk_US", "xk_DE"]
KS_L = ["ks_l_ship", "ks_l_emp", "ks_va_ratio", "ks_ship_per_emp", "ks_cap_int"]
KS_M = ["ks_m_ship", "ks_m_emp", "ks_m_va", "ks_m_est", "ks_m_cap"]
KS_X = ["ks_expint", "ks_exp_minus_ship_m"]
IO_D = ["io_dn_m", "io_up_m", "io_dn_cov"]
IO_P = ["io_dn_pout", "io_dn_pKR", "io_dn_pcov"]
IO_S = ["io_exp_ratio"]
CN_K = ["cnk_sh", "cnk_chg"]
CN_H = ["im_cn_sh", "im_cn_chg"]
Q_OUT = ["q_KRout", "q_CNout", "q_JPout", "q_USout", "q_EUout"]
MISC = ["share_in_k", "n_ksic"]


def add(extra):
    return lambda f, s: ([*f, *[e for e in extra if e not in f]], s)


def drop(rm):
    return lambda f, s: ([c for c in f if c not in rm], s)


def sp(**kw):
    return lambda f, s: (f, {**s, **kw})


def par(**kw):
    return lambda f, s: (f, {**s, "params": {**s.get("params", {}), **kw}})


def wswap(f, s):
    return ([("w_" + c if c in PAT_LVL else c) for c in f], s)


EXPS = {
    # id: (change 설명, 변환 함수(f, spec) -> (f, spec), note)
    "e0": ("B1 품목 자기이력 XGB(고정 기준선)", None, "BASE"),
    "e1": ("출발 챔피언 G2-e4 = B1 + 특허 수준·점유(pl_*, p_domshare, p_outshare, p_krshare5)", None, "START"),
    # (a) 괴리·5개국 수출 (외부 무역데이터)
    "e2": ("(a) 5개국 수출 HS6 점유(KR·CN·JP·US·DE, 수집연도 ≤ t0) 추가", add(X5_H), ""),
    "e3": ("(a) 기술-수출 괴리(한국 기원 해외출원 점유 − 5개국 수출 점유, KSIC·품목) 추가", add(GAP), ""),
    "e4": ("(a) 5개국 수출 점유 변화(KR·CN, 5개국 합 성장) 추가", add(X5_CHG), ""),
    "e5": ("(a) 괴리 6종 → 핵심 2종(gap_kr_h, gap_kr_lr)으로 축소", drop(["gap_kr", "gap_cn", "gap_jp", "gap_kr5"]), "1차 정의(이미 포함된 피처 추가=무변경)를 재정의"),
    # (c) KOSIS
    "e6": ("(c) KOSIS 광업제조업조사 KSIC 수준(출하·종사자·부가가치율·노동생산성·자본집약) 추가", add(KS_L), ""),
    "e7": ("(c) KOSIS 3년 변화(출하·종사자·부가가치·사업체·자산) 추가", add(KS_M), ""),
    "e8": ("(c) 생산기반 대비 수출(수출/출하 로그비, 수출−출하 모멘텀) 추가", add(KS_X), ""),
    # (b) IO
    "e9": ("(b) 산업연관표 전방(수요)·후방(공급) 산업 출하 모멘텀 추가", add(IO_D), ""),
    "e10": ("(b) 산업연관표 하류 산업 특허 포지션(해외점유·KR 수준) 추가", add(IO_P), ""),
    # (d) 대중 수입
    "e11": ("(d) KSIC 대중 수입점유·3년 변화 추가", add(CN_K), ""),
    "e12": ("(d) 품목 대중 수입비중·변화 추가", add(CN_H), ""),
    # (e) 결합 방식
    "e13": ("(e) 다중 KSIC 품목 특허 피처 결합: 단순평균 → KOSIS 출하 가중", wswap, ""),
    # 특허 추가
    "e14": ("기원국별 해외출원 점유(q_KRout·CNout·JPout·USout·EUout) 추가", add(Q_OUT), ""),
    "e15": ("품목의 KSIC 내 비중·연결 KSIC 수 추가", add(MISC), ""),
    # (f) 모형
    "e16": ("(f) max_depth 3→4", par(max_depth=4), ""),
    "e17": ("(f) max_depth 3→2, n_estimators 300→500", par(max_depth=2, n_estimators=500), ""),
    "e18": ("(f) n_estimators 300→600, learning_rate 0.03→0.02", par(n_estimators=600, learning_rate=0.02), ""),
    "e19": ("(f) 단조 제약 p_outshare 음(−)", sp(mono={"p_outshare": -1}), ""),
    "e20": ("(f) 특허 선형 LAD 1단계 + XGB 잔차", sp(model="lin_then_xgb", lin_feats=["p_outshare", "p_domshare", "p_krshare5", "pl_KRout", "pl_FOR"]), ""),
    "e21": ("(f) 최근 원점 가중(연 0.85)", sp(recency=0.85), ""),
    "e22": ("(f) 학습 원점 2007 이후로 제한(초기 5개국 수출 결측 구간 제외)", sp(min_train_t0=2007), ""),
    "e23": ("(f) min_child_weight 1→20 (희소 KSIC 과적합 억제)", par(min_child_weight=20), ""),
    "e24": ("(f) colsample_bytree 0.8→0.5", par(colsample_bytree=0.5), ""),
    "e25": ("(a) 품목 수준 타국 괴리(CN·JP·US·EU 기술점유 − 수출점유) 추가", add(["gap_cn_h", "gap_jp_h", "gap_us_h", "gap_eu_h"]), ""),
    "e26": ("(a) 괴리 변화(한국·중국 기술점유 3년 변화 − 수출점유 3년 변화) 추가", add(["gapm_kr_h", "gapm_cn_h"]), ""),
    "e27": ("(a) 품목 한국 수출점유의 KSIC 대비 상대위치 추가", add(["x5_KR_rel"]), ""),
    "e28": ("(a) KSIC 수준 5개국 수출점유 원값(xk_*) 추가", add(XK), ""),
    "e29": ("특허 수준(pl_* 10개) 제거 — 괴리·점유만 유지", drop([c for c in PAT_LVL if c.startswith("pl_")]), ""),
    "e30": ("(c) KOSIS 출하 3년 변화 1개만 추가", add(["ks_m_ship"]), ""),
    "e31": ("(f) n_estimators 300→600, learning_rate 0.03→0.02 (챔피언 e3 위 재시험)", par(n_estimators=600, learning_rate=0.02), ""),
    "e32": ("(a) 중국 수출점유 수준·변화(x5_CN, x5c_CN) 추가", add(["x5_CN", "x5c_CN"]), ""),
    "e33": ("자기이력 모멘텀 원점 내 상대화(mom3·mom1·k_mom3·imp_mom − 원점 중앙값) 추가", add(["mom3_rel", "mom1_rel", "k_mom3_rel", "imp_mom_rel"]), ""),
    "e34": ("장기 자기이력(직전 창 모멘텀 t0-8..t0-6→t0-5..t0-3, 고점 대비 낙폭) 추가", add(["mom_prev", "dd"]), ""),
    "e35": ("고점 대비 낙폭(dd) 제거(직전 창 모멘텀만 유지)", drop(["dd"]), "1차 정의(dd 추가)는 e34에 이미 포함돼 무변경 → 재정의"),
    "e36": ("(f) n_estimators 300→600, learning_rate 0.03→0.02 (챔피언 e34 위 재시험)", par(n_estimators=600, learning_rate=0.02), ""),
    "e37": ("(c) KOSIS 출하 3년 변화 1개 추가 (챔피언 e34 위 재시험)", add(["ks_m_ship"]), ""),
    "e38": ("(f) max_depth 3→4 (챔피언 e34 위 재시험)", par(max_depth=4), ""),
    "e39": ("(a) 품목 수준 타국 괴리(CN·JP·US·EU) 추가 (e34 위 재시험)", add(["gap_cn_h", "gap_jp_h", "gap_us_h", "gap_eu_h"]), ""),
    "e40": ("직전 창 모멘텀 원점 내 상대화(mom_prev_rel) 추가", add(["mom_prev_rel"]), ""),
    "e41": ("(b) 산업연관표 전방·후방 출하 모멘텀 추가 (e34 위 재시험)", add(IO_D), ""),
    "e42": ("(d) KSIC 대중 수입점유·변화 추가 (e34 위 재시험)", add(CN_K), ""),
    "e43": ("(a) 5개국 수출 수집연도 3년 간격 → 매년(ye = t0, 변화는 정확히 3년) — R2N3 보충 수집 반영", sp(variant="_annual"), ""),
    "e44": ("(f) 목적함수 절대오차 → pseudo-Huber", par(objective="reg:pseudohubererror", huber_slope=0.5), ""),
    "e45": ("약한 괴리 피처 제거(gap_kr5, gap_jp)", drop(["gap_kr5", "gap_jp"]), ""),
    "e46": ("(a) 괴리 절대값(|gap_kr_h|·|gap_kr_lr|·|gap_cn|, 역U자 관계 반영) 추가", add(["gap_kr_h_abs", "gap_kr_lr_abs", "gap_cn_abs"]), ""),
    "e47": ("(f) 동일 피처 depth2·500트리 XGB와 50:50 블렌드", sp(model="blend", feats_b=None), "feats_b=챔피언 피처"),
    "e48": ("(f) reg_lambda 1→10", par(reg_lambda=10), ""),
    "e49": ("(f) min_child_weight 1→5", par(min_child_weight=5), ""),
    "e50": ("(a) 괴리 6종을 원점 내 백분위로 교체(중국 특허 급증에 따른 분포 이동 제거)", lambda f, s: ([c + "_rk" if c.startswith("gap_") else c for c in f], s), ""),
    "e51": ("(a) 괴리 6종의 원점 내 백분위 추가(원값 유지)", lambda f, s: (f + [c + "_rk" for c in f if c.startswith("gap_")], s), ""),
    "e52": ("특허 점유 3종(p_outshare·p_krshare5·p_domshare) 원점 내 백분위 추가", add(["p_outshare_rk", "p_krshare5_rk", "p_domshare_rk"]), ""),
    # G1 재시험용(챔피언 G1-e34 위)
    "g3": ("(a) 기술-수출 괴리 6종 추가 (G1 챔피언 위 재시험)", add(GAP), ""),
    "g7": ("(c) KOSIS 3년 변화 추가 (G1 챔피언 위 재시험)", add(KS_M), ""),
    "g14": ("기원국별 해외출원 점유 추가 (G1 챔피언 위 재시험)", add(Q_OUT), ""),
    "g25": ("(a) 품목 수준 5개국 괴리(KR·CN·JP·US·EU) 추가", add(["gap_kr_h", "gap_cn_h", "gap_jp_h", "gap_us_h", "gap_eu_h"]), ""),
    "g9": ("(b) 산업연관표 전방·후방 출하 모멘텀 추가 (G1 재시험)", add(IO_D), ""),
}


def run(eid):
    change, fn, note = EXPS[eid]
    if eid == "e0":
        R = B1; feats, spec = own, {}
        return record(FLOW, eid, "-", change, feats, spec, R, B1, note="고정 기준선", force_decision="BASE")
    if eid == "e1":
        feats, spec = own + PAT_LVL, {}
        R = evaluate(D, feats, spec)
        return record(FLOW, eid, "e0", change, feats, spec, R, B1, note="출발 챔피언", force_decision="START")
    ch = champion(FLOW)
    feats, spec = fn(list(ch["feats"]), dict(ch["spec"]))
    if spec.get("model") == "blend" and spec.get("feats_b") is None:
        spec["feats_b"] = list(feats)
    DD = load_panel(FLOW, spec["variant"]) if spec.get("variant") else D
    assert (DD[["hs6", "t0"]].values == D[["hs6", "t0"]].values).all()
    feats = [c for c in feats if c in DD.columns]
    R = evaluate(DD, feats, spec)
    return record(FLOW, eid, ch["exp_id"], change, feats, spec, R, B1, note=note)


if __name__ == "__main__":
    for e in sys.argv[2:]:
        run(e)
