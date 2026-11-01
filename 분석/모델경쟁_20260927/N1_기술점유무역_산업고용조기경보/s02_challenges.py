# -*- coding: utf-8 -*-
"""2단계: 챔피언/도전자 순차 실험(피처 추가·교체). 사용법: python s02_challenges.py <batch>
각 도전은 현재 챔피언(R2/champion.json)에 변경 1개를 적용 → 규칙(Δρ≥0.003 & 승률≥60%)으로 ACCEPT/ROLLBACK."""
import sys
from engine import *

KEYS = ("alpha", "train_tg", "gap", "hl", "aug", "fe", "xf", "bag", "model")
def run_cfg(F, cfg):
    hl = cfg.get("hl")
    wf = None if not hl else (lambda t, v: 0.5 ** ((v - t) / hl))
    return preds(F, alpha=cfg.get("alpha", 5.0), train_tg=cfg.get("train_tg"), gap=cfg.get("gap", 3), wfun=wf, aug=cfg.get("aug"), fe=cfg.get("fe"),
                 xf=cfg.get("xf"), bag=cfg.get("bag", False), model=MODELS.get(cfg.get("model")))
def _xgb():
    from xgboost import XGBRegressor
    return XGBRegressor(max_depth=2, n_estimators=150, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, nthread=2, random_state=0)
def _xgb_rank():
    from sklearn.pipeline import make_pipeline
    return None
MODELS = {"xgb": _xgb}

def challenge(eid, change, fn, note="", **kw):
    st = champ_state(); sC = pd.Series({int(k): v for k, v in st["rho"].items()})
    F2, extra = fn(list(st["F"]))
    cfg = {k: st.get(k) for k in KEYS if st.get(k) is not None}; cfg.update(extra)
    o = run_cfg(F2, cfg)
    acc, s = log_exp(eid, st["id"], change, o, champ=sC, note=note)
    if acc:
        save_champ({"id": eid, "F": F2, **cfg, "rho": s.round(5).to_dict()})
    return acc

add = lambda cols: (lambda F: (F + cols, {}))
cfgc = lambda **kw: (lambda F: (F, kw))
def drop(cols):
    return lambda F: ([c for c in F if c not in cols], {})
def swapm(mp):
    return lambda F: ([mp.get(c, c) for c in F], {})
def swap(old, new):
    return lambda F: ([new if c == old else c for c in F], {})

BATCH = {
 "patent1": [
  ("E02", "+dffam_sh_Q3(KR 출원 중 해외패밀리 보유 비중 3년 변화, 시차3)", add(["dffam_sh_Q3"]), "family 스냅샷 → 시차3로 보수화. 단일 ρ −0.23(0/10 양수)"),
  ("E03", "+ddom_sh_Q2(KR 출원 중 국내우선권 비중 변화=기술 국산화, 시차2)", add(["ddom_sh_Q2"]), "우선권은 출원 시 확정(스냅샷 아님)"),
  ("E04", "+grant_sh_Q2(품질: 시차2 창 출원 중 t0까지 등록된 비율)", add(["grant_sh_Q2"]), "등록일 ≤ t0 조건으로 원점 시점 정보"),
  ("E05", "+dkrcn_lr_L2(KR/CN 특허 상대규모 이동, 시차2)", add(["dkrcn_lr_L2"]), "개선후보 (a) 경쟁국 격차"),
  ("E06", "+lead_frgn_L2(해외 모멘텀 − KR 모멘텀: 국가 간 선행 격차, 시차2)", add(["lead_frgn_L2"]), "개선후보 (a) T3식 국가 간 선행"),
 ],
 "ext1": [
  ("E07", "dffam_sh_Q3 → dffam_sh_Q2(시차3→2 교체)", swap("dffam_sh_Q3", "dffam_sh_Q2"), "시차 민감도(스냅샷 위험 증가 쪽)"),
  ("E08", "+g3_est(KOSIS 사업체수 3년 성장, t0-3→t0)", add(["g3_est"]), "외부: 광업제조업조사 사업체수. 단일 ρ 0.29(8/8)"),
  ("E09", "+pi_g3(KOSIS 광공업생산지수 3자리 3년 성장)", add(["pi_g3"]), "외부: 생산지수 3자리(50/50 연계)"),
  ("E10", "+shd_g3(내수출하지수 3년 성장, 2자리·C26은 3자리)", add(["shd_g3"]), "외부: 내수·수출 출하지수"),
  ("E11", "+xg3(한국 수출 3년 성장, Comtrade HS6→KSIC 1/m)", add(["xg3"]), "파일럿 M4 재도전(1/m 가중으로 재정의)"),
  ("E12", "+dtsi(무역특화지수 (X−M)/(X+M) 3년 변화)", add(["dtsi"]), "외부: 한국 수출·수입 결합"),
  ("E13", "+ims(KOSIS 업종별 수입점유비 2자리 수준)", add(["ims"]), "외부: 수입점유비(2010~, 2008·09 결측→0.5)"),
 ],
 "mix1": [
  ("E14", "Ridge α 5→10", cfgc(alpha=10.0), "모형 하이퍼파라미터"),
  ("E15", "Ridge α →2", cfgc(alpha=2.0), "모형 하이퍼파라미터(반대 방향)"),
  ("E16", "+ix_cn_dsh(대중 수입점유 × KR 특허점유변화 상호작용)", add(["ix_cn_dsh"]), "트랙 가설: 기술점유가 중국 노출 피해를 완충하는가"),
  ("E17", "+io_lab(산업연관표 피용자보수/총산출, 2010표)", add(["io_lab"]), "외부: 산업연관표 구조. 단일 ρ 0.126(10/10)"),
  ("E18", "+io_upm(산업연관표 중간투입 중 수입품 비중)", add(["io_upm"]), "외부: 공급망 수입의존(예시1 개념을 고용 예측 피처로)"),
  ("E19", "학습 타깃 = 종사자+사업체 3년성장 순위평균(평가는 종사자)", cfgc(train_tg="y_emp_h3+y_est_h3"), "다중 타깃 잡음 완화"),
  ("E20", "학습 타깃 = 종사자+출하 3년성장 순위평균(평가는 종사자)", cfgc(train_tg="y_emp_h3+y_shp_h3"), "다중 타깃 잡음 완화"),
  ("E21", "최근 원점 가중(반감기 4년)", cfgc(hl=4), "구조 변화 대응"),
  ("E22", "특허 시차2 KR 모멘텀·점유변화 → 공개일≤t0 조건판(P2)", swapm({"mom_KR_L2": "mom_KR_P2", "dsh_KR_L2": "dsh_KR_P2"}), "누수 점검(공개 지연)"),
  ("E23", "+cninv_mom_L2(CN 발명특허(실용신안 제외) 모멘텀, 시차2)", add(["cninv_mom_L2"]), "중국 기술 추격 신호"),
  ("E24", "+g1_est(KOSIS 사업체수 1년 성장)", add(["g1_est"]), "외부: 광업제조업조사"),
 ],
 "patent2": [
  ("E25", "+ap_ui(KR 출원 중 대학·연구기관 출원 비중, 시차2)", add(["ap_ui"]), "특허 출원인 구성(출원 시 기재명, 규칙 분류). 단일 ρ 0.221(9/10)"),
  ("E26", "+ap_smemom(상위5 출원인 제외 기업 출원 모멘텀, 시차2)", add(["ap_smemom"]), "중소·중견 기술활동. 단일 ρ 0.182(10/10)"),
  ("E27", "+rg_stock_g(등록원부 존속 특허 스톡 3년 성장, 사건일≤t0)", add(["rg_stock_g"]), "외부: 수집 등록원부(설정·소멸 사건)"),
  ("E28", "+rg_extr(등록원부 연차료불납 소멸 비율, 3년 창)", add(["rg_extr"]), "외부: 수집 등록원부. 포기 신호"),
  ("E29", "Ridge α →1", cfgc(alpha=1.0), "하이퍼파라미터 추가 탐색"),
  ("E30", "+ix_cn_ffam(대중 수입점유 × 해외패밀리 비중 변화)", add(["ix_cn_ffam"]), "상호작용: 중국 노출 산업에서 기술 국제화(해외 이전) 신호"),
 ],
 "mix2": [
  ("E31", "+hist_y(산업 고유 추세: 결과가 t0 이전에 끝난 과거 원점들의 종사자 성장 순위 평균)", add(["hist_y"]), "과거 레이블만 사용(t0' ≤ t0−3). 2011은 1개 원점, 2018은 7개"),
  ("E32", "학습 보강: t0=v−2 원점의 2년 성장 레이블 추가", cfgc(aug=[["y_emp_h2", 2]]), "최신 정보 반영 시도(결과 ≤ v)"),
 ],
 "mix3": [
  ("E33", "산업 고정효과 보정(학습 잔차 KSIC 평균, 수축 n/(n+2), λ=0.5)", cfgc(fe=[0.5, 2]), "지속 오차 산업(비철금속 등) 보정 시도"),
  ("E34", "−mom_KR_L2(계수 음수·공선성, 제거)", drop(["mom_KR_L2"]), "단순화"),
  ("E35", "+dom_sh_Q2(KR 출원 중 국내 우선권 비중 수준, 시차2)", add(["dom_sh_Q2"]), "기술 자립도 수준. 챔피언 잔차 상관 +0.18(10/10)"),
  ("E36", "+dsh_CN_L2(5개국 중 CN 특허 점유 변화, 시차2)", add(["dsh_CN_L2"]), "중국 기술 추격"),
  ("E37", "+sh_EP_L2(5개국 중 EP 특허 점유 수준, 시차2)", add(["sh_EP_L2"]), "유럽 기술 비중(고부가·틈새 산업 대리)"),
  ("E38", "+mw_g3(한국 전세계 수입 3년 성장, HS6→KSIC)", add(["mw_g3"]), "외부: 무역(내수 수요 대리)"),
  ("E39", "+io_inv(산업연관표 고정자본형성/총수요: 투자재 산업)", add(["io_inv"]), "외부: 산업연관표 수요구조"),
 ],
 "model2": [
  ("E40", "피처·타깃 변환: 순위 → 정규점수(norm.ppf)", cfgc(xf="normal"), "꼬리 산업 강조"),
  ("E41", "원점 배깅(학습 원점 1개씩 제외한 Ridge 평균)", cfgc(bag=True), "원점 잡음 완화(학습 원점 ≥3일 때)"),
  ("E42", "모형 교체: XGBoost(깊이2, 150트리, lr .05) 순위 회귀", cfgc(model="xgb"), "비선형 상호작용(표본 작아 과적합 위험)"),
 ],
 "leak1": [
  ("E43", "dffam_sh_Q2 → ddom_sh_Q2(스냅샷 없는 성분: 외국 우선권 비중 변화만)", swap("dffam_sh_Q2", "ddom_sh_Q2"), "누수 점검: family 스냅샷 제거판"),
  ("E44", "dffam_sh_Q2 → dffam_sh_Q2 + ddom_sh_Q3 대체 없이 dffam_sh_Q3 로 되돌림(보수판)", swap("dffam_sh_Q2", "dffam_sh_Q3"), "누수 점검: 30개월 국내단계 진입 여유 확보판"),
 ],
 "leak2": [
  ("E45", "dffam_sh_Q2 → dffam_pit_Q2(원점 시점 해외패밀리: 외국우선권 or 첫 해외공개연도 ≤ t0)", swap("dffam_sh_Q2", "dffam_pit_Q2"), "누수 제거판. 채택되면 스냅샷 위험 해소"),
 ],
 "patent3": [
  ("E46", "+grant_sh_Q2 재도전(t0까지 등록 비율 = 특허 품질)", add(["grant_sh_Q2"]), "E04(구 챔피언) 롤백 후 새 챔피언에서 재도전, 잔차상관 +0.15"),
  ("E47", "mom_KR_L2+dsh_KR_L2 → kr_md(두 순위 합 1개 지수)", lambda F: ([c for c in F if c not in ("mom_KR_L2", "dsh_KR_L2")] + ["kr_md"], {}), "공선성 완화·모수 축소"),
  ("E48", "frgn_mom+kab_mom → glob_idx(해외 모멘텀+한국발 해외출원 순위 합)", lambda F: ([c for c in F if c not in ("frgn_mom", "kab_mom")] + ["glob_idx"], {}), "모수 축소"),
 ],
 "transfer1": [
  ("E49", "+tsi(무역특화지수 (X−M)/(X+M) 3년 합 수준)", add(["tsi"]), "모듈 D에서 채택(+0.038)된 피처 이식"),
  ("E50", "+pi_g1(생산지수 3자리 1년 성장)", add(["pi_g1"]), "모듈 D에서 채택된 피처 이식"),
 ],
}
if __name__ == "__main__":
    for eid, ch, fn, note in BATCH[sys.argv[1]]:
        challenge(eid, ch, fn, note)
    print("현재 챔피언", champ_state()["id"], champ_state()["F"])
