# -*- coding: utf-8 -*-
"""동결 명세 작성: final_spec_r6.json (챔피언 = cache/champ_state.json, dev 기준값 = experiments.csv)."""
import json, hashlib
import pandas as pd
from n9_lib import *

st = json.load(open(f"{CACHE}/champ_state.json", encoding="utf-8"))
ex = pd.read_csv(f"{OUT}/experiments.csv", encoding="utf-8-sig").set_index("exp_id")
path = []
for e in ex.index:
    if e.startswith("C") and ex.at[e, "decision"] == "ACCEPT":
        path.append(e)
sha = lambda f: hashlib.sha256(open(os.path.join(OUT, f), "rb").read()).hexdigest()
spec = dict(
    track="N9 (J1) 정부R&D 과제의 50개 산업 특허 성과 조기 예측",
    frozen="2026-10-04",
    source_note="NTIS Open API 활용(비상업·내부 분석)",
    unit="NTIS 과제고유번호 × 과제연도 t (보안과제 제외)",
    outcome="그 과제 성과로 등록된 제공범위(50 KSIC) KR 특허 중 출원연도 ∈ [t, t+1] >= 1",
    main_variant="보고 지연판(이력 피처 출원연도 <= t-2) — 파일럿 결과를 본 뒤의 사후 결정(IDEAS_R5 권고), 원판 병기",
    champion=st["tag"], path="R05(파일럿 M0+M1+M2 지연판) -> " + " -> ".join(path),
    model=dict(model="xgb", params=XGB_PARAMS, neg_frac=NEG_FRAC, neg_seed=0, seeds=[0]),
    feats=st["feats"],
    txt_clf=dict(script="n9_textclf.py", vectorizer="char_wb 2-3gram TF-IDF, 2007–2013 과제 텍스트로 적합 후 고정(cache/txt_vec.pkl)",
                 clf="SGDClassifier(log_loss, alpha=1e-5, max_iter=20, random_state=0)", train="t <= v-2, 학습행은 과제연도 묶음 5겹 OOF"),
    references=dict(
        M0_L2=dict(desc="주 기준선: 파일럿 M0, 이력 지연판", feats=M0_L2),
        M0_orig=dict(desc="파일럿 M0 원판(이력 출원 <= t-1, 정보 1년 더 씀)", feats=M0_ORIG),
        strong_L2=dict(desc="강한 기준선 M0_L2+M2_L2(제공 50산업 성과 이력)", feats=M0_L2 + M2_L2),
        strong_orig=dict(desc="강한 기준선 원판 M0+M2", feats=M0_ORIG + M2_ORIG),
        pilot_L2=dict(desc="파일럿 설계 모형 M0+M1+M2 지연판(IDEAS_R5 사전 고정안)", feats=M0_L2 + M1 + M2_L2),
        pilot_orig=dict(desc="파일럿 설계 모형 M0+M1+M2 원판", feats=M0_ORIG + M1 + M2_ORIG)),
    dev_reference=dict(origins=DEV_ORIGINS, auc_champ=[float(ex.at[st["tag"], f"auc_{v}"]) for v in DEV_ORIGINS],
                       auc_M0_L2=[float(ex.at["R04", f"auc_{v}"]) for v in DEV_ORIGINS]),
    final=dict(F=F, test_t=FINAL_T, train_t=[2010, 2020], outcome_window="출원 2022-01-01~2023-12-31",
               metrics="AUC, skill vs M0_L2(주)·M0_orig·strong_L2, 상위 1/5/10% 적중률, 상위 100건, 상위 10% 재현율, 수행기관(정규화 기관명) 군집 부트스트랩 1,000회 95% CI",
               secondary="원판 챔피언, 파일럿 설계 모형(지연판·원판), 로지스틱 챔피언, 수행주체 하위집단"),
    code_sha256={f: sha(f) for f in ["n9_lib.py", "n9_build.py", "n9_text.py", "n9_textclf.py", "final_run_r6.py"]},
    artifact_sha256={"cache/txt_vec.pkl": sha("cache/txt_vec.pkl")},
)
json.dump(spec, open(f"{OUT}/final_spec_r6.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(spec["champion"], spec["path"], len(spec["feats"]))
