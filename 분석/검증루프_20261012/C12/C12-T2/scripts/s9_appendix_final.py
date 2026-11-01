# C12-T2 s9: s6 자동 대조 결과에 수동 확인(원천 위치를 직접 찾은 것)을 덧붙여 appendix_tokens.csv 작성
import sys, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T2"
d = pd.read_csv(W + r"\cache\appendix_tokens_raw.csv", encoding="utf-8-sig", dtype=str).fillna("")
MAN = {
 ("463", "229,514"): ("일치(인접 로그)", r"(MC)\_shared_report\label_origin_R10\confirm_R12\logs\r12_prep.log 'eval 229514'(출처 열 json에는 없음)"),
 ("478", "5"): ("구조 상수", "5개국"), ("539", "15"): ("구조 상수", "±15% 기준"), ("585", "50"): ("구조 상수", "50개 산업"),
 ("594", "50"): ("구조 상수", "50개 산업"), ("598", "2"): ("구조 상수", "2년 안"),
 ("491", "0.371"): ("일치(다른 파일)·출처 열 불완전", r"(MC)\_shared_report\label_validity\label_validity.md 23행 κ 0.371(출처 열 t0_txt_agree.json에는 κ 없음)"),
 ("546", "0.121"): ("일치(다른 파일)·출처 열 불완전", r"검증루프 C01\C01-T1\sync_sensitivity.csv CN 2011-2018 raw 0.1209"),
 ("546", "0.036"): ("일치(다른 파일)·출처 열 불완전", "같은 파일 A 0.0359"), ("546", "0.087"): ("일치(다른 파일)·출처 열 불완전", "같은 파일 B 0.087(p 0.0789→0.08)"),
 ("565", "22.4"): ("간접 일치", "C09-T2가 T3 예측 parquet에서 본문 같은 표기(기준선 동점 범위 22.4–35.5%) 원천 재계산 '정확'"),
 ("565", "35.5"): ("간접 일치", "같음"),
 ("618", "0.034"): ("일치(원천에서 계산)", "final_result_r6.json pos 108 / n 317,840 = 0.0340%"),
 ("619", "0.034"): ("일치(원천에서 계산)", "같음; 1.23/0.034 = 36배"),
 ("625", "1,312"): ("일치(원천에서 계산)", "s1_meta_o2019.json DART 8,264 × 0.1588 = 1,312.3"),
 ("625", "318"): ("부분(이중 반올림 가능)", r"C06\C06-T1\out\s3_meta.json ESS 318.5(1자리 저장) → 사사오입 319; 원값 미보존"),
 ("637", "71"): ("일치·출처 열 구경로", r"C10\C10-T3\res\s6_t3_summary.json reg.retention txt 0.286 → 1−0.286=71%(출처 열은 c5_R10)"),
 # 편집 메모(원천 경로 없음) 중 대조 가능한 것
 ("683", "0.6286"): ("일치", "confirm_R12\\res_main.csv DART5 PROV 0.6286"), ("683", "0.5617"): ("일치", "res_main.csv DART5 사후최강값 0.5617"),
 ("690", "0.0505"): ("일치", "res_main.csv DART5 PROV-BEST_lo"), ("690", "0.0965"): ("일치", "PROV-BEST_hi"), ("690", "0.1045"): ("일치", "PROV-BEST_w_hi"),
 ("678", "229,514"): ("일치(인접 로그)", "r12_prep.log eval 229514"), ("676", "39.2"): ("일치", "c1_label_agreement.csv KR free5 0.3922"),
 ("677", "39.2"): ("일치", "같음"), ("677", "100"): ("일치", "res_prep.json k_free_alt_eq_t1 1.0"),
 ("681", "33"): ("일치", "res_prep.json candidates 33개"), ("681", "22"): ("일치", "manuscript_v2_from_response.md 제출 범위 '공개 최강' 22회"),
 ("681", "0"): ("일치", "manuscript_v1_from_response.md 제출 범위 0회"), ("689", "13"): ("일치", "facts_supplement_R12.csv 13행(머리 제외)"),
}
verdict2, note2 = [], []
for _, r in d.iterrows():
    k = (r.v10_line, r.token)
    if k in MAN: v, n = MAN[k]
    elif r.verdict == "원천 경로 없음(메모 문장)": v, n = "편집 기록(대조 대상 아님)", "구조 숫자(절 번호·순서·'1건' 등)"
    elif r.verdict == "원천=DB(별도 질의)": v, n = "일치(DB 읽기 전용 재집계)", "registry_full.sqlite mode=ro: 13개 테이블 12,248,048행, patent 656,360 pay 3,520,838 cofpr 723,645 applicant 710,410 inventor 1,763,079"
    else: v, n = r.verdict, r.match
    verdict2.append(v); note2.append(n)
d["final_verdict"] = verdict2; d["final_note"] = note2
# A단계와 겹치는 토큰 표시
d.loc[d.a_stage.str.startswith("변경됨"), "final_verdict"] = "v11–v13에서 바뀜 → A단계(diff_blocks) 감사"
d.loc[d.v13_line == "", "final_verdict"] = "v11–v13에서 바뀜 → A단계(diff_blocks) 감사"
def cls(v):
    if v.startswith("v11–v13"): return "A(C12 A단계)"
    if v.startswith("일치(약함"): return "약한 일치(작은 정수·구조)"
    if v.startswith("일치") or v.startswith("간접"): return "A(C12 재계산)" if not v.startswith("간접") else "A-간접"
    if v.startswith("구조"): return "구조 상수"
    if v.startswith("편집"): return "편집 기록"
    return "확인 필요"
d["now_cls"] = d.final_verdict.map(cls)
cols = ["v10_line", "v13_line", "row_id", "token", "c09_cls", "now_cls", "final_verdict", "final_note", "match", "src_value", "src", "a_stage", "token_in_v13", "context"]
d[cols].to_csv(W + r"\appendix_tokens.csv", index=False, encoding="utf-8-sig")
print(len(d)); print(d.final_verdict.value_counts().to_string()); print(d.now_cls.value_counts().to_string())
print(d[d.final_verdict.str.startswith("불일치")][["v10_line", "row_id", "token"]].to_string())
