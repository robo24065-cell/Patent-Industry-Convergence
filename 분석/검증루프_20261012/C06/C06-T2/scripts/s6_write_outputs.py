# C06-T2 s6: docx_vs_md_diff.csv, fig_audit.csv 작성(앞 단계 cache·verify_0 결과를 근거로)
import csv, json, os, sys
sys.stdout.reconfigure(encoding='utf-8')
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C06\C06-T2"
C = os.path.join(OUT, 'cache')
s1 = json.load(open(os.path.join(C, 's1_summary.json'), encoding='utf-8'))
v0 = json.load(open(os.path.join(OUT, 'verify_0', 'v0_results.json'), encoding='utf-8'))

# ---------------- docx_vs_md_diff.csv ----------------
ATTR = {
 217: ('E-C05T1-01(반영 기록 5)', '낡음(이전 판 값)', 'v6 표 6 신설 IPC 행 비고 칸이 비어 있음 → 현재: 재분류 소급 셀(부상률 38%), 반도체 통합 시 0.403'),
 223: ('E-C05T1-02(반영 기록 5)', '낡음(이전 판 값)', "v6 '22셀은 IPC 2023 신설 분류' → 현재 '39셀은 2022–25년 신설 분류'"),
 275: ('E-C05T2-02(반영 기록 5)', '낡음(이전 판 값)', 'v6에 N7 기준일 확인 층 0.18 [0.10, 0.26] 문장 없음'),
 323: ('지면 조정(반영 기록 5: 3.6.2 3자리 업종 CI 생략)', '낡음(이전 판 문구, 판정 영향 없음)', 'v6에는 CI·KIND 4자리 문장이 있음. 현재판은 CI 삭제'),
 404: ('지면 조정(반영 기록 5: 표 12 선결 조건에 4.4-1·2 한계 해소 흡수)', '낡음(이전 판 문구)', ''),
 -1: ('지면 조정(반영 기록 5: 4.3 표 12 뒤 문단 삭제)', '낡음(이전 판 문구)', 'v6 docx에만 있는 문단(2단계는 ‘50개 풀 위 대체 시험’…)'),
 409: ('지면 조정(반영 기록 5: 4.4-1 수치 괄호를 3.6.2 참조로)', '낡음(이전 판 문구)', "v6 'NPS 특허 가중(−3.6%p)·가입자 300명 이상 기업(−4.0%p)·그룹 안 5자리 선택(−21.7%p)' → 현재 '…·대기업·…(3.6.2)'"),
 411: ('E-C05T3-01·E-C05T1-03(반영 기록 5) + 지면 조정(CN 37.8% 괄호 삭제)', '낡음(이전 판 값)', "v6 'N9 상위 1%와 N8 맨 위 목록은 강한 기준선 수준' — E-C05T3-01 이전 문구, H10 신설 소급 문장 없음"),
 412: ('E-C05T2-01·E-C05T3-04(반영 기록 5) + 4.4-4 압축', '낡음(이전 판 값)', "v6 '사후 목록이라 생존 편향이 있다'만 있고 연결률 18%→96%·N9 새 양성 0 없음"),
}
rows = []
for s in s1:
    rows.append({'비교': s['label'], 'md_sha1': s['md_sha1'][:8], 'docx_sha1': s['docx_sha1'][:8], '구분': '요약',
                 'md_줄': '', 'md_종류': '', 'md_문단(현재판 또는 빌드 원고)': f"기대 문단 {s['n_expected']} / docx 문단 {s['n_docx']} / 그림 {s['n_img_docx']}·{s['n_fig_md']}",
                 'docx_문단': f"차이 행 {s['n_diff_rows']} (유사도 {s['ratio']:.4f})", 'ERRATA_관련': '', '판정': '일치' if s['n_diff_rows'] == 0 else '차이 있음', '비고': f"제출 범위 = 원고 {s['cut_line']}행 '# [부록' 이전"})
for r in csv.DictReader(open(os.path.join(C, 's1_diffs_raw.csv'), encoding='utf-8-sig')):
    ln = int(r['md_line']) if r['md_line'] else -1
    a = ATTR.get(ln, ('(미분류)', '불일치', ''))
    rows.append({'비교': r['pair'], 'md_sha1': '', 'docx_sha1': '', '구분': r['op'], 'md_줄': r['md_line'], 'md_종류': r['md_kind'],
                 'md_문단(현재판 또는 빌드 원고)': r['md_text'], 'docx_문단': r['docx_text'], 'ERRATA_관련': a[0], '판정': a[1], '비고': a[2]})
for v in ('v7', 'v6'):
    x = v0['xml_' + v]
    rows.append({'비교': f'verify_0 XML 직접 파싱 {v}docx_vs_md_v3', 'md_sha1': '508b1c33', 'docx_sha1': '', '구분': '재확인', 'md_줄': ';'.join(map(str, x['md_lines_differing'])),
                 'md_종류': '', 'md_문단(현재판 또는 빌드 원고)': f"XML 문단 {x['n_xml']} = python-docx {x['n_python_docx']}: {x['xml_equals_python_docx']}",
                 'docx_문단': f"차이 블록 {x['n_diff_blocks_vs_md_v3']}", 'ERRATA_관련': '', '판정': '재확인 일치' , '비고': 'python-docx diff와 같은 원고 줄 집합'})
with open(os.path.join(OUT, 'docx_vs_md_diff.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

# ---------------- fig_audit.csv ----------------
F = []
def add(fig, num, item, src, recompute, png, text, verdict, note=''):
    F.append({'그림파일': fig, '원고_그림번호': num, '항목': item, '원천': src, '재계산·대조 결과': recompute,
              'PNG_표기': png, '본문·캡션 인용(현재 원고 줄)': text, '판정': verdict, '비고': note})
MCs = '(MC)=분석\\모델경쟁_20260927'
add('fig02', '그림 3', '9개 트랙 AUC 모형·기준선, 공식 skill [CI]', '(MC)\\각 트랙 final_result*.json(T1 R3·T3 R3·T4 R2·N4 r4·N5 r4·N6 r5·N7 r5·N8 r6·N9 r6)',
    'JSON 값 3자리 반올림이 CSV와 모두 같음(T1 skill CI 0.4536/0.4885→[0.454,0.488] 등). verify_0: 예측 파일 순위합 AUC로 T3(R3\\preds\\FINAL_R3_t0_2021 yb)·N4·N5·N6·N7·N8·N9 독립 재계산 7/7 일치(3자리)',
    'PNG 9행 숫자·속 빈 점(T4·N5 조건부) 모두 CSV와 같음', '17·53·205–216·233·259·267·275·283–284·289행 등 같은 값', '일치',
    'T1·T4는 예측 파일 없이 JSON 대조만(T1 쌍 단위 예측 미저장). 주의: VERIFY_R5\\preds\\RERUN_FINAL_t0_2021은 누수 수정 전 값(skill 0.4242=원고 0.424)')
add('fig02', '그림 3', '보수적 skill·기준·등급', 'R6 §3.2, 트랙 VERIFY 표(T1 R5 SBB 0.395, T3 R5 SB3 0.182, T4 R5 0.124, N4 R5 0.140, N5 R5 0.0002, N7 R6 0.251), N6·N8·N9 final JSON',
    'N6 0.1417·N8 0.3118·N9 0.3557은 JSON, N8·N9는 verify_0 예측 파일 재계산도 일치. 나머지는 VERIFY 표 값과 일치', '값·기준 문구 CSV와 같음', '116·127·130·216·240·283–284·289행', '일치')
add('fig03', '그림 8', '(a) 제공·공개 최강·FREE-IPC·TXT-FREE·연계표 기대값, Δ와 CI', '(MC)\\_shared_report\\label_origin_R10\\confirm_R12\\res_main.csv NPS5·DART5',
    'CSV = res_main 값 그대로(소수 4자리). 사사오입 %p: −0.6[−2.0,+1.0]·+7.3[+5.1,+9.7]·+11.5[+9.8,+13.1]·+28.7[+25.8,+31.7]; 샌드박스 재생성 CSV 바이트 동일',
    'PNG 막대 숫자 30.3/30.9/29.3/18.8/14.7, 62.9/55.5/53.6/34.2/23.1과 주석 CSV와 같음', '표 9(316–317행)·13·321–323·328·351–353행 같은 값', '일치',
    '특허 가중 −3.6[−7.6,−0.8]·+8.1[+5.9,+10.5], 3자리 +1.8/+1.2/+1.9, 사후 최강 +6.7(0.6286−0.5617)도 res_main에서 재현')
add('fig03', '그림 8', '(b) 순환 점검 KR·해외·TXT-FREE', '(MC)\\_shared_report\\label_origin_R10\\res_p1.csv(가족 다수결·기업평균)',
    'KR 26.98/해외 26.83, +0.15[−0.87,+1.13]; DART 61.37/62.07, −0.69[−2.34,+0.78] 일치', 'PNG 27.0/26.8/16.8, 61.4/62.1/31.9, 주석 ±2자리 일치', '326행 같은 값', '일치')
add('fig04', '그림 5', '(b)(c) KR 출원·KOSIS 출하·사업체 연성장(2008–09→2016–17)', '(MC)\\_shared_report\\fusion_advantage\\split34\\t5_cases.csv',
    'CSV = t5_cases 그대로. (a) n_KR로 LCD·OLED KR 성장 독립 재계산: 로그 −6.59%·+6.17%(복리면 −6.37%·+6.36%) → 원고 값은 로그 정의와 일치',
    'PNG 숫자 CSV와 같음. 단 x축 이름은 \'연평균 성장률 (%)\'로 \'로그\' 표기 없음', '9·194·253·376행(로그 성장률 명시)', '일치(표기 누락 경고)',
    '캡션 197행에 로그 정의가 없어 그림만 보면 복리로 읽힐 수 있음(−6.4% 대 −6.6%)')
add('fig04', '그림 5', '(a) C2620 LCD·OLED KR 출원 지수(2005=100)', 'split34\\fig_smallmultiples_long.csv', '지수 2005=100 확인, 개발 기간 실선·사후 점선', 'PNG 축·범례 CSV와 같음', '—', '일치')
add('fig05', '그림 4', '5개국×50산업 RTA(출원 2017–19, 동일가중)', '(MC)\\_shared_report\\fusion_advantage\\map5\\fig1_rta_map_P3.csv',
    'verify_0: 제공 메타(_shared\\meta 50 part, 출원 2017–19 발명특허 1,764,898건)에서 독립 재계산, 최대 절대 차 8.9e-16. KR>1 25, 1위 KR7·CN18·EP9·JP8·US8, KR÷1위<0.7 24개 재현',
    'PNG 머리 요약·셀 값·굵은 산업명 7개 CSV와 같음', '15·161–162·168·372행(화장품 1.72, OLED 1.68, 무인 항공기 1.61, 한의약품 0.11, 프린터 0.15)', '일치',
    '29150(산업용 오븐) KR 0.9991은 1.00으로 찍히지만 RTA≤1 열에 있음(경계 표기, 오류 아님)')
add('fig06', '(원고 미사용)', '기술×수출 사분면 48 KSIC', 'map5\\fig2_quadrant_tech_export.csv',
    '사분면 12/11/3/22, ρ 0.3562 재현(표 5와 같음)', 'PNG 주석에 ρ 0.356만 있고 정의 민감도(0.30–0.34) 없음. 샌드박스 재렌더와 픽셀 차 23,196(legacy 스크립트 변경 후 미재렌더)',
    '표 5(176–183행)는 그림 대신 사용', '낡음(미사용 그림, 제출본 밖)', '제출본에 들어가지 않으므로 재빌드 사유 아님')
add('fig07', '(원고 미사용)', 'T3 미래 전망 KSIC별 예측 부상 비율', '(MC)\\T3_기술셀부상예측\\R3\\future_forecast_r3_ksic.csv, future_forecast_r3.csv',
    '5,328셀·532셀, 상위 4개 21.9/19.6/19.5/18.8%, KR 0건 54.3% 재현. 신설 분류 셀은 39셀(H10B/K/N 12셀 포함)',
    "PNG 각주 '22셀은 IPC 2023 신설 서브클래스' — E-C05T1-02 반영 이전 문구(현재 원고 223행 '39셀은 2022–25년 신설')", '223행', '낡음(미사용 그림, 제출본 밖)',
    'figs_list.md 7행도 \'신설 22셀\'. 그림을 다시 쓰면 각주 교체 후 재렌더')
add('fig08', '그림 6', 'T1 대기실 지수 상위 15개 산업 비중', '(MC)\\T1_산업대기실\\R3\\future_forecast_r3.csv(기업명 열 제외)',
    '합 9,573.56, 위험쌍 1,544,097, 상위 5개 8.65/6.91/6.16/5.12/3.47% 합 30.30%, 비중=지수/합 차 4e-9', 'PNG 15행 숫자 CSV와 같음, 기업명 없음', '248·253·387행', '일치',
    '쌍 단위 미래 예측은 저장되지 않아(final_run_r3.py 366행 메모리 집계) 집계 수준 대조만 가능')
add('fig09', '그림 7', '(a) 상위 1·5·10% 적중률, 기저율', '(MC)\\N9_정부RnD과제_특허성과예측\\final_result_r6.json',
    'verify_0: final_preds_r6.parquet(75,077과제)에서 k=751·3,754·7,508로 독립 재계산 78.30/52.58/39.69 대 54.86/41.69/31.90, 기저율 6.97% 일치',
    'PNG 78.3/54.9, 52.6/41.7, 39.7/31.9, 기저율 6.97% 같음', '259·264·388행(상위 10% 재현율 56.9% 대 45.8%도 JSON 일치)', '일치')
add('fig09', '그림 7', '(b) 기준선 7종 상위 1%·상위 100건', 'final_result_r6.json + R6 §1.1(상위 100건 하드코딩 93/72/62/97/96)',
    'verify_0 재계산: 상위 1% 78.3/54.9/57.3/69.9/68.3/78.4/80.6, 상위100 93/72/62/80/89/97/96 → PNG·CSV 표기 값 모두 일치', 'PNG 상위100 표기 5개(파일럿 지연판·M0 원판은 빈칸)',
    '265행(80.6%·78.4%, 96·97건, 93건)', '일치(CSV 빈칸 2개는 E-C05T3-05 보류 중)', '빈칸은 값 누락일 뿐 틀린 값 아님')
add('fig10', '그림 2', '(a) T3 시간축, (b) 8단계 사례 숫자', 'T3 R3 FINAL_REPORT_R3.md(0.4275, 55건, 0.25–0.48%, 1e−16), T1 R3 FINAL_REPORT_R3.md(0.0, 9.0e-8), R6 §1.1·§2(위약·해시 6개)',
    '모든 숫자 원천 보고서 문구와 일치. 개발 0.428→최종 0.423(차 −0.005)', 'PNG 문구 CSV와 같음', '109·124–127행(표 4) 같은 값', '일치')
add('fig01', '그림 1', '세 다리·중심·대조군 숫자 문구', 'fig01_framework.csv 숫자 출처(res_fam_check 0.873/0.870/0.905, t0b 0.39997, R9·R10), 표 3',
    '87–90%, 그룹 40.0%, 5자리 7.1%, TXT 29.4%, TXT-FREE 40.4%, 해외 FREE-IPC 48–54%, KOSIS 50/50·16/50, HS6 48/50 — 현재 원고 13·31·60·79·88–89·134행과 일치',
    'PNG 문구 CSV와 같음(샌드박스 재렌더 픽셀 동일)', '82행 캡션', '일치(조건 누락 경고)',
    "대조군 상자에 R12 이후 주 비교 대상인 '공개 최강(최대 58.8%)'·KR FREE-IPC 55.5%가 없음(2.4·3.6.2와 비교 사다리 불일치). KPSS·OpenAlex는 최종 모형 제외(99행)이나 다리 ③에 표기")
add('전체', '그림 1–8', '제출본 내장 그림 = 현재 figs PNG', '보고서_초안_v7.docx word/media, figs\\*.png', 'docx 내장 8개 PNG 해시가 현재 figs 파일과 모두 같고 순서도 원고 자리표시와 같음(v5·v6도 동일). pdf 그림 8개',
    '', '원고 81·111·152·165·196·250·261·309행 자리표시', '일치')
add('전체', '그림 1–10', '샌드박스 재생성(원본 스크립트 사본, 출력만 C06-T2\\regen)', 'make_data_figs.py·make_diagrams.py 현재 원천',
    '10개 CSV 바이트 동일. 사용 그림 8개 PNG 픽셀 차 0. 미사용 fig06·fig07 PNG만 차이', '', '', '일치(사용 그림)')
add('전체', '—', 'C04 낡은 facts 8행(OFF-T3-TOP10·TOP100·OFF-N7-SB·LBL-CW-GROUP·DATA-REG-EVENTS·LINK-HS6-EXP·C-AGREE-KR·A-COVER) 전파 여부', 'figs\\fig*.csv 전수 문자열 검색',
    "26.0%·기준선 33%·122,974·18.7%·KSIC()·FREE-IPC 39.2·9,054/15.9% 없음. 41.9는 fig01 CSV 주석의 '연계표 배정 문서만 41.9%'(정확한 조건부 값)뿐. 0.124는 T4 보수적 값(정상)",
    'PNG에도 없음', '—', '일치(전파 없음)', "figs_list.md(원고 밖 설명 문서)는 '같은 발명 일치 89.7%·연계표 그룹 41.9%·신설 22셀'로 낡음")
with open(os.path.join(OUT, 'fig_audit.csv'), 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(F[0].keys())); w.writeheader(); w.writerows(F)
print(len(rows), len(F))
