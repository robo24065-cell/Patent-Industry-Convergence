# C10-T4 s15: 줄 수 상쇄 변형(V5–V8) — 표 13·참고문헌 추가분을 같은 쪽의 짧은 꼬리 줄로 상쇄
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\build_test"
src = open(os.path.join(T, "ms_V0.md"), encoding="utf-8").read()
A0 = "| 분류·보정 | 통계분류포털 KSIC 해설서, IPC-KSIC 연계표(2013, 대조군), WIPO IP 통계(공개건수 보정) |"
A1 = "| 분류·보정 | 통계분류포털 KSIC 해설서, IPC-KSIC 연계표(2013, 대조군), KSIC 신구 연계표(국가데이터처, 공공누리 제4유형), WIPO IP 통계(공개건수 보정) |"
N0 = "NTIS Open API 활용(국가과학기술지식정보서비스 과제·성과검색, 2026-09-29 수집; 비상업, 제3자 제공 금지)"
N1 = "NTIS Open API 활용(과제·성과검색, 2026-09-29 수집; 비상업, 제3자 제공 금지)"
B0 = "Reliance on Science(Marx & Fuegi, CC BY-NC 4.0, 비상업)"
B1 = "Reliance on Science(Marx & Fuegi 2020·2022, CC BY-NC 4.0, 비상업)"
R0 = "- Marx, M., & Fuegi, A. (2020). Reliance on science: Worldwide front-page patent citations to scientific articles. *Strategic Management Journal*, 41(9), 1572–1594.\n"
R1 = R0 + "- Marx, M., & Fuegi, A. (2022). Reliance on science by inventors: Hybrid extraction of in-text patent-to-article citations. *Journal of Economics & Management Strategy*, 31(2), 369–392.\n"
D0 = "지식재산처·한국특허전략개발원(KISTA). 15,141,159행. 공유 금지, 집계값만 수록"
D1 = "지식재산처·한국특허전략개발원(KISTA). 공유 금지, 집계값만 수록"
K0 = "KOSIS(통계청 광업·제조업조사(10인 이상, 명목), 광공업생산지수)"
K1 = "KOSIS(통계청(현 국가데이터처) 광업·제조업조사(10인 이상, 명목), 광공업생산지수)"
for s in (A0, N0, B0, R0, D0, K0):
    assert src.count(s) == 1, s
V = {
    "V5": src.replace(A0, A1).replace(N0, N1),
    "V6": src.replace(A0, A1).replace(N0, N1).replace(B0, B1).replace(R0, R1),
    "V7": src.replace(A0, A1).replace(N0, N1).replace(B0, B1).replace(R0, R1).replace(D0, D1),
    "V8": src.replace(A0, A1).replace(N0, N1).replace(B0, B1).replace(R0, R1).replace(D0, D1).replace(K0, K1),
}
for k, t in V.items():
    open(os.path.join(T, f"ms_{k}.md"), "w", encoding="utf-8").write(t)
    print(k, len(t) - len(src))
