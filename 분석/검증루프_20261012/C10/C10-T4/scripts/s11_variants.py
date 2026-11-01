# C10-T4 s11: 제안 수정의 쪽수 영향 테스트용 원고 변형 생성(이 폴더 build_test\ 안에서만)
import os, sys
sys.stdout.reconfigure(encoding="utf-8")
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C10\C10-T4\build_test"
src = open(os.path.join(T, "ms_V0.md"), encoding="utf-8").read()
A0 = "| 분류·보정 | 통계분류포털 KSIC 해설서, IPC-KSIC 연계표(2013, 대조군), WIPO IP 통계(공개건수 보정) |"
A1 = "| 분류·보정 | 통계분류포털 KSIC 해설서, IPC-KSIC 연계표(2013, 대조군), KSIC 신구 연계표(국가데이터처, 공공데이터포털 15159305, 공공누리 제4유형), WIPO IP 통계(공개건수 보정) |"
B0 = "Reliance on Science(Marx & Fuegi, CC BY-NC 4.0, 비상업)"
B1 = "Reliance on Science(Marx & Fuegi 2020·2022, CC BY-NC 4.0, 비상업)"
R0 = "- Marx, M., & Fuegi, A. (2020). Reliance on science: Worldwide front-page patent citations to scientific articles. *Strategic Management Journal*, 41(9), 1572–1594.\n"
R1 = R0 + "- Marx, M., & Fuegi, A. (2022). Reliance on science by inventors: Hybrid extraction of in-text patent-to-article citations. *Journal of Economics & Management Strategy*, 31(2), 369–392.\n"
C0 = "| 기업 | 국민연금공단 가입 사업장 내역(공공데이터포털), "
C1 = "| 기업 | 출원인-법인 연결표(2026-07), 국민연금공단 가입 사업장 내역(공공데이터포털), "
for s in (A0, B0, R0, C0):
    assert src.count(s) == 1, s
V = {
    "V1": src.replace(A0, A1),
    "V2": src.replace(A0, A1).replace(B0, B1).replace(R0, R1),
    "V3": src.replace(A0, A1).replace(B0, B1).replace(R0, R1).replace(C0, C1),
}
for k, t in V.items():
    open(os.path.join(T, f"ms_{k}.md"), "w", encoding="utf-8").write(t)
    print(k, len(t) - len(src))
