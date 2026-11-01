# C12-T2 s4c: 최종 제안 묶음 시험 빌드용 변형(build_test\ 안에만). 원고는 읽기만.
# F1 = 필수 제안(73 상속 정의+연결 민감도, 143 '각각', 395 'skill', 410 '동명 복원') + 그림 1 출처 문구(figs_fix)
# F2 = F1 + E-C11T2-07(73행 괄호에 병합) + E-C11T3-01(표 13 판 표기)
# 직전 시도의 P73/PC73은 공개번호 연결 하한을 47.0(이중 반올림)으로 적었으므로 46.9로 고친다(원값 0.469499).
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T2\build_test"
src = open(R + r"\manuscript_v3.md", encoding="utf-8").read()
E = {
 "F73": ("한국 패밀리 문서는 KR 라벨을 상속한다.", "한국 패밀리가 있는 해외 문서는 그 KR 문서의 해설서 단독 라벨을 상속한다(공개번호 연결이면 46.9–53.5%)."),
 "F73C": ("한국 패밀리 문서는 KR 라벨을 상속한다.", "한국 패밀리가 있는 해외 문서는 그 KR 문서의 해설서 단독 라벨을 상속한다(공개 시점 무관, 상속분의 22%는 더 늦게 공개된 KR 문서; 공개번호 연결이면 46.9–53.5%)."),
 "P143": ("현행 빈티지, 빼도 T1 skill 변화 0.0022 이하)", "현행 빈티지, 각각 빼도 T1 skill 변화 0.0022 이하)"),
 "P395": ("T4 4개월 전향 채점은 0.487 [0.431, 0.535]였다", "T4 4개월 전향 채점은 skill 0.487 [0.431, 0.535]였다"),
 "P410": ("N6(미등록 포함) +0.006–0.010", "N6(미등록 포함 동명 복원) +0.006–0.010"),
 "C11T3_01": ("Reliance on Science(Marx & Fuegi, CC BY-NC 4.0, 비상업), OpenAlex, KPSS 특허 가치 자료(Kogan·Papanikolaou·Seru·Stoffman)", "Reliance on Science(Marx & Fuegi, 2026-09-25판, CC BY-NC 4.0, 비상업), OpenAlex, KPSS 특허 가치 자료(Kogan·Papanikolaou·Seru·Stoffman, 2026-09-09판)"),
}
for k, (a, b) in E.items(): assert src.count(a) == 1, (k, src.count(a))
V = {"F1": ["F73", "P143", "P395", "P410"], "F2": ["F73C", "P143", "P395", "P410", "C11T3_01"]}
J = json.load(open(os.path.join(T, "variants.json"), encoding="utf-8"))
for v, keys in V.items():
    s = src
    for k in keys:
        a, b = E[k]; assert s.count(a) == 1, k; s = s.replace(a, b)
    open(os.path.join(T, f"ms_{v}.md"), "w", encoding="utf-8").write(s)
    J["variants"][v] = dict(edits=keys, figs="figs_fix", d_chars=len(s) - len(src), per_edit={k: len(E[k][1]) - len(E[k][0]) for k in keys})
    print(v, J["variants"][v])
for k, (a, b) in E.items(): J["edits"][k] = dict(old=a, new=b, d=len(b) - len(a))
json.dump(J, open(os.path.join(T, "variants.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
