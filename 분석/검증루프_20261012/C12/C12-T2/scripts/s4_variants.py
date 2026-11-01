# C12-T2 s4: 시험 빌드용 원고 변형 생성(build_test\ 안에만). 원고는 읽기만.
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T2\build_test"
src = open(R + r"\manuscript_v3.md", encoding="utf-8").read()
E = {
 # 내 제안(P)
 "P73": ("한국 패밀리 문서는 KR 라벨을 상속한다.", "한국 패밀리가 있는 해외 문서는 그 KR 문서의 해설서 단독 라벨을 상속한다(공개번호 연결이면 47.0–53.5%)."),
 "P143": ("현행 빈티지, 빼도 T1 skill 변화 0.0022 이하)", "현행 빈티지, 각각 빼도 T1 skill 변화 0.0022 이하)"),
 "P395": ("T4 4개월 전향 채점은 0.487 [0.431, 0.535]였다", "T4 4개월 전향 채점은 skill 0.487 [0.431, 0.535]였다"),
 "P410": ("N6(미등록 포함) +0.006–0.010", "N6(미등록 포함 동명 복원) +0.006–0.010"),
 # 미반영 C11 선택 항목(ERRATA 문구)
 "C11T2_07": ("한국 패밀리 문서는 KR 라벨을 상속한다.", "한국 패밀리 문서는 KR 라벨을 상속한다. 상속은 공개 시점과 무관하게 패밀리 전체에서 가져온다(상속 라벨의 22%는 해외 문서보다 늦게 공개된 KR 문서 기준)."),
 "C11T1_03": ("(연결률: 마지막 활동 2005년 이전 18%, 2021년 이후 96%)", "(연결률: 마지막 활동 2005년 이전 18%, 2021년 이후 96%)(2013년 이후 발급 코드는 99% 이상이 사업자번호를 갖고 그 이전 코드는 17–42%뿐이어서 이전 코드의 번호는 사후 보완분으로 보인다; 2013년 이후 코드만 보면 N4 0.19, N7 0.25)"),
 "C11T3_01": ("Reliance on Science(Marx & Fuegi, CC BY-NC 4.0, 비상업), OpenAlex, KPSS 특허 가치 자료(Kogan·Papanikolaou·Seru·Stoffman)", "Reliance on Science(Marx & Fuegi, 2026-09-25판, CC BY-NC 4.0, 비상업), OpenAlex, KPSS 특허 가치 자료(Kogan·Papanikolaou·Seru·Stoffman, 2026-09-09판)"),
 "C11T3_02": ("Reliance on Science 과학 의존도를 붙였다.", "Reliance on Science 과학 의존도를 붙였다. RoS는 제공 family 필드(확장 패밀리)의 US 등록번호로 잇는다(연결 US 특허의 약 3할은 같은 출원인의 다른 발명; 초대형 패밀리 연결을 빼도 skill 변화 +0.002)."),
 "C168tie": ("한국 RTA 순위상관은 0.19–0.37이며", "한국 RTA 순위상관은 0.19–0.37(최빈 동률 처리에 따라 상한 0.37–0.48)이며"),
}
for k, (a, b) in E.items():
    assert src.count(a) == 1, (k, src.count(a))
def apply(keys):
    s = src
    for k in keys:
        a, b = E[k]; assert s.count(a) == 1, k; s = s.replace(a, b)
    return s
V = {
 "N0": ([], "figs"),
 "P": (["P73", "P143", "P395", "P410"], "figs_fix"),
 "C1": (["C11T2_07"], "figs"),
 "C2": (["C11T1_03"], "figs"),
 "C3": (["C11T3_01"], "figs"),
 "C4": (["C11T3_02"], "figs"),
 "C5": (["C168tie"], "figs"),
}
meta = {}
for k, (keys, fd) in V.items():
    t = apply(keys)
    open(os.path.join(T, f"ms_{k}.md"), "w", encoding="utf-8").write(t)
    meta[k] = dict(edits=keys, figs=fd, d_chars=len(t) - len(src),
                   per_edit={e: len(E[e][1]) - len(E[e][0]) for e in keys})
    print(k, meta[k])
json.dump(dict(edits={k: dict(old=a, new=b, d=len(b) - len(a)) for k, (a, b) in E.items()}, variants=meta),
          open(os.path.join(T, "variants.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
