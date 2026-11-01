# C12-T2 s4b: 묶음 변형 PC = P(73·143·395·410 + 그림 1 출처) + E-C11T2-07(73행 괄호에 병합) + E-C11T3-01
import os, sys, json
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
T = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C12\C12-T2\build_test"
src = open(R + r"\manuscript_v3.md", encoding="utf-8").read()
J = json.load(open(os.path.join(T, "variants.json"), encoding="utf-8"))
E = {k: (v["old"], v["new"]) for k, v in J["edits"].items()}
E["PC73"] = ("한국 패밀리 문서는 KR 라벨을 상속한다.", "한국 패밀리가 있는 해외 문서는 그 KR 문서의 해설서 단독 라벨을 상속한다(공개 시점 무관, 상속분의 22%는 더 늦게 공개된 KR 문서; 공개번호 연결이면 47.0–53.5%).")
s = src
for k in ["PC73", "P143", "P395", "P410", "C11T3_01"]:
    a, b = E[k]; assert s.count(a) == 1, k; s = s.replace(a, b)
open(os.path.join(T, "ms_PC.md"), "w", encoding="utf-8").write(s)
J["edits"]["PC73"] = dict(old=E["PC73"][0], new=E["PC73"][1], d=len(E["PC73"][1]) - len(E["PC73"][0]))
J["variants"]["PC"] = dict(edits=["PC73", "P143", "P395", "P410", "C11T3_01"], figs="figs_fix", d_chars=len(s) - len(src),
                           per_edit={k: J["edits"][k]["d"] for k in ["PC73", "P143", "P395", "P410", "C11T3_01"]})
json.dump(J, open(os.path.join(T, "variants.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(J["variants"]["PC"])
