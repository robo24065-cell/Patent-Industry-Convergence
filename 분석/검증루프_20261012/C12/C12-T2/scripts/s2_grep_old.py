# C12-T2 s2: 바뀐 숫자의 옛값이 원고 다른 곳·facts·그림 CSV·make_diagrams에 남았는지 전수 검색
import sys, glob, os, json, re
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
MS = R + r"\manuscript_v3.md"
ms = open(MS, encoding="utf-8").read().split("\n")
sub_end = next(i for i, l in enumerate(ms) if re.match(r"^# \[(부록|편집 메모)", l))  # 0-based index of first appendix line
targets = {
 # 블록: [옛 문자열들]
 "13/244/407 AUC 범위": ["0.002~0.004", "공개 대안 셋"],
 "68 US FREE-IPC": ["51.0%"],
 "69 JP FREE-IPC": ["48.2%", "48.2–53.5", "48.2-53.5"],
 "340 TXT 범위": ["29.6–31.1", "29.6-31.1", "TXT 30.5%", "TXT 29.6%", "TXT 31.1%"],
 "341 보존율·Δ": ["+0.0239", "+0.0255", "25–29%", "71–75%", "hyb_free 25%", "0.7613", "0.7628"],
 "342/357 RCA": ["0.571", "−0.026", "-0.026", "0.157, 0.664", "+0.100 ["],
 "648 UB": ["49.3%", "55.2%", "C-AGREE-UB"],
 "168 ρ 범위": ["0.20–0.36", "0.20-0.36"],
 "355/636 T3 MG": ["+0.0020 [", "−0.0002, +0.0041", "7/9 (같다)", "0.7947"],
 "482/547 C-LAND": ["32/16/5", "0.360", "28/12/3", "0.199", "RTA>1 32", "RTA>1 28"],
 "90/429/507 IO표 명칭": ["ECOS 산업연관표", "ECOS(한국은행 산업연관표", "산업연관표·물가 (ECOS)", "산업연관표·물가(ECOS)", "한국은행 산업연관표 2010"],
 "143 시점": ["−0.002 이하", "0.005 이하"],
 "C-T3 kr_rebuild": ["+0.0084", "0.0100 ["],
}
files = [MS, R + r"\facts\facts.csv", R + r"\facts\facts_supplement.csv", R + r"\facts\facts_supplement_R12.csv", R + r"\figs\make_diagrams.py", R + r"\figs\make_data_figs.py"] + sorted(glob.glob(R + r"\figs\*.csv")) + [R + r"\figs_list.md"]
res = []
for blk, olds in targets.items():
    for f in files:
        try: L = open(f, encoding="utf-8-sig").read().split("\n")
        except Exception as e: continue
        for i, l in enumerate(L):
            for o in olds:
                if o in l:
                    where = os.path.basename(f)
                    if f == MS: where = "원고(제출 범위)" if i < sub_end else "원고(부록·메모)"
                    key = l.split(",")[0] if f.endswith(".csv") else ""
                    res.append(dict(block=blk, old=o, file=where, line=i + 1, key=key[:40], text=l[:200]))
print("제출 범위 끝(0-based 첫 부록 행):", sub_end + 1)
for r in res: print(r["block"], "|", r["old"], "|", r["file"], r["line"], r["key"], "|", r["text"][:150])
json.dump(dict(sub_end_line=sub_end + 1, hits=res), open(sys.argv[1], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
