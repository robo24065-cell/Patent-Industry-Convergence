# C18-T3 s7 생성기: C16-T4 s4_residual.py(읽기만)를 경로만 바꿔 s7_residual.py로 쓰고, 형식 검사를 덧붙인다.
import sys
sys.stdout.reconfigure(encoding="utf-8")
L = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012"
s = open(L + r"\C16\C16-T4\scripts\s4_residual.py", encoding="utf-8").read()
rep = [
 ("# C16-T4 s4: facts(원본/패치 사본) ↔ 현재 원고(4cbf3d55) 숫자 토큰 재대조",
  "# C18-T3 s7(C16-T4 s4_residual.py 경로만 변경): facts(원본/패치 사본) ↔ 현재 원고(v19 b316df8e) 숫자 토큰 재대조"),
 ('OUT = L + r"\\C16\\C16-T4"', 'OUT = L + r"\\C18\\C18-T3"'),
 ("facts_patch_consolidated.csv", "facts_patch_v19.csv"),
 ('open(OUT + r"\\facts_residual_mismatch.csv"', 'open(OUT + r"\\facts_work\\facts_residual_mismatch.csv"'),
 ('open(OUT + r"\\facts_recheck_all.csv"', 'open(OUT + r"\\facts_work\\facts_recheck_all.csv"'),
 ('open(OUT + r"\\s4_out.json"', 'open(OUT + r"\\cache\\s7_out.json"'),
 ("import csv, re, sys, json, io", "import csv, re, sys, json, io, hashlib"),
]
for a, b in rep:
    assert a in s, a
    s = s.replace(a, b, 1) if a.startswith("import") else s.replace(a, b)
EXTRA = r'''
# ==== C18 추가: 형식 검사(BOM·CRLF·패치 대상 밖 행 바이트 동일) ====
assert hashlib.sha1(open(R + r"\manuscript_v3.md", "rb").read()).hexdigest().startswith("b316df8e")
FD = R + r"\facts"; WK = OUT + r"\facts_work"; fmt = {}
PATCH = list(csv.DictReader(open(OUT + r"\facts_patch_v19.csv", encoding="utf-8-sig")))
for name in ["facts.csv", "facts_supplement.csv", "facts_supplement_R12.csv"]:
    a = open(FD + "\\" + name, "rb").read(); b = open(WK + "\\" + name, "rb").read()
    def rows(x):
        t = x.decode("utf-8-sig"); rr = list(csv.reader(io.StringIO(t.replace("\r\n", "\n"))))
        return {r[0]: r for r in rr[1:] if r}
    ra, rb = rows(a), rows(b)
    pat = set(r["id"].split("→")[-1] for r in PATCH if r["파일"] == name) | set(r["id"].split("→")[0] for r in PATCH if r["파일"] == name)
    changed = sorted(k for k in ra if k in rb and ra[k] != rb[k]); added = sorted(k for k in rb if k not in ra); removed = sorted(k for k in ra if k not in rb)
    # 바이트 수준: 패치 대상 밖 행의 원문 줄이 사본에 그대로 있는지
    la = set(a.decode("utf-8-sig").split("\r\n")); lb = set(b.decode("utf-8-sig").split("\r\n"))
    fmt[name] = dict(bom_src=a.startswith(b"\xef\xbb\xbf"), bom_work=b.startswith(b"\xef\xbb\xbf"),
                     crlf_src=a.count(b"\r\n"), lf_only_src=a.count(b"\n") - a.count(b"\r\n"),
                     crlf_work=b.count(b"\r\n"), lf_only_work=b.count(b"\n") - b.count(b"\r\n"),
                     rows_src=len(ra), rows_work=len(rb), changed=len(changed), added=len(added), removed=removed,
                     changed_or_added_not_in_patch=[k for k in changed + added if k not in pat],
                     src_lines_missing_in_work=len(la - lb))
S["형식"] = fmt
json.dump(S, open(OUT + r"\cache\s7_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False, indent=1))
'''
open(L + r"\C18\C18-T3\scripts\s7_residual.py", "w", encoding="utf-8").write(s + EXTRA)
print("ok")
