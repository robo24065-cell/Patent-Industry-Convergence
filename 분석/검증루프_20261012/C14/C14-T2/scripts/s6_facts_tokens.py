# C14-T2 s6: facts.csv·facts_supplement*.csv 행과 v15 부록 출처표(같은 ID) 숫자 토큰 전수 대조 → cache/s6_token_diff.csv
import sys, re, csv, json
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C14\C14-T2"
ms = open(R + r"\manuscript_v3.md", encoding="utf-8").read().split("\n")
app = {}
start = next(i for i, l in enumerate(ms) if l.startswith("# [부록"))
end = next(i for i, l in enumerate(ms) if l.startswith("# [편집"))
for i in range(start, end):
    l = ms[i]
    if not l.startswith("|") or l.startswith("|---"): continue
    c = [x.strip() for x in l.strip().strip("|").split("|")]
    if len(c) < 4 or c[0] in ("ID", "id"): continue
    key = re.sub(r"\s*\(.*?\)\s*$", "", c[0]).strip()
    app.setdefault(key, []).append(dict(line=i + 1, sect=c[1], val=" | ".join(c[2:-1]), src=c[-1]))
num = re.compile(r"[−\-+]?\d[\d,]*\.?\d*")
def toks(s):
    out = []
    for m in num.findall(str(s)):
        t = m.replace("−", "-").replace(",", "")
        if t in ("-", "+"): continue
        out.append(t.lstrip("+"))
    return out
rows = []
for fn in ["facts.csv", "facts_supplement.csv", "facts_supplement_R12.csv"]:
    raw = list(csv.reader(open(R + "\\facts\\" + fn, encoding="utf-8-sig")))
    hdr = raw[0]; fixed = []
    for r in raw[1:]:
        if len(r) > len(hdr): r = r[:len(hdr) - 1] + [" / ".join(r[len(hdr) - 1:])]  # 필드 수 초과 행(R12 2행): 넘친 칸을 note로 합침
        fixed.append(r + [""] * (len(hdr) - len(r)))
    df = pd.DataFrame(fixed, columns=hdr).fillna("")
    for ri, r in df.iterrows():
        fid = r["id"].strip()
        a = app.get(fid)
        frag = r.get("정확한 문장 조각", ""); cond = r.get("조건·경고", "")
        if not a:
            rows.append(dict(file=fn, row=ri + 2, id=fid, in_appendix=False, app_line="", facts_frag=frag[:300], v15_val="", only_in_facts="", only_in_v15="", cond=cond[:200]))
            continue
        v = a[0]["val"]
        tf, tv = toks(frag), toks(v)
        of = [x for x in tf if x not in tv]; ov = [x for x in tv if x not in tf]
        rows.append(dict(file=fn, row=ri + 2, id=fid, in_appendix=True, app_line=a[0]["line"], facts_frag=frag[:300], v15_val=v[:400],
                         only_in_facts=" ".join(of), only_in_v15=" ".join(ov), cond=cond[:200]))
T = pd.DataFrame(rows)
T.to_csv(W + r"\cache\s6_token_diff.csv", index=False, encoding="utf-8-sig")
print("facts rows", len(T), "in appendix", int(T.in_appendix.sum()), "with facts-only tokens", int((T.only_in_facts != "").sum()))
print("appendix ids", len(app), "appendix ids not in facts", len(set(app) - set(T.id)))
json.dump(sorted(set(app) - set(T.id)), open(W + r"\cache\s6_app_ids_not_in_facts.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60)
print(T[(T.only_in_facts != "")][["file", "row", "id", "app_line", "only_in_facts", "only_in_v15"]].to_string())
