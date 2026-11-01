# C18-T3 s7(C16-T4 s4_residual.py 경로만 변경): facts(원본/패치 사본) ↔ 현재 원고(v19 b316df8e) 숫자 토큰 재대조
# 주 지표(A): facts '정확한 문장 조각'의 숫자 토큰 중 원고 해당 위치(부록 같은 ID 행 값 칸)에 없는 것
# 보조 지표(B): 부록 값 칸의 숫자 토큰 중 facts(문장 조각+조건·경고) 어디에도 없는 것(facts 미동기)
# 부록에 ID가 없는 패치 대상은 원고 전체(편집 메모 제외)에서 찾는다.
import csv, re, sys, json, io, hashlib
sys.stdout.reconfigure(encoding='utf-8')
AN = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
R = AN + r"\보고서_20261012"; L = AN + r"\검증루프_20261012"; OUT = L + r"\C18\C18-T3"
ms = open(R + r"\manuscript_v3.md", encoding='utf-8').read().replace('\r\n', '\n').split('\n')
st = next(i for i, s in enumerate(ms) if s.startswith('# [부록')); en = next(i for i, s in enumerate(ms) if s.startswith('# [편집 메모'))
APP = {}
for i in range(st, en):
    s = ms[i]
    if s.startswith('| ') and not s.startswith('| ---'):
        c = [x.strip() for x in s.strip().strip('|').split(' | ')]
        if len(c) >= 4 and c[0] != 'id':
            key = re.sub(r'\s*\((보충|보충 R12)\)$', '', c[0])
            APP[key] = dict(line=i + 1, val=' | '.join(c[2:-1]))
FULL = '\n'.join(ms[:en])
NUM = re.compile(r'(?<![A-Za-z0-9_])[−\-+]?\d[\d,]*(?:\.\d+)?')
def toks(s):
    out = set()
    for m in NUM.finditer(s.replace('–', ' – ')):
        t = m.group(0).replace('−', '-').replace('+', '')
        t = t.replace(',', '') if re.fullmatch(r'-?\d{1,3}(,\d{3})+(\.\d+)?', t) else t.rstrip(',')
        t = t.lstrip('-') if t.startswith('-') else t   # 부호는 토큰 비교에서 무시(원고·facts 표기 혼용)
        out.add(t)
    return out
def load(p):
    rows = list(csv.reader(io.StringIO(open(p, encoding='utf-8-sig').read())))
    h = rows[0]; out = []
    for r in rows[1:]:
        if len(r) >= len(h): d = dict(zip(h, r[:len(h) - 1] + [','.join(r[len(h) - 1:])])); out.append(d)
    return out
def check(fdir, label):
    res = []
    for name in ['facts.csv', 'facts_supplement.csv', 'facts_supplement_R12.csv']:
        for r in load(fdir + '\\' + name):
            i = r['id']; frag = r.get('정확한 문장 조각', ''); cond = r.get('조건·경고', '')
            ft = toks(frag)
            if i in APP:
                at = toks(APP[i]['val'])
                missA = sorted(ft - at); missB = sorted(at - (ft | toks(cond)))
                res.append(dict(판=label, 파일=name, id=i, 대조위치=f"부록 {APP[i]['line']}행", A_facts에만=' '.join(missA), nA=len(missA), B_부록에만=' '.join(missB), nB=len(missB)))
            else:
                fullt = toks(FULL)
                missA = sorted(ft - fullt)
                used = len(ft & fullt) > 0 and len(missA) < len(ft)
                res.append(dict(판=label, 파일=name, id=i, 대조위치='원고 전체(부록 행 없음)' + ('' if used else ', 원고 미사용 추정'), A_facts에만=' '.join(missA), nA=len(missA), B_부록에만='', nB=0))
    return res
pre = check(R + r"\facts", '원본')
post = check(OUT + r"\facts_work", '패치 사본')
patched_ids = set(r['id'].split('→')[-1] for r in csv.DictReader(open(OUT + r"\facts_patch_v19.csv", encoding='utf-8-sig')))
def summ(rs):
    app = [r for r in rs if r['대조위치'].startswith('부록')]
    return dict(부록ID행=len(app), A불일치행=sum(1 for r in app if r['nA'] > 0), B미동기행=sum(1 for r in app if r['nB'] > 0),
                A또는B=sum(1 for r in app if r['nA'] > 0 or r['nB'] > 0))
S = dict(원본=summ(pre), 패치=summ(post))
# 잔여 목록: 패치 사본 기준, 부록 대조 행 중 A 또는 B가 남은 것 + 패치 대상 중 원고 전체 대조에서 A가 남은 것(원고 미사용 제외)
resid = [r for r in post if (r['대조위치'].startswith('부록') and (r['nA'] > 0 or r['nB'] > 0))
         or (r['id'] in patched_ids and not r['대조위치'].startswith('부록') and r['nA'] > 0 and '미사용' not in r['대조위치'])]
SHORT = {'MAP-KR-TOP', 'DATA-REGISTRY', 'LINK-T1-R3'}
for r in resid:
    r['패치대상'] = 'Y' if r['id'] in patched_ids else 'N'
    if r['대조위치'].startswith('부록'):
        r['분류'] = '부록 셀 축약(…)으로 facts가 더 완전 — 오류 아님' if r['id'] in SHORT else '불일치(조사 필요)'
    else:
        r['분류'] = '대조 위치 없음: facts 전용·보충 행 숫자(원고 미사용)'
with open(OUT + r"\facts_work\facts_residual_mismatch.csv", 'w', encoding='utf-8-sig', newline='') as f:
    fn = ['판', '파일', 'id', '패치대상', '분류', '대조위치', 'nA', 'A_facts에만', 'nB', 'B_부록에만']
    w = csv.DictWriter(f, fieldnames=fn, extrasaction='ignore'); w.writeheader(); w.writerows(resid)
with open(OUT + r"\facts_work\facts_recheck_all.csv", 'w', encoding='utf-8-sig', newline='') as f:
    fn = ['판', '파일', 'id', '대조위치', 'nA', 'A_facts에만', 'nB', 'B_부록에만']
    w = csv.DictWriter(f, fieldnames=fn, extrasaction='ignore'); w.writeheader(); w.writerows(pre + post)
app_missing = [k for k in APP if k not in set(r['id'] for r in post)]
S['부록ID_facts없음(패치 후)'] = app_missing
S['잔여행'] = len(resid); S['잔여_부록대조_불일치'] = sum(1 for r in resid if r['분류'].startswith('불일치')); S['잔여_부록축약'] = sum(1 for r in resid if r['분류'].startswith('부록 셀')); S['잔여_대조위치없음'] = sum(1 for r in resid if r['분류'].startswith('대조 위치')); S['잔여_A'] = sum(1 for r in resid if r['nA'] > 0); S['잔여_B만'] = sum(1 for r in resid if r['nA'] == 0 and r['nB'] > 0)
allpost=[r for r in post if not r['대조위치'].startswith('부록')]
S['참고_비부록행_전체'] = len(allpost); S['참고_비부록행_원고에없는숫자있음'] = sum(1 for r in allpost if r['nA']>0)
json.dump(S, open(OUT + r"\cache\s7_out.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(S, ensure_ascii=False))
for r in resid: print(r['id'], r['패치대상'], r['대조위치'], 'A:', r['A_facts에만'][:80], '| B:', r['B_부록에만'][:100])

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
