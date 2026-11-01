# -*- coding: utf-8 -*-
"""verify_0: 작업자 코드와 다른 경로(순수 파이썬 파서·sqlite3 직접 질의·다른 체크섬 구현·다른 정규화/유사도)로
(a) 체크섬 실패율·DART/권리목록 일치율, (d) 하위 층 비중과 '옛 이름 = 다른 코드' 반박 시험을 재현·반박 시도.
쓰기: 이 폴더(verify_0)만. 입력 읽기 전용."""
import os, re, json, sqlite3, unicodedata, random, collections
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
HERE = os.path.dirname(os.path.abspath(__file__))
CA = os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt")
def ro(p): return sqlite3.connect("file:" + p.replace("\\", "/") + "?mode=ro", uri=True)
OUT = {}
# --- 순수 파이썬 파서 ---
rows = []
with open(CA, "rb") as f:
    raw = f.read().decode("utf-8")
lines = raw.split("\n")
hdr = lines[0].split("¶")
for ln in lines[1:]:
    if not ln.strip():
        continue
    p = ln.split("¶")
    rows.append([x.strip() for x in p])
OUT["rows"] = len(rows)
digits = lambda s: "".join(ch for ch in s if ch.isdigit())
# 다른 구현: 법인번호 — 가중 벡터를 문자열로, 검증자리 = (10 - (sum % 10)) % 10 를 divmod로
JW = "121212121212"
def jcheck(n):
    if len(n) != 13: return None
    t = sum(int(a) * int(b) for a, b in zip(n[:12], JW))
    return divmod(10 - divmod(t, 10)[1], 10)[1] == int(n[12])
BW = (1, 3, 7, 1, 3, 7, 1, 3, 5)
def bcheck(n):
    if len(n) != 10: return None
    t = 0
    for i, w in enumerate(BW):
        t += int(n[i]) * w
    t += (int(n[8]) * 5) // 10
    return (10 - t % 10) % 10 == int(n[9])
jn = [digits(r[3]) for r in rows]; bn = [digits(r[4]) for r in rows]
jr = [jcheck(x) for x in jn if x]; br = [bcheck(x) for x in bn if x]
OUT["a1_jur_nonblank"] = len(jr); OUT["a1_jur_fail"] = jr.count(False); OUT["a1_jur_fail_rate"] = jr.count(False) / len(jr)
OUT["a2_biz_nonblank"] = len(br); OUT["a2_biz_fail"] = br.count(False); OUT["a2_biz_fail_rate"] = br.count(False) / len(br)
iy = collections.Counter(); iyf = collections.Counter()
for r, j in zip(rows, jn):
    if j:
        y = r[0][1:5]; iy[y] += 1; iyf[y] += (jcheck(j) is False)
OUT["a1_fail_rate_le2024"] = sum(iyf[y] for y in iy if y <= "2024") / sum(iy[y] for y in iy if y <= "2024")
OUT["a1_fail_rate_2025"] = iyf["2025"] / iy["2025"]; OUT["a1_fail_rate_2026"] = iyf["2026"] / iy["2026"]
# --- DART 대조(sqlite3 직접) ---
d = ro(os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_company.sqlite"))
dart = [(digits(a or ""), digits(b or ""), n) for a, b, n in d.execute("select jurir_no, bizr_no, corp_name from company")]
j2b = {}
for r, j, b in zip(rows, jn, bn):
    if j and j not in j2b: j2b[j] = (b, r[1])
agree = tot = blank = found = 0
for j, b, n in dart:
    if len(j) != 13: continue
    if j in j2b:
        found += 1
        cb = j2b[j][0]
        if not cb: blank += 1
        elif len(b) == 10:
            tot += 1; agree += (cb == b)
OUT["a3_found"] = found; OUT["a3_ca_biz_blank"] = blank; OUT["a3_pairs"] = tot; OUT["a3_agree"] = agree; OUT["a3_agree_rate"] = agree / tot
OUT["a3_dart_jur_fail"] = sum(jcheck(j) is False for j, _, _ in dart if len(j) == 13)
# --- 권리목록 대조(sqlite 집계) ---
b = ro(os.path.join(BASE, r"수집파이프라인\business_rights.sqlite"))
rl = {}
for cd, bz, jr_ in b.execute("select applicant_cd, max(business_no), max(corp_rgstno) from right_list group by applicant_cd"):
    rl[cd] = (digits(bz or ""), digits(jr_ or ""))
cam = {r[0]: (bnn, jnn) for r, bnn, jnn in zip(rows, bn, jn)}
cmpb = [(cam[c][0] == v[0]) for c, v in rl.items() if c in cam and cam[c][0] and v[0]]
cmpj = [(cam[c][1] == v[1]) for c, v in rl.items() if c in cam and cam[c][1] and v[1]]
OUT["a4_biz_compared"] = len(cmpb); OUT["a4_biz_agree_rate"] = sum(cmpb) / len(cmpb)
OUT["a4_jur_compared"] = len(cmpj); OUT["a4_jur_agree_rate"] = sum(cmpj) / len(cmpj)
# --- (d) 다른 정규화·유사도: NFKC + 한글/영숫자만 + 법인형태 제거, 문자 bigram Dice ---
FORM = ["주식회사", "유한회사", "유한책임회사", "합자회사", "합명회사", "(주)", "㈜", "(유)", "株式会社", "coltd", "co.,ltd", "inc", "corporation", "corp", "ltd"]
def nz(s):
    s = unicodedata.normalize("NFKC", s or "").lower()
    for f_ in FORM: s = s.replace(f_, "")
    return re.sub(r"[^0-9a-z가-힣]", "", s)
def dice(a, b):
    a, b = nz(a), nz(b)
    if not a or not b: return None
    if a == b: return 1.0
    A = collections.Counter(a[i:i + 2] for i in range(max(1, len(a) - 1))); B = collections.Counter(b[i:i + 2] for i in range(max(1, len(b) - 1)))
    inter = sum((A & B).values()); return 2 * inter / (sum(A.values()) + sum(B.values()))
r = ro(os.path.join(BASE, r"수집파이프라인\registry_full.sqlite"))
app = collections.defaultdict(set); napp = collections.Counter()
for rn, cd in r.execute("select rgst_no, applicantCd from applicant"):
    if cd: app[rn].add(cd)
solo = {rn: next(iter(s)) for rn, s in app.items() if len(s) == 1}
own0 = {}
for rn, nm in r.execute("select rgst_no, ownerName from owner where seq=0"):
    if rn in solo: own0[rn] = nm
byc = collections.defaultdict(collections.Counter)
for rn, nm in own0.items():
    byc[solo[rn]][nm] += 1
import pyarrow.parquet as pq
K = pq.read_table(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"), columns=["등록번호", "출원인명"]).to_pydict()
prov = collections.defaultdict(collections.Counter)
for rn, nm in zip(K["등록번호"], K["출원인명"]):
    if rn and rn in solo and nm:
        prov[solo[rn]][re.split(r"\s*[|;]\s*", nm)[0]] += 1
caname = {r_[0]: r_[1] for r_ in rows}
codes = [c for c in set(solo.values()) if c[:1] == "1" and c in caname and (cam[c][0] or cam[c][1])]
scored = low = 0; lowlist = []
for c in codes:
    cands = []
    if byc[c]: cands.append(byc[c].most_common(1)[0][0])
    if prov[c]: cands.append(prov[c].most_common(1)[0][0])
    if not cands: continue
    s = max((dice(caname[c], x) or 0) for x in cands)
    scored += 1
    if s < 0.5:
        low += 1; lowlist.append((c, cands))
OUT["d_scored"] = scored; OUT["d_low_dice_lt05"] = low; OUT["d_low_share"] = low / scored
# 반박 시험: 하위 층의 옛 이름이 CA 안 다른 국내 법인 코드의 이름과 같고(정규화 일치) 그 코드의 법인번호가 다른가?
name2codes = collections.defaultdict(set)
for r_, j in zip(rows, jn):
    if r_[0][:1] == "1": name2codes[nz(r_[1])].add((r_[0], j))
hit = []
for c, cands in lowlist:
    myj = cam[c][1]
    for x in cands:
        if nz(x) == nz(caname[c]): continue
        others = [(oc, oj) for oc, oj in name2codes.get(nz(x), set()) if oc != c]
        if others:
            hit.append(dict(cd=c[:5] + "****" + c[-3:], ca_name=caname[c], old_name=x, other_codes=len(others), other_same_jur=sum(oj == myj for _, oj in others)))
            break
OUT["refute_low_with_oldname_as_other_code"] = len(hit)
OUT["refute_examples"] = hit[:30]
random.seed(1)
json.dump(OUT, open(os.path.join(HERE, "v0_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=0))
