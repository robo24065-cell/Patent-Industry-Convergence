# -*- coding: utf-8 -*-
"""
getBusinessRightList 검증 — 기업 전체 포트폴리오를 받을 수 있는가
────────────────────────────────────────────────────────────────
파라미터 확정:
  searchType  1=사업자등록번호  2=출원인번호  3=법인등록번호
  searchVal   해당 값 (샘플은 하이픈 포함: 301-81-08302)
  pageNo / numOfRows / type=json

핵심 검정
  [Q1] 세 가지 키가 다 되는가, 하이픈 유무는 상관없는가
  [Q2] 응답에 어떤 필드가 오는가
  [Q3] ★ 융합데이터 50개 KSIC **밖의 특허도 나오는가**
       → 나온다면 '기업 포트폴리오 절단' 문제가 풀린다
"""
import json, os, sqlite3, sys, io, time
import urllib.request, urllib.error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
BASE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from collect_registry import DATAGO_KEYS

URL = "http://apis.data.go.kr/1430000/PttRgstRtInfoInqSvc/getBusinessRightList"
KEY = DATAGO_KEYS[-1]
log = lambda m: print(m, flush=True)


def call(stype, val, rows=100, page=1):
    u = (f"{URL}?serviceKey={KEY}&pageNo={page}&numOfRows={rows}&type=json"
         f"&searchType={stype}&searchVal={val}")
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "probe"}),
                                   timeout=30)
        return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def brief(body, n=200):
    return body[:n].replace("\n", " ")


# ══════════════════════════════════════════════════════════════
log("[Q1] 공식 샘플로 먼저 확인")
st, b = call(1, "301-81-08302")
log(f"  HTTP {st} | {brief(b)}")
try:
    j = json.loads(b)
    log(f"  resultCode={j.get('resultCode')} totalCount={j.get('totalCount')}")
except Exception:
    j = None

# 우리 데이터의 키 확보
d = sqlite3.connect(os.path.join(BASE, "공모전데이터", "외부데이터", "DART", "dart_company.sqlite"))
samples = d.execute("SELECT corp_name, bizr_no, jurir_no FROM company "
                    "WHERE bizr_no<>'' AND jurir_no<>'' AND induty_code LIKE '26%' LIMIT 3").fetchall()
d.close()
full = os.path.join(HERE, "registry_full.sqlite")
appl = []
if os.path.exists(full):
    f = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    appl = f.execute("SELECT applicantCd, applicantName, COUNT(*) n FROM applicant "
                     "WHERE applicantCd IS NOT NULL GROUP BY 1 ORDER BY n DESC LIMIT 3").fetchall()
    f.close()

log("\n[Q1-b] 우리 데이터 키로 — 하이픈 유무 포함")
for nm, biz, jur in samples[:2]:
    bz = f"{biz[:3]}-{biz[3:5]}-{biz[5:]}" if len(biz) == 10 else biz
    for stype, val, lab in ((1, bz, "사업자(하이픈)"), (1, biz, "사업자(숫자만)"),
                            (3, jur, "법인등록번호")):
        st, b = call(stype, val, rows=5)
        try:
            jj = json.loads(b); rc = jj.get("resultCode"); tc = jj.get("totalCount")
        except Exception:
            rc = tc = "?"
        log(f"  {nm[:14]:16s} {lab:14s} {val:16s} → rc={rc} total={tc}")
        time.sleep(0.4)

log("\n[Q1-c] 출원인번호(searchType=2)")
for cd, nm, n in appl:
    st, b = call(2, cd, rows=5)
    try:
        jj = json.loads(b); rc = jj.get("resultCode"); tc = jj.get("totalCount")
    except Exception:
        rc = tc = "?"
    log(f"  {str(nm)[:16]:18s} 출원인번호 {cd} (융합데이터 내 {n:,}건) → rc={rc} total={tc}")
    time.sleep(0.4)

# ══════════════════════════════════════════════════════════════
log("\n[Q2] 응답 구조 — 성공 사례 하나를 펼쳐본다")
target = None
for cd, nm, n in appl:
    st, b = call(2, cd, rows=100)
    try:
        jj = json.loads(b)
    except Exception:
        continue
    if jj.get("resultCode") == "000" and jj.get("totalCount"):
        target = (cd, nm, n, jj); break
    time.sleep(0.4)

if target:
    cd, nm, n_fusion, jj = target
    it = jj.get("items")
    lst = it if isinstance(it, list) else (it.get("item") if isinstance(it, dict) else None)
    if isinstance(lst, dict):
        lst = [lst]
    log(f"  출원인 {nm} ({cd})")
    log(f"  totalCount = {jj.get('totalCount'):,}  |  융합데이터 내 보유 = {n_fusion:,}건")
    if lst:
        log(f"  레코드 키: {list(lst[0])}")
        log(f"  첫 레코드: {json.dumps(lst[0], ensure_ascii=False)[:400]}")

        # ── Q3 절단 검정
        log("\n[Q3] ★ 융합데이터 50개 KSIC 밖의 특허도 나오는가")
        c = sqlite3.connect(f"file:{os.path.join(HERE,'registry.sqlite')}?mode=ro", uri=True)
        ours = {r[0] for r in c.execute("SELECT rgst_no FROM pending")}
        c.close()
        keys = [k for k in lst[0] if "rgst" in k.lower() or "regi" in k.lower()]
        log(f"  등록번호로 보이는 키: {keys}")
        if keys:
            k0 = keys[0]
            got = [str(x.get(k0, "")).strip() for x in lst]
            inside = sum(1 for g in got if g in ours)
            log(f"  받은 {len(got)}건 중 융합데이터에 있는 것 {inside}건, "
                f"**밖의 것 {len(got)-inside}건**")
            if len(got) - inside > 0:
                log("  → ★ 절단 밖 특허가 나온다. 기업 전체 포트폴리오 복원 가능.")
            else:
                log("  → 이 표본에서는 전부 안쪽. 더 큰 표본으로 재확인 필요.")
else:
    log("  성공 사례를 못 찾았다. 응답 원문:")
    log(f"  {brief(b, 500)}")
