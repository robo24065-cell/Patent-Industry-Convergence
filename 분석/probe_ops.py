# -*- coding: utf-8 -*-
"""
안 쓰고 있던 오퍼레이션 3종 타진
────────────────────────────────────────────────────────────────
  getDesignHistory      디자인(등록번호 30…)  — 우리 대상에 있나?
  getMarkHistory        상표(40…)            — 우리 대상에 있나?
  getBusinessRightList  ★ 사업자번호·법인등록번호·출원인번호로 그 법인의 권리 목록

★ getBusinessRightList 가 중요한 이유
  제공 데이터(융합데이터)는 KSIC 세세분류 **50개로 잘려 있다.**
  어떤 기업이 그 50개 밖에서 낸 특허는 우리 눈에 아예 안 보인다.
  이 절단 때문에 '기업 단위 분석'은 구조적으로 불가능했다
  (아이디어 심사에서 심사위원 3인 전원이 이 이유로 치명결함 판정을 내렸다).
  출원인번호로 그 기업의 **전체 포트폴리오**를 복원할 수 있다면 그 제약이 풀린다.

우리가 가진 조회 키
  applicantCd  출원인번호  570,282행 (완전추출 결과, applicant 테이블)
  법인등록번호  385,256개 중 99.4%
  사업자번호   385,256개 중 72.4% / DART 12,322개사는 100%
"""
import json, os, sqlite3, sys, io, time
import urllib.request, urllib.error

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
HERE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인"
sys.path.insert(0, HERE)
from collect_registry import DATAGO_KEYS

BASEURL = "http://apis.data.go.kr/1430000/PttRgstRtInfoInqSvc"
KEY = DATAGO_KEYS[-1]          # 수집기가 쓰는 키와 겹치지 않게 마지막 키 사용
log = lambda m: print(m, flush=True)


def call(op, **params):
    q = "&".join(f"{k}={v}" for k, v in params.items())
    u = f"{BASEURL}/{op}?serviceKey={KEY}&{q}&type=json"
    try:
        r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "probe"}),
                                   timeout=30)
        return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


# ══════════════════════════════════════════════════════════════
log("[1] 우리 수집 대상에 디자인·상표가 있는가")
c = sqlite3.connect(f"file:{os.path.join(HERE,'registry.sqlite')}?mode=ro", uri=True)
rows = c.execute("SELECT substr(rgst_no,1,1) p, COUNT(*) FROM pending GROUP BY 1 ORDER BY 2 DESC").fetchall()
kind = {"1": "특허", "2": "실용신안", "3": "디자인", "4": "상표"}
for p, n in rows:
    log(f"  등록번호 {p}… = {kind.get(p,'?'):8s} {n:8,}건")
log("  → 융합데이터는 특허·실용신안만 담고 있다. 디자인·상표는 대상에 없다.")

# ══════════════════════════════════════════════════════════════
log("\n[2] ★ getBusinessRightList — 조회 키를 바꿔가며 타진")
# 실제 데이터에서 키 샘플 확보
full = os.path.join(HERE, "registry_full.sqlite")
appl_cd = biz = jurir = None
if os.path.exists(full):
    f = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    r = f.execute("SELECT applicantCd, applicantName FROM applicant "
                  "WHERE applicantCd IS NOT NULL LIMIT 1").fetchone()
    if r:
        appl_cd, appl_nm = r
        log(f"  샘플 출원인번호 {appl_cd} ({appl_nm})")
    f.close()
d = sqlite3.connect(os.path.join(HERE, "..", "공모전데이터", "외부데이터", "DART", "dart_company.sqlite"))
r = d.execute("SELECT corp_name, bizr_no, jurir_no FROM company "
              "WHERE bizr_no<>'' AND jurir_no<>'' LIMIT 1").fetchone()
d.close()
if r:
    nm, biz, jurir = r
    log(f"  샘플 사업자번호 {biz} / 법인등록번호 {jurir} ({nm})")
c.close()

for label, params in (
        ("출원인번호", {"applicantNo": appl_cd}),
        ("출원인번호(별칭)", {"applicantCd": appl_cd}),
        ("사업자번호", {"bizrNo": biz}),
        ("법인등록번호", {"jurirNo": jurir}),
        ("사업자번호(별칭)", {"businessNo": biz})):
    if not list(params.values())[0]:
        continue
    st, body = call("getBusinessRightList", **params)
    head = body[:220].replace("\n", " ")
    log(f"\n  [{label}] {list(params)[0]}={list(params.values())[0]}")
    log(f"    HTTP {st} | {head}")
    if st == 200:
        try:
            j = json.loads(body)
            if j.get("resultCode") == "000":
                it = j.get("items") or {}
                log(f"    ★ 성공 — items 키: {list(it)[:12] if isinstance(it,dict) else type(it)}")
                log(f"    totalCount = {j.get('totalCount')}")
        except Exception:
            pass
    time.sleep(0.5)

# ══════════════════════════════════════════════════════════════
log("\n[3] 디자인·상표 오퍼레이션 응답 확인 (참고용)")
for op, no in (("getDesignHistory", "3005000000000"), ("getMarkHistory", "4000000010000")):
    st, body = call(op, rgstNo=no)
    log(f"  {op}: HTTP {st} | {body[:150].replace(chr(10),' ')}")
    time.sleep(0.5)
