# -*- coding: utf-8 -*-
"""r3kosis_ 공용 도우미 — KOSIS OpenAPI 호출.
키는 수집파이프라인/collect_kosis_new.py 에서 정규식으로 '읽기만' 한다(파일 수정 금지).
키 값은 절대 출력·저장하지 않는다(오류 메시지에서도 가림).
"""
import os, re, json, time, urllib.request, urllib.parse

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(BASE, "분석", "모델경쟁_20260927", "_shared", "ext")
RAW = os.path.join(EXT, "r3kosis_raw")
os.makedirs(RAW, exist_ok=True)

def _read_key():
    p = os.path.join(BASE, "수집파이프라인", "collect_kosis_new.py")
    with open(p, encoding="utf-8") as f:
        s = f.read()
    m = re.search(r'KEY\s*=\s*urllib\.parse\.quote\(\s*"([^"]+)"\s*\)', s)
    return urllib.parse.quote(m.group(1))

KEY = _read_key()
_last = [0.0]
MIN_GAP = 0.6
CALLS = {"n": 0, "err": 0}

def _redact(s):
    return str(s).replace(KEY, "<KEY>").replace(urllib.parse.unquote(KEY), "<KEY>")

class KosisLimit(Exception):
    pass

def call(url_base, params, timeout=120, tries=3):
    """params dict (apiKey 자동). 반환: 파싱된 JSON(list/dict). 한도·차단 신호면 KosisLimit."""
    q = dict(params); q["apiKey"] = "__K__"
    qs = urllib.parse.urlencode(q, safe="+,").replace("__K__", KEY)
    url = f"{url_base}?{qs}"
    for a in range(tries):
        gap = time.time() - _last[0]
        if gap < MIN_GAP:
            time.sleep(MIN_GAP - gap)
        _last[0] = time.time()
        CALLS["n"] += 1
        try:
            r = urllib.request.urlopen(url, timeout=timeout).read().decode("utf-8", errors="replace")
            try:
                j = json.loads(r)
            except Exception:
                # KOSIS 는 가끔 비표준 JSON(따옴표 없는 키)을 준다
                r2 = re.sub(r'([{,])\s*([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', r)
                j = json.loads(r2)
            if isinstance(j, dict) and "err" in j:
                code = str(j.get("err"))
                # KOSIS 오류코드: 20 필수변수 누락, 21 잘못된 변수, 30 데이터 없음, 31 4만셀 초과,
                # 40 호출건수 제한, 41 트래픽 제한, 42 사용자 이용제한 → 40번대는 즉시 중단
                if code in ("40", "41", "42"):
                    raise KosisLimit(_redact(j))
            return j
        except KosisLimit:
            raise
        except Exception as e:
            CALLS["err"] += 1
            if a == tries - 1:
                return {"err": "EXC", "errMsg": _redact(f"{type(e).__name__}: {e}")[:200]}
            time.sleep(3 * (2 ** a))

LIST_URL = "https://kosis.kr/openapi/statisticsList.do"
SEARCH_URL = "https://kosis.kr/openapi/statisticsSearch.do"
META_URL = "https://kosis.kr/openapi/statisticsData.do"
DATA_URL = "https://kosis.kr/openapi/Param/statisticsParameterData.do"

def search(word, n=50):
    return call(SEARCH_URL, {"method": "getList", "format": "json", "searchNm": word,
                             "sort": "RANK", "startCount": 1, "resultCount": n})

def meta(tbl, typ, org="101"):
    return call(META_URL, {"method": "getMeta", "type": typ, "orgId": org, "tblId": tbl,
                           "format": "json"})

def data(tbl, itm, objs, prd_se, st, en, org="101"):
    p = {"method": "getList", "orgId": org, "tblId": tbl, "itmId": itm, "format": "json",
         "jsonVD": "Y", "prdSe": prd_se, "startPrdDe": st, "endPrdDe": en}
    for i, o in enumerate(objs, 1):
        p[f"objL{i}"] = o
    return call(DATA_URL, p, timeout=180)
