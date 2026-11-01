# 국세청 사업자등록 상태조회 수집기 (이어받기 가능, 1회 실행 시간 제한, 키 값 비출력)
import sys, json, time, os, requests, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from r3nts_keytest import load_keys, URL
EXT = os.path.dirname(os.path.abspath(__file__))
W = os.path.join(EXT, "r3nts_work")
OUT = os.path.join(W, "nts_status_raw.jsonl")
LOG = os.path.join(W, "collect.log")
BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 510   # 초
GAP = float(sys.argv[2]) if len(sys.argv) > 2 else 0.4
key = dict(load_keys())[1]   # 키1 = 승인 계정(시험 호출로 확인)
tg = pd.read_parquet(os.path.join(W, "targets.parquet"))
done = set()
if os.path.exists(OUT):
    with open(OUT, encoding="utf-8") as f:
        for l in f:
            try: done.add(json.loads(l)["b_no"])
            except Exception: pass
todo = [b for b in tg.bizno if b not in done]
def log(m):
    with open(LOG, "a", encoding="utf-8") as f: f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + m + "\n")
log(f"start done={len(done)} todo={len(todo)}")
t0 = time.time(); ncall = nok = nfail = 0; backoff = 2
fo = open(OUT, "a", encoding="utf-8")
i = 0
stop_reason = "budget"
while i < len(todo):
    if time.time() - t0 > BUDGET: break
    batch = todo[i:i + 100]
    try:
        r = requests.post(URL, params={"serviceKey": key, "returnType": "JSON"}, json={"b_no": batch}, timeout=40)
        ncall += 1
    except Exception as e:
        nfail += 1; log(f"exception {type(e).__name__}; backoff {backoff}s"); time.sleep(backoff); backoff = min(backoff * 2, 120)
        if nfail > 8: stop_reason = "repeated_exceptions"; break
        continue
    if r.status_code != 200:
        nfail += 1
        msg = r.text[:200].replace(key, "<KEY>")
        log(f"HTTP {r.status_code} {msg}")
        if r.status_code in (401, 403, 429) or "LIMITED" in msg.upper() or "초과" in msg:
            stop_reason = f"limit_or_auth_{r.status_code}"; break
        time.sleep(backoff); backoff = min(backoff * 2, 120)
        if nfail > 8: stop_reason = "repeated_http_errors"; break
        continue
    d = r.json()
    got = {x["b_no"]: x for x in d.get("data", [])}
    ts = time.strftime("%Y-%m-%d")
    for b in batch:
        x = got.get(b, {"b_no": b, "missing": True})
        x["query_date"] = ts
        fo.write(json.dumps(x, ensure_ascii=False) + "\n")
    fo.flush(); nok += 1; backoff = 2; i += 100; nfail = 0  # 연속 실패만 셈
    time.sleep(GAP)
else:
    stop_reason = "all_done"
fo.close()
log(f"end calls={ncall} ok={nok} fail={nfail} processed={i} remaining={max(0,len(todo)-i)} reason={stop_reason} secs={time.time()-t0:.0f}")
print(f"calls={ncall} ok={nok} fail={nfail} processed={min(i,len(todo))} remaining={max(0,len(todo)-i)} reason={stop_reason}")
