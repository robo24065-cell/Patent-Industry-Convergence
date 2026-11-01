# NTIS 성과(rpatent) 전수 수집 — 연도(PY) 분할 x 페이지 1000건. 이어받기 가능. 1회 실행 ≤ 8분.
# 종료코드: 0=시간 소진(다시 실행), 2=전체 완료, 3=오류/한도 신호로 중단
import json, re, sys, time, gzip
import xml.etree.ElementTree as ET
import ntis_common as n
WORK = n.WORK; WORK.mkdir(exist_ok=True)
STATE = WORK / "ntis_rpatent_state.json"
LOG = WORK / "ntis_rpatent_collect.log"
YEARS = list(range(2007, 2027))
PAGE = 1000
T_LIMIT = 480
def log(msg):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + n.redact(msg) + "\n")
def parse(x):
    x = re.sub(r"^<\?xml[^>]*\?>", "", x.strip())
    root = ET.fromstring(x)
    err = root.find("ERROR")
    if err is not None:
        return None, None, (err.findtext("CODE"), err.findtext("MESSAGE"))
    total = int(root.findtext("TOTALHITS") or -1)
    rows = []
    for hit in root.iter("HIT"):
        r = {"SITID": hit.get("SITID")}
        for c in hit:
            r[c.tag] = (c.text or "").strip()
            if c.get("code") is not None:
                r[c.tag + "_code"] = c.get("code")
        rows.append(r)
    return total, rows, None
def main():
    st = json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else {"years": {}, "calls": 0}
    t0 = time.time()
    for y in YEARS:
        ys = st["years"].setdefault(str(y), {"next": 1, "total": None, "got": 0, "done": False})
        while not ys["done"]:
            if time.time() - t0 > T_LIMIT:
                STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
                log(f"시간 소진 종료 calls={st['calls']}"); return 0
            sp = ys["next"]
            try:
                x = n.call("public_result", {"collection": "rpatent", "addQuery": f"PY={y}/SAME",
                                             "startPosition": sp, "displayCnt": PAGE}, timeout=120)
                st["calls"] += 1
                total, rows, err = parse(x)
            except Exception as e:
                STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
                log(f"예외 y={y} sp={sp}: {type(e).__name__} {e}"); return 3
            if err:
                STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
                log(f"API오류 y={y} sp={sp}: {err}"); return 3
            if ys["total"] is None: ys["total"] = total
            elif total != ys["total"]:
                log(f"경고 y={y} total 변동 {ys['total']}->{total}"); ys["total"] = total
            with gzip.open(WORK / f"ntis_rp_{y}_{sp:06d}.jsonl.gz", "wt", encoding="utf-8") as f:
                for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")
            ys["got"] = sp - 1 + len(rows)
            ys["next"] = sp + PAGE
            if len(rows) == 0 or ys["next"] > ys["total"]:
                ys["done"] = True; log(f"연도 완료 y={y} total={ys['total']} got={ys['got']}")
            STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"전체 완료 calls={st['calls']}"); return 2
if __name__ == "__main__":
    sys.exit(main())
