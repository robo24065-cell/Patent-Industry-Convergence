# 국세청 상태조회 API: data.go.kr 키 중 승인 계정 확인 (키 값은 출력하지 않음)
import re, sys, json, time, urllib.parse, requests
KEYFILE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\API키_기록.txt"
URL = "https://api.odcloud.kr/api/nts-businessman/v1/status"
def load_keys():
    t = open(KEYFILE, encoding="utf-8").read()
    sec = t.split("[공공데이터포털")[1].split("[KRX]")[0]
    keys = []
    for l in sec.splitlines():
        m = re.match(r"\s*키(\d+)[^:]*:\s*(\S+)", l)
        if m: keys.append((int(m.group(1)), urllib.parse.unquote(m.group(2))))
    return keys
if __name__ == "__main__":
    which = [int(x) for x in sys.argv[1:]] or [1, 2]
    for kid, k in load_keys():
        if kid not in which: continue
        r = requests.post(URL, params={"serviceKey": k, "returnType": "JSON"}, json={"b_no": ["1048134422"]}, timeout=30)
        body = r.text[:300]
        print(kid, r.status_code, body.replace(k, "<KEY>"))
        time.sleep(1.5)
