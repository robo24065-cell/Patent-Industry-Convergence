# -*- coding: utf-8 -*-
"""R7 데이터 3: data.go.kr 파일데이터(로그인 불필요) 1건 다운로드 — firm_nps_fetch.py와 같은 절차.
사용: python d3_datagokr_file.py <publicDataPk> <저장파일명(r7_data_ 접두어)>
예의: 요청 간 2초, check-limit needCaptcha=true면 즉시 중단(우회 금지)."""
import sys, os, re, json, time, subprocess
R7 = os.path.dirname(os.path.abspath(__file__)); EXT = os.path.join(os.path.dirname(R7), "_shared", "ext")
PK, OUT = sys.argv[1], sys.argv[2]
assert OUT.startswith("r7_data_")
UA = "Mozilla/5.0"; REF = f"https://www.data.go.kr/data/{PK}/fileData.do"
def curl(args):
    r = subprocess.run(["curl", "-s", "-f", "-A", UA, "-e", REF] + args, capture_output=True)
    if r.returncode: raise RuntimeError(f"curl rc={r.returncode}")
    return r.stdout
page = curl([REF]).decode("utf-8", "replace")
dpk = re.search(r'name="publicDataDetailPk" value="([^"]+)"', page).group(1)
time.sleep(2)
meta = json.loads(curl(["-H", "X-Requested-With: XMLHttpRequest",
    f"https://www.data.go.kr/tcs/dss/selectFileDataDownload.do?publicDataDetailPk={dpk}&publicDataPk={PK}&atchFileId=&fileDetailSn=1&publicDataTyCode=PR0051"]))
if not meta.get("status"): sys.exit(f"meta fail {meta}")
aid, sn = meta["atchFileId"], meta["fileDetailSn"]
print("파일:", {k: meta.get(k) for k in ("fileName", "atchFileId") if k in meta}, meta.get("dataSetFileDetailInfo", {}).get("orginlFileNm") if isinstance(meta.get("dataSetFileDetailInfo"), dict) else "")
time.sleep(2)
lim = json.loads(curl(["-X", "POST", "-H", "X-Requested-With: XMLHttpRequest", "-d", f"atchFileId={aid}&fileDetailSn={sn}", "https://www.data.go.kr/cmm/cmm/check-limit.json"]))
if lim.get("needCaptcha"): sys.exit("needCaptcha=true -> 중단")
time.sleep(2)
dst = os.path.join(EXT, OUT)
subprocess.run(["curl", "-s", "-f", "-A", UA, "-e", REF, "-o", dst, f"https://www.data.go.kr/cmm/cmm/fileDownload.do?atchFileId={aid}&fileDetailSn={sn}"], check=True)
print("저장", dst, os.path.getsize(dst))
print(json.dumps({k: v for k, v in meta.items() if isinstance(v, (str, int))}, ensure_ascii=False)[:600])
