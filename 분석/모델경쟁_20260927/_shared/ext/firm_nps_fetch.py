# -*- coding: utf-8 -*-
"""국민연금공단_국민연금 가입 사업장 내역(data.go.kr 파일데이터 15083277, 로그인 불필요) 연말 스냅샷 수집.
- 각 연도 12월 자료(2015-12 ~ 2025-12) + 최신(2026-08). 원본 CSV는 스크래치패드에 받고, 필요한 컬럼만 parquet로 ext에 저장.
- 예의: 요청 간 2초 이상, check-limit가 needCaptcha=true를 주면 즉시 중단(우회 금지), HTTP 오류 시 중단.
- 개인 식별 정보 없음(사업자번호 앞 6자리만 공개됨). 상세주소 컬럼은 저장하지 않음.
실행: python firm_nps_fetch.py [ym ...]
"""
import os, sys, time, json, subprocess
import pandas as pd

EXT = os.path.dirname(os.path.abspath(__file__))
RAW = os.environ.get("NPS_RAW", r"C:\Users\PC\AppData\Local\Temp\claude\C--Users-PC-Downloads------------------------\72c91564-e815-4ac3-bfb0-ac338e5a5b96\scratchpad\nps_raw")
os.makedirs(RAW, exist_ok=True)
PK = "15083277"
SNAP = {  # 자료기준월 -> publicDataDetailPk (주기성 과거데이터 목록에서 확인)
    "2015-12": "uddi:b9bf303e-a60e-4a49-a517-99797889484e",
    "2016-12": "uddi:dd4d69f9-2937-4785-83e0-b5fefdc361df",
    "2017-12": "uddi:f616dc6d-eb7a-45b1-a205-f91cdab72e40",
    "2018-12": "uddi:088e933f-b16a-4002-9122-d4fb0c0f9832",
    "2019-12": "uddi:02ad33d5-f387-4446-8884-a96bbf321a55",
    "2020-12": "uddi:b2a76201-9cf7-458d-a6e0-c2194e6ee40f",   # 2021-01-21 게시
    "2021-12": "uddi:aa411324-faf8-4c2d-a9e4-84caac160856",   # 2022-01-21 게시
    "2022-12": "uddi:e80b5227-bbd2-4620-ae95-b82064f54da0",
    "2023-12": "uddi:5cdf7e9e-dc5d-4369-96d3-d22744d8d10d",
    "2024-12": "uddi:3a89a14e-7230-467a-bf07-9ca33d06812d",
    "2025-12": "uddi:10a6e7bd-a2ee-4ee1-967c-bb9a6aea89a9",
    "2026-08": "uddi:8e0d590f-768e-4cbd-9560-c7168c8d13b6",
}
UA = "Mozilla/5.0"
REF = f"https://www.data.go.kr/data/{PK}/fileData.do"
COLS = ["ym", "name", "biz6", "stat", "zip", "addr_jibun", "addr_road", "ldong", "adong", "sido", "sgg", "emd",
        "form", "indcd", "indnm", "adpt_dt", "readpt_dt", "rrg_dt", "n_emp", "amt", "n_new", "n_loss"]
KEEP = ["ym", "name", "biz6", "stat", "ldong", "form", "indcd", "indnm", "adpt_dt", "readpt_dt", "rrg_dt", "n_emp", "amt", "n_new", "n_loss"]


def curl(args):
    r = subprocess.run(["curl", "-s", "-f", "-A", UA, "-e", REF] + args, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"curl rc={r.returncode}")
    return r.stdout


def fetch(ym, pk):
    out = os.path.join(EXT, f"firm_nps_wkpl_{ym.replace('-', '')}.parquet")
    if os.path.exists(out):
        print(ym, "exists"); return
    meta = json.loads(curl(["-H", "X-Requested-With: XMLHttpRequest",
                            f"https://www.data.go.kr/tcs/dss/selectFileDataDownload.do?publicDataDetailPk={pk}&publicDataPk={PK}&atchFileId=&fileDetailSn=1&publicDataTyCode=PR0051"]))
    if not meta.get("status"):
        raise RuntimeError(f"meta fail {ym}")
    aid, sn = meta["atchFileId"], meta["fileDetailSn"]
    time.sleep(2)
    lim = json.loads(curl(["-X", "POST", "-H", "X-Requested-With: XMLHttpRequest", "-d", f"atchFileId={aid}&fileDetailSn={sn}",
                           "https://www.data.go.kr/cmm/cmm/check-limit.json"]))
    if lim.get("needCaptcha"):
        print("needCaptcha=true -> 중단"); sys.exit(2)
    time.sleep(2)
    raw = os.path.join(RAW, f"nps_{ym}.csv")
    if not os.path.exists(raw):
        subprocess.run(["curl", "-s", "-f", "-A", UA, "-e", REF, "-o", raw,
                        f"https://www.data.go.kr/cmm/cmm/fileDownload.do?atchFileId={aid}&fileDetailSn={sn}"], check=True)
    with open(raw, "rb") as f:
        head = f.read(4)
    enc = "utf-8-sig" if head.startswith(b"\xef\xbb\xbf") else "cp949"
    df = pd.read_csv(raw, encoding=enc, dtype=str, encoding_errors="replace")
    if df.shape[1] != len(COLS):
        print(ym, "컬럼수", df.shape[1], list(df.columns)[:25])
        raise RuntimeError("schema")
    df.columns = COLS
    df = df[KEEP]
    for c in ["n_emp", "amt", "n_new", "n_loss"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["snap"] = ym
    df.to_parquet(out, index=False)
    print(ym, df.shape, "ym값", df.ym.value_counts().index[:2].tolist(), "enc", enc, flush=True)
    time.sleep(2)


if __name__ == "__main__":
    yms = sys.argv[1:] or list(SNAP)
    for ym in yms:
        fetch(ym, SNAP[ym])
