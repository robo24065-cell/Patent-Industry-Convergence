# -*- coding: utf-8 -*-
# C11-T1 공용 경로. 읽기 전용 입력과 이 폴더(쓰기) 경로.
import os, sqlite3
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
AN = os.path.join(BASE, "분석")
W = os.path.join(AN, r"검증루프_20261012\C11\C11-T1")
CACHE = os.path.join(W, "cache")
CA_TXT = os.path.join(BASE, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt")
DART_DB = os.path.join(BASE, r"공모전데이터\외부데이터\DART\dart_company.sqlite")
DART_CA = os.path.join(BASE, r"공모전데이터\외부데이터\DART\corp_applicant.parquet")
REG_DB = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
BR_DB = os.path.join(BASE, r"수집파이프라인\business_rights.sqlite")
MC = os.path.join(AN, "모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
MANU = os.path.join(AN, r"보고서_20261012\manuscript_v3.md")


def ro(path):
    return sqlite3.connect("file:" + path.replace("\\", "/") + "?mode=ro", uri=True)


def guard_path(p):
    ap = os.path.abspath(p)
    if not ap.lower().startswith(os.path.abspath(W).lower()):
        raise RuntimeError("쓰기 금지 경로: " + ap)
    return ap


def read_ca():
    import pandas as pd
    rows = []
    with open(CA_TXT, encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\u00b6")
        for line in f:
            parts = line.rstrip("\n").split("\u00b6")
            rows.append(parts)
    return hdr, rows
