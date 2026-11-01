# -*- coding: utf-8 -*-
r"""R9 T1·기업업종 TXT 대비 시험 — 공용. 쓰기는 이 폴더(T1_산업대기실\TXT_R9)에만. 나머지는 읽기만."""
import os, sys, time, re, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
OUT = os.path.join(MC, r"T1_산업대기실\TXT_R9")
CACHE = os.path.join(OUT, "cache"); RES = os.path.join(OUT, "res")
NC8 = os.path.join(MC, r"T1_산업대기실\NONCIRC_R8")
R8 = os.path.join(MC, "_ideas_R8")
LV = os.path.join(MC, r"_shared_report\label_validity")
EXT = os.path.join(MC, r"_shared\ext")
KSIC_DEF = os.path.join(BASE, r"공모전데이터\외부데이터\KSIC해설\ksic50_definitions.json")
_T = time.time()


def log(*a):
    print(*a, f"[{round(time.time() - _T)}s]", flush=True)


def ro(path):
    import sqlite3
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


# label_validity lv_common 그룹 규칙(복사)
GROUP_PREFIX = [
    ("C1500", ["15"]), ("C2030", ["202"]), ("C2042", ["2041"]), ("C2043", ["2042"]), ("C2049", ["2049"]),
    ("C2100", ["21"]), ("C2300", ["23"]), ("C2410", ["241"]), ("C2420", ["242"]), ("C2590", ["259"]),
    ("C2610", ["261"]), ("C2620", ["262"]), ("C2630", ["263"]), ("C2640", ["264"]), ("C2650", ["265"]),
    ("C2710", ["271"]), ("C2720", ["272"]), ("C2730", ["273"]), ("C2810", ["281"]), ("C2850", ["285"]),
    ("C2890", ["289"]), ("C2918", ["2918"]), ("C2919", ["291"]), ("C2920", ["292"]), ("C3000", ["30"]),
    ("C3120", ["312"]), ("C3130", ["313"]), ("C3300", ["33"]),
]
OTHER_PREFIX = [
    ("C2010", ["201"]), ("C2020", ["2031"]), ("C2041", ["2032"]), ("C2050", ["205"]), ("C2200", ["22"]),
    ("C2430", ["243"]), ("C2510", ["251"]), ("C2520", ["252"]), ("C2660", ["266"]), ("C2740", ["274"]),
    ("C2820", ["282"]), ("C2830", ["283"]), ("C2840", ["284"]), ("C3110", ["311"]), ("C3191", ["3191"]),
    ("C3192", ["3192"]), ("C3199", ["319"]), ("C3200", ["32"]), ("C1000", ["10"]), ("C1100", ["11"]),
    ("C1200", ["12"]), ("C1300", ["13"]), ("C1400", ["14"]), ("C1600", ["16"]), ("C1700", ["17"]),
    ("C1800", ["18"]), ("C1900", ["19"]),
]
SPACE = [g for g, _ in GROUP_PREFIX if g != "C2918"]


def ksic_to_group(code, include_other=False):
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    c = re.sub(r"\D", "", str(code))
    if len(c) < 2:
        return None
    for g, pre in GROUP_PREFIX + (OTHER_PREFIX if include_other else []):
        for p in pre:
            if c.startswith(p):
                return g
    return None


def ksic9_crosswalk_3digit(g):
    m = {"C1500": "152", "C2100": "212", "C2300": "239", "C3000": "303", "C3300": "333",
         "C2030": "202", "C2042": "204", "C2043": "204", "C2049": "204"}
    if g in m:
        return m[g]
    if g is None or not isinstance(g, str):
        return None
    return g[1:4]


KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
K2G = {k: ksic_to_group(k) for k in KSIC50}
G2K = {}
for k, g in K2G.items():
    G2K.setdefault(g, []).append(k)
MULTI_K = sorted([k for k in KSIC50 if len(G2K[K2G[k]]) > 1])
UNIQ = {g: v[0] for g, v in G2K.items() if len(v) == 1}
