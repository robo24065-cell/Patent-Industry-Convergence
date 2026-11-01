# -*- coding: utf-8 -*-
r"""R7 §3 제공 KSIC 라벨의 외부 타당성 — 공용 함수.
- 공식 IPC-KSIC 연계표(특허청 2013, KSIC 9차, 61개 산업그룹, 해상도 혼합 2~4자리)를 KSIC 10차 코드에 맞춘다.
- 차수 맞춤: 연계표 그룹을 KSIC 10차 접두어 규칙으로 정의(예: 9차 C2042 '잉크·페인트·코팅제' = 10차 2041 '일반용 도료').
- 쓰기는 이 폴더(_shared_report/label_validity)에만. 원본은 읽기 전용.
"""
import os, re, time, sqlite3
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
EXT = os.path.join(MC, r"_shared\ext")
META = os.path.join(MC, r"_shared\meta\*.parquet").replace("\\", "/")
OUT = os.path.join(MC, r"_shared_report\label_validity")
CW_XLSX = os.path.join(BASE, r"공모전데이터\외부데이터\20131002_IPC-KSIC\20131002_IPC-KSIC.xlsx")
DEV_END = 2017  # 산업 단위 비교 개발 기간 상한(T1 최종시험 F=2018 이전)
_T = time.time()


def log(*a):
    print(*a, f"[{round(time.time() - _T)}s]", flush=True)


def ro(path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


# 연계표 그룹 → KSIC 10차 접두어(라벨공간 27개 그룹). 앞의 규칙이 우선(2918이 291보다 먼저).
GROUP_PREFIX = [
    ("C1500", ["15"]), ("C2030", ["202"]), ("C2042", ["2041"]), ("C2043", ["2042"]), ("C2049", ["2049"]),
    ("C2100", ["21"]), ("C2300", ["23"]), ("C2410", ["241"]), ("C2420", ["242"]), ("C2590", ["259"]),
    ("C2610", ["261"]), ("C2620", ["262"]), ("C2630", ["263"]), ("C2640", ["264"]), ("C2650", ["265"]),
    ("C2710", ["271"]), ("C2720", ["272"]), ("C2730", ["273"]), ("C2810", ["281"]), ("C2850", ["285"]),
    ("C2890", ["289"]), ("C2918", ["2918"]), ("C2919", ["291"]), ("C2920", ["292"]), ("C3000", ["30"]),
    ("C3120", ["312"]), ("C3130", ["313"]), ("C3300", ["33"]),
]
# 라벨공간 밖이지만 10차 코드로 식별 가능한 다른 연계표 그룹(기업 등록업종 매핑용, 참고)
OTHER_PREFIX = [
    ("C2010", ["201"]), ("C2020", ["2031"]), ("C2041", ["2032"]), ("C2050", ["205"]), ("C2200", ["22"]),
    ("C2430", ["243"]), ("C2510", ["251"]), ("C2520", ["252"]), ("C2660", ["266"]), ("C2740", ["274"]),
    ("C2820", ["282"]), ("C2830", ["283"]), ("C2840", ["284"]), ("C3110", ["311"]), ("C3191", ["3191"]),
    ("C3192", ["3192"]), ("C3199", ["319"]), ("C3200", ["32"]), ("C1000", ["10"]), ("C1100", ["11"]),
    ("C1200", ["12"]), ("C1300", ["13"]), ("C1400", ["14"]), ("C1600", ["16"]), ("C1700", ["17"]),
    ("C1800", ["18"]), ("C1900", ["19"]),
]


def ksic_to_group(code, include_other=False):
    """KSIC 10차 코드(2~5자리) → 연계표 그룹. 코드가 그룹 접두어보다 짧으면 판정 불가(None)."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    c = re.sub(r"\D", "", str(code))
    if len(c) < 2:
        return None
    rules = GROUP_PREFIX + (OTHER_PREFIX if include_other else [])
    for g, pre in rules:
        for p in pre:
            if c.startswith(p):
                return g
            # 코드가 접두어보다 짧고 접두어와 앞이 겹치면 판정 불가(예: '29'는 2918/291/292 중 어디인지 모름)
    return None


def ipc_norm(s):
    s = re.sub(r"\s+", "", str(s or "")).upper()
    m = re.match(r"^([A-H]\d{2}[A-Z])(\d+)/(\d+)", s)
    if m:
        return f"{m.group(1)}{int(m.group(2))}/{m.group(3)}", f"{m.group(1)}{int(m.group(2))}", m.group(1)
    m = re.match(r"^([A-H]\d{2}[A-Z])", s)
    if m:
        return "", "", m.group(1)
    return "", "", ""


def load_crosswalk():
    cc = pd.read_excel(CW_XLSX, sheet_name=1, dtype=str)
    cc.columns = ["ind", "ipc", "cnt"]
    cc["cnt"] = pd.to_numeric(cc.cnt, errors="coerce").fillna(0)
    cc = cc[cc.ind.str.match(r"^[A-Z]\d{4}$", na=False)].copy()
    z = cc.ipc.map(ipc_norm)
    cc["sg"] = [a for a, b, c in z]
    cc["mg"] = [b for a, b, c in z]
    cc["sc"] = [c for a, b, c in z]
    return cc


SPACE = [g for g, _ in GROUP_PREFIX]
SPACE = [g for g in SPACE if g != "C2918"]  # 50개 라벨이 하나도 없는 그룹은 라벨공간에서 제외 → 27개


def make_lookup(cc, constrained=True):
    """IPC 키(subgroup→main group→subclass) → (그룹, 확신도). constrained=True면 라벨공간 27개 그룹 안에서 argmax.
    확신도는 항상 '전체 61개 그룹 대비' 그 그룹 몫."""
    tabs = {}
    for lvl in ("sg", "mg", "sc"):
        g = cc[cc[lvl] != ""].groupby([lvl, "ind"], as_index=False).cnt.sum()
        tot = g.groupby(lvl).cnt.sum().rename("tot")
        gg = g[g.ind.isin(SPACE)] if constrained else g
        gg = gg.sort_values(["cnt", "ind"], ascending=[False, True]).drop_duplicates(lvl)
        gg = gg.merge(tot, left_on=lvl, right_index=True)
        gg["conf"] = gg.cnt / gg.tot
        tabs[lvl] = gg.set_index(lvl)[["ind", "conf"]]
    return tabs


def assign(df, tabs, col="main_ipc"):
    """df[col] 메인 IPC → cw_group, cw_conf, cw_level."""
    z = df[col].map(ipc_norm)
    sg = pd.Series([a for a, b, c in z], index=df.index)
    mg = pd.Series([b for a, b, c in z], index=df.index)
    sc = pd.Series([c for a, b, c in z], index=df.index)
    out = pd.DataFrame(index=df.index, columns=["cw_group", "cw_conf", "cw_level"], dtype=object)
    for lvl, key in (("sg", sg), ("mg", mg), ("sc", sc)):
        need = out.cw_group.isna()
        t = tabs[lvl]
        hit = need & key.isin(t.index)
        out.loc[hit, "cw_group"] = t.loc[key[hit], "ind"].values
        out.loc[hit, "cw_conf"] = t.loc[key[hit], "conf"].values
        out.loc[hit, "cw_level"] = lvl
    out["cw_conf"] = pd.to_numeric(out.cw_conf)
    return out


def kappa(a, b):
    ok = a.notna() & b.notna()
    a, b = a[ok].astype(str), b[ok].astype(str)
    po = float((a.values == b.values).mean())
    pa, pb = a.value_counts(normalize=True), b.value_counts(normalize=True)
    pe = float(sum(pa.get(c, 0) * pb.get(c, 0) for c in set(pa.index) | set(pb.index)))
    return (po - pe) / (1 - pe), po, pe, int(ok.sum())


def ksic9_crosswalk_3digit(g):
    """연계표 그룹 → 비교용 3자리(10차). 2자리 그룹(15·21·23·30·33)은 라벨공간 안 3자리가 유일하므로 승격."""
    m = {"C1500": "152", "C2100": "212", "C2300": "239", "C3000": "303", "C3300": "333",
         "C2030": "202", "C2042": "204", "C2043": "204", "C2049": "204"}
    if g in m:
        return m[g]
    if g is None or not isinstance(g, str):
        return None
    return g[1:4]
