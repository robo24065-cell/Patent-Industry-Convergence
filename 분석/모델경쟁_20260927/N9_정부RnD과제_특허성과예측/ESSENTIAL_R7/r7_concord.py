# -*- coding: utf-8 -*-
"""R7 필수성 검증 — 공식 IPC-KSIC 연계표(특허청 2013, KSIC 9차) 기반 '기존 다른 데이터' 산업 배정기.
읽기 전용: 공모전데이터\\외부데이터\\20131002_IPC-KSIC\\20131002_IPC-KSIC.xlsx (시트 '전체연계표': 산업코드, IPC 서브그룹, 건수).

배정 해상도: 연계표 산업코드는 A0000(대분류)·C1500(중분류)·C2610(소분류)·C2918(세분류) 해상도가 섞여 있고 세세분류(5자리)는 없다.
  KSIC 9차→10차 개정 영향을 피하려고 비교 해상도를 소분류 3자리로 상한한다(9·10차 3자리는 50개 산업이 속한 25개 소분류에서 안정).
  연계표에 3자리 항목이 없는 중분류(15·21·23·30·33)는 제공 라벨공간에 그 중분류 소분류가 하나뿐일 때만 3자리로 승격.
두 가지 배정:
  grp_free : 연계표 전체 산업 중 argmax (→ '50개 산업 범위인가?'를 연계표로 판정할 때. 범위 = argmax ∈ 25개 소분류)
  grp_in   : 제공 라벨공간 25개 소분류 안에서 argmax (→ 범위안 특허에 '어느 산업인가'를 붙일 때, 공정 비교용)
IPC 조회는 서브그룹 → 메인그룹 → 서브클래스 → 클래스 순 대체(fallback)."""
import os, re, json
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
CCX = os.path.join(BASE, r"공모전데이터\외부데이터\20131002_IPC-KSIC\20131002_IPC-KSIC.xlsx")
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924', '25932', '26111', '26112',
          '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323', '26422', '26429', '26519', '26529', '27199', '27211',
          '27212', '27301', '28111', '28122', '28511', '28902', '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280',
          '30393', '31202', '31312', '33301', '33309']
G50 = sorted({k[:3] for k in KSIC50})          # 25개 소분류


def ipc_keys(s):
    """'H01L 21/28' -> ('H01L21/28', 'H01L21', 'H01L', 'H01')"""
    s = re.sub(r"\s+", "", str(s)).upper()
    m = re.match(r"^([A-H]\d{2}[A-Z])(\d+)/(\d+)", s)
    if m:
        return f"{m.group(1)}{m.group(2)}/{m.group(3)}", f"{m.group(1)}{m.group(2)}", m.group(1), m.group(1)[:3]
    m = re.match(r"^([A-H]\d{2}[A-Z])", s)
    if m:
        return "", "", m.group(1), m.group(1)[:3]
    m = re.match(r"^([A-H]\d{2})", s)
    return ("", "", "", m.group(1)) if m else ("", "", "", "")


def _grp_of(code):
    d = code[1:5]
    if d == "0000":
        return code[0]                 # 대분류(A,B,D,E,...)
    if d[2:4] == "00":
        two = d[:2]
        cand = [g for g in G50 if g[:2] == two]
        return cand[0] if len(cand) == 1 else two      # 승격(라벨공간 후보 유일) 또는 중분류 그대로
    return d[:3]                       # 소분류 3자리(세분류 4자리도 3자리로 상한)


_LK = {}


def lookups():
    if _LK:
        return _LK
    cc = pd.read_excel(CCX, sheet_name="전체연계표", dtype=str)
    cc.columns = ["ind", "ipc", "cnt"]
    cc["cnt"] = pd.to_numeric(cc.cnt, errors="coerce").fillna(0)
    cc = cc[cc.ind.str.match(r"^[A-Z]\d{4}$", na=False)].copy()
    cc["grp"] = cc.ind.map(_grp_of)
    ks = cc.ipc.map(ipc_keys)
    cc["sg"], cc["mg"], cc["sc"], cc["cl"] = [k[0] for k in ks], [k[1] for k in ks], [k[2] for k in ks], [k[3] for k in ks]
    for space in ("free", "in"):
        c = cc if space == "free" else cc[cc.grp.isin(G50)]
        tabs = {}
        for lvl in ("sg", "mg", "sc", "cl"):
            g = c[c[lvl] != ""].groupby([lvl, "grp"], as_index=False).cnt.sum()
            tot = g.groupby(lvl).cnt.sum()
            g = g.sort_values(["cnt", "grp"], ascending=[False, True]).drop_duplicates(lvl)
            tabs[lvl] = dict(zip(g[lvl], g.grp)), dict(zip(g[lvl], g.cnt / g[lvl].map(tot)))
        _LK[space] = tabs
    _LK["n_pairs"] = int(len(cc)); _LK["n_ind"] = int(cc.ind.nunique())
    return _LK


def assign(ipcs, space="free"):
    """IPC 문자열 배열 -> (소분류 그룹, 조회 수준) 배열. 미배정은 ''."""
    T = lookups()[space]
    cache = {}
    out_g, out_l = [], []
    for s in ipcs:
        if s is None or pd.isna(s) or s == "":
            out_g.append(""); out_l.append("none"); continue
        if s in cache:
            g, l = cache[s]
        else:
            k = ipc_keys(s); g, l = "", "none"
            for lvl, key in zip(("sg", "mg", "sc", "cl"), k):
                if key and key in T[lvl][0]:
                    g, l = T[lvl][0][key], lvl; break
            cache[s] = (g, l)
        out_g.append(g); out_l.append(l)
    return np.array(out_g, dtype=object), np.array(out_l, dtype=object)


def first_ipc(s):
    if s is None or pd.isna(s):
        return None
    return str(s).split(",")[0].split("|")[0].strip() or None


if __name__ == "__main__":
    L = lookups()
    print("연계표 쌍", L["n_pairs"], "산업코드", L["n_ind"], "G50", len(G50), G50)
    reach = {g: (g in set(L["free"]["sc"][0].values()) | set(L["free"]["sg"][0].values())) for g in G50}
    print("free 배정으로 도달 가능한 소분류", sum(reach.values()), "/", len(G50), [g for g, v in reach.items() if not v])
