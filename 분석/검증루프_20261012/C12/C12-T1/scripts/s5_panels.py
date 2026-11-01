# -*- coding: utf-8 -*-
"""C12-T1 s5: 시나리오별 패널(학습 t0 2012–2020 + 시험 t0 2022) 생성. 동결 n5lib.build_panel을 그대로 쓰고, 기초표 k의 레이블(refl)·모집단만 바꾼다.
n5lib CACHE는 이 폴더 cache/n5로 돌린다(트랙 cache 미사용). 사용: python s5_panels.py <scen> [t0 ...]
  off : 공식(trade_etsi_sep_link 기반 load_kr 그대로)
  s1  : refl = C12 완전 키(엄격 ∪ 검증 통과 확장, lvl≤2) 최초 반영일
  s4  : 2011 일괄 이관분(반영 2011·서명 <2011)만으로 선언된 26429 출원을 k에서 제거(엄격 키 기준 = 공식 레이블 규칙)
  s3  : 모집단 확장(정보용) — KSIC ∈ EXP_KS 이고 메인 IPC 서브클래스 ∈ 26429 핵심 집합
산출: cache/panels/<scen>_<t0>.parquet (있으면 건너뜀)"""
import os, sys, json, time
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import n5lib
n5lib.CACHE = os.path.join(W, "cache", "n5"); os.makedirs(n5lib.CACHE, exist_ok=True)
PD = os.path.join(W, "cache", "panels"); os.makedirs(PD, exist_ok=True)
t = time.time()
scen = sys.argv[1]
t0s = [int(x) for x in sys.argv[2:]] or list(range(2012, 2021)) + [2022]
EXP_KS = ["26429", "26422", "26529", "27212", "27211", "26519", "26323"]   # 26429 + 연결 KR 선언 ≥20건 KSIC(link_bias.csv c3_ksic_dist)


def strict_full_refl():
    M = pd.read_parquet(os.path.join(W, "cache", "link_matches.parquet"), columns=["appl_no", "refl", "strict", "lvl", "sig"])
    M = M[M.lvl <= 2]
    return M[M.strict == 1].groupby("appl_no").refl.min(), M.groupby("appl_no").refl.min()


k = n5lib.load_kr("26429"); ex = n5lib.load_family(k, "26429"); E = n5lib.load_ext("26429")
if scen == "s1":
    _, rf = strict_full_refl()
    k = k.copy(); k["refl_off"] = k.refl; k["refl"] = k.appl_no.map(rf).astype(float)
    print("s1 refl 변경 출원", int(((k.refl.fillna(0) != k.refl_off.fillna(0))).sum()), "새로 선언", int((k.refl.notna() & k.refl_off.isna()).sum()),
          "더 이른 반영", int((k.refl < k.refl_off).sum()), flush=True)
    k = k.drop(columns=["refl_off"])
elif scen == "s4":
    M = pd.read_parquet(os.path.join(W, "cache", "link_matches.parquet"), columns=["appl_no", "refl", "sig", "strict", "lvl", "ksic"])
    M = M[(M.strict == 1) & (M.ksic == "26429")]
    M["bulk"] = (M.refl // 10000 == 2011) & (M.sig < 20110101)
    b = M.groupby("appl_no").bulk.all()
    drop = set(b[b].index)
    print("s4 제거(이관분만으로 선언)", len(drop), "k 안", int(k.appl_no.isin(drop).sum()), flush=True)
    k = k[~k.appl_no.isin(drop)].copy()
elif scen == "s4b":   # 정보용(변경 기록 2): 이관 선언 기록만 지우고 남은 선언으로 최초 반영일 재계산(엄격 키)
    M = pd.read_parquet(os.path.join(W, "cache", "link_matches.parquet"), columns=["appl_no", "refl", "sig", "strict", "lvl", "ksic"])
    M = M[(M.strict == 1) & (M.ksic == "26429")]
    M = M[~((M.refl // 10000 == 2011) & (M.sig < 20110101))]
    rf = M.groupby("appl_no").refl.min()
    k = k.copy(); old = k.refl.copy(); k["refl"] = k.appl_no.map(rf).astype(float)
    print("s4b refl 변경", int((k.refl.fillna(0) != old.fillna(0)).sum()), "선언 사라짐", int((k.refl.isna() & old.notna()).sum()), flush=True)
elif scen == "s3":
    core = json.load(open(os.path.join(W, "cache", "ipc_core.json")))["core"]
    ks_, exs = [], []
    Es = {"ros": [], "ntis": []}
    for ks in EXP_KS:
        kk = n5lib.load_kr(ks); ks_.append(kk)
        exs.append(n5lib.load_family(kk, ks))
        e = n5lib.load_ext(ks); Es["ros"].append(e["ros"]); Es["ntis"].append(e["ntis"])
        n5lib.tlog(t, "load", ks, len(kk))
    k = pd.concat(ks_, ignore_index=True)
    k["sc"] = k.ipc_sc if "ipc_sc" in k else k.sg.str[:4]
    k = k[k.sc.isin(core)].drop(columns=["sc"]).drop_duplicates("appl_no").copy()
    ex = pd.concat(exs, ignore_index=True).drop_duplicates(["fkey", "kr"])
    E = dict(E); E["ros"] = pd.concat(Es["ros"], ignore_index=True).drop_duplicates(); E["ntis"] = pd.concat(Es["ntis"], ignore_index=True).drop_duplicates()
    print("s3 k", len(k), k.ksic.value_counts().to_dict(), "선언", int(k.refl.notna().sum()), flush=True)
elif scen != "off":
    raise SystemExit("unknown scen")
for t0 in t0s:
    p = os.path.join(PD, f"{scen}_{t0}.parquet")
    if os.path.exists(p):
        continue
    U = n5lib.build_panel(t0, "26429", "final" if t0 == 2022 else "dev", k, ex, E)
    U.to_parquet(p, index=False)
    n5lib.tlog(t, scen, t0, len(U), int(U.y.sum()))
    if time.time() - t > 450:
        print("시간 제한 근접 — 재실행으로 이어서", flush=True); sys.exit(3)
print("done", scen, round(time.time() - t), flush=True)
