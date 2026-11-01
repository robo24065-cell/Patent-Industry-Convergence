# -*- coding: utf-8 -*-
"""(d2) 수동 판정 기록(작업자가 s06b_evidence.json을 읽고 판정) → mislink_sample.csv, 오연결률 추정 cache/s08_out.json.
판정: S=같은 법인(개명·표기차·스냅샷 시차), U=판정 불가(독립 근거 없음), M=오연결 의심(다른 법인 근거).
근거 코드: NPS=국민연금 사업자번호 앞6자리 사업장명에 옛 이름(또는 두 이름) 존재, DART=DART 법인번호 회사명이 옛 이름과 일치,
          OWN=같은 코드 단독 특허의 등록 당시 권리자명에 두 이름 모두 존재, EN=CA 영문명이 옛 이름/약칭, CORE=이름 핵심어 공유(그룹 접두 변경),
          PUB=공개적으로 알려진 사명 변경(작업자 지식, 독립 자료로 확인 안 함), LAG=등록원부(2026-08)가 CA(2026-07) 뒤 개명을 반영, NAME=이름 일치."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
import pandas as pd, numpy as np
ev = json.load(open(os.path.join(CACHE, "s06b_evidence.json"), encoding="utf-8"))
J = {0: ("S", "NPS"), 2: ("S", "CORE: MAHLE/말레 공통, EN"), 3: ("U", "NPS에 옛 이름 없음"), 4: ("S", "NPS"), 6: ("S", "NPS"), 8: ("S", "NPS"), 9: ("S", "NPS"),
     10: ("S", "PUB(콤텍시스템→아이티센씨티에스)"), 11: ("S", "NPS(홀딩스 전환: 법인격은 같고 영업은 신설법인으로 이동 가능)"), 12: ("S", "NPS"), 13: ("S", "NPS"), 14: ("S", "NPS"),
     18: ("S", "PUB(태광실업→TKG태광)"), 19: ("S", "NPS"), 20: ("S", "OWN"), 23: ("U", "독립 근거 없음"), 24: ("U", "독립 근거 없음"), 26: ("U", "독립 근거 없음(번호 없음)"),
     27: ("S", "EN(KOLED=Korea LED 약칭)"), 28: ("S", "NPS"), 29: ("S", "NPS"), 31: ("S", "NPS(보고인더스트리즈)"), 33: ("U", "독립 근거 없음"), 34: ("U", "독립 근거 없음"),
     36: ("S", "NPS"), 37: ("S", "NPS"), 39: ("U", "OWN에 옛 이름 3개뿐, CA 이름 없음"), 40: ("S", "OWN·NPS"), 43: ("S", "NPS"), 44: ("S", "NPS"), 45: ("S", "DART(법인번호 회사명=씨앤비텍)"),
     46: ("U", "독립 근거 없음"), 47: ("S", "NPS"), 48: ("S", "EN(DONG SUNG A&T)"), 51: ("U", "독립 근거 없음"), 52: ("S", "EN·CORE(SYTS=삼양통상 약칭)"), 55: ("S", "NPS"),
     57: ("S", "OWN"), 61: ("U", "독립 근거 없음"), 62: ("S", "NPS"), 63: ("S", "NPS(우리조명지주)"), 64: ("S", "OWN"),
     65: ("U", "NPS에 두 이름 있으나 이후 '양도'로 밥스트코리아에 이전 → 별개 법인 가능"), 66: ("U", "독립 근거 없음"), 67: ("S", "DART·NPS"),
     68: ("S", "PUB(STX중공업→HD현대마린엔진)"), 69: ("U", "독립 근거 없음"), 71: ("S", "CORE(일렘, TKG 그룹 접두)"), 74: ("S", "NPS"), 75: ("U", "독립 근거 없음"),
     78: ("S", "NPS"), 80: ("S", "EN·NPS"), 81: ("S", "OWN"), 83: ("U", "NPS에 두 이름 있으나 이후 '일부양도'로 우암건설에 이전 → 별개 법인 가능"), 85: ("U", "독립 근거 없음"),
     86: ("S", "NPS"), 87: ("S", "DART·NPS"), 88: ("S", "CORE(소모)·NPS"), 90: ("S", "NPS"), 91: ("S", "NPS"), 92: ("U", "독립 근거 없음"), 94: ("U", "CORE(이에스) 약함, 독립 근거 없음"),
     95: ("U", "독립 근거 없음"), 99: ("S", "NPS"), 100: ("U", "독립 근거 없음"), 101: ("S", "NPS"), 103: ("S", "NPS"), 104: ("S", "NPS"), 105: ("S", "NPS"), 107: ("S", "NPS"),
     108: ("S", "NPS"), 109: ("S", "NPS"), 112: ("S", "NPS"), 114: ("U", "CORE(신록)뿐, 계열사일 수 있음"), 116: ("S", "NPS"), 117: ("U", "독립 근거 없음"), 118: ("U", "독립 근거 없음"),
     120: ("S", "NPS"), 124: ("U", "독립 근거 없음"), 128: ("S", "NPS"), 129: ("U", "독립 근거 없음"), 130: ("S", "NPS"), 133: ("S", "NPS"), 134: ("S", "EN·NPS"), 136: ("S", "NPS"),
     139: ("S", "NPS"), 140: ("S", "NPS"), 142: ("S", "NPS"), 144: ("S", "NPS"), 145: ("S", "NPS"), 149: ("S", "NPS"), 151: ("S", "NPS"), 153: ("U", "독립 근거 없음"),
     155: ("S", "NPS(분할 존속법인 개명, 신설법인이 옛 이름 사용)"), 156: ("S", "NPS"), 159: ("S", "NPS"), 160: ("S", "NPS"), 167: ("S", "NPS"), 171: ("S", "NPS"), 172: ("S", "NPS(에스케이마이크로웍스)"),
     # 쌍 ①(등록원부 현재 이름) 하위 25: 모두 스냅샷 시차 개명
     1: ("S", "LAG·EN·DART"), 15: ("S", "LAG·NPS·DART"), 17: ("S", "LAG·OWN·NPS"), 21: ("S", "LAG·NPS"), 25: ("S", "LAG·EN(KS AUTOSYS)"), 42: ("S", "LAG·EN·OWN"), 50: ("S", "LAG·OWN·NPS"),
     53: ("S", "LAG·EN(SJ ENERTEC)·DART"), 54: ("S", "LAG·DART·NPS"), 58: ("S", "LAG·NPS·DART"), 84: ("S", "LAG·OWN·DART"), 93: ("S", "LAG·OWN·NPS"), 96: ("S", "LAG·OWN"),
     110: ("S", "LAG·NPS"), 113: ("S", "LAG·EN(Tobaek)"), 121: ("S", "LAG·NPS"), 137: ("S", "LAG·OWN·NPS"), 141: ("S", "LAG·NPS"), 161: ("S", "LAG·NPS"), 162: ("S", "LAG·NPS"),
     163: ("S", "LAG·EN(HANA AI)·NPS"), 166: ("S", "LAG·EN(YEON)·NPS"), 169: ("S", "LAG·NPS"), 173: ("S", "LAG·EN·NPS"), 174: ("S", "LAG·NPS")}
rows = []
for i, e in enumerate(ev):
    lab, basis = J.get(i, ("S", "NAME") if "high" in e["strata"] else (None, None))
    if lab is None:
        raise SystemExit(f"판정 누락 #{i} {e['strata']}")
    old = [x[0] for x in e["own0_names"]][:2] + [x[0] for x in e["prov_names"]][:1]
    rows.append(dict(idx=i, strata="|".join(e["strata"]), cd_masked=e["cd"][:5] + "****" + e["cd"][-3:], ca_name=e["ca_name"], ca_ename=e["ca_ename"],
                     reg_name_current=e["reg_name_current"], old_names="; ".join(dict.fromkeys(old)), dart_name=e["dart"], nps_names="; ".join(e["nps_top"][:2]),
                     s_own0=e["s_own0"], s_prov=e["s_prov"], s1=e["s1"], n_pat=e["n_pat"], years=f"{e['years'][0]}-{e['years'][1]}",
                     nts=(e["nts"] or {}).get("b_stt"), judgment=lab, basis=basis))
R = pd.DataFrame(rows)
R.to_csv(guard_path(os.path.join(W, "mislink_sample.csv")), index=False, encoding="utf-8-sig")
P = pd.read_parquet(os.path.join(CACHE, "s06b_pairs.parquet"))
O = json.load(open(os.path.join(CACHE, "s06b_out.json"), encoding="utf-8"))
low = R[R.strata.str.contains(r"(^|\|)low($|\|)")]
high = R[R.strata.str.contains("high")]
reg1 = R[R.strata.str.contains("low_regname")]
res = {}
for nm, g in [("low", low), ("high", high), ("low_regname", reg1)]:
    res[nm] = g.judgment.value_counts().to_dict(); res[nm + "_n"] = len(g)
pl = O["low_codes"] / O["s_indep"]["n"]
mU = (low.judgment.isin(["M", "U"])).mean(); mM = (low.judgment == "M").mean()
ph = (high.judgment.isin(["M", "U"])).mean()
res["low_share_scored"] = pl
res["est_mislink_lower(M만)"] = pl * mM
res["est_mislink_upper(M+U)"] = pl * mU
res["est_mislink_worst(하위 층 전부)"] = pl
res["high_stratum_MU_rate"] = ph
res["patent_weighted_low_share"] = O["low_share_patents"]
res["est_mislink_upper_patent_weighted"] = O["low_share_patents"] * mU
res["unscored_codes(독립 이름 없음)"] = O["codes_no_indep_name"]
res["verdict"] = "수정 필요" if pl * mU > 0.05 else "영향 없음(추정 오연결률 ≤5%)"
json.dump(res, open(guard_path(os.path.join(CACHE, "s08_out.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=0, default=float))
