# -*- coding: utf-8 -*-
r"""C07-T1: s3·s3b·s4 결과를 sensitivity_313.csv로 모으고 PREREG 판정을 계산한다."""
import os, sys, json, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
H = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C07\C07-T1"
R = {v: json.load(open(os.path.join(H, rf"cache\s3_{v}.json"), encoding="utf-8")) for v in ("B0", "V1a", "V1a_x", "V2a")}
rows = []
b = R["B0"]
desc = {"V1a": "끝 창 2015–16(파일 값, ÷7; FE 둘째 창 2012–13→2015–16 ÷3), 특허 창도 같이 이동",
        "V1a_x": "끝 창 2015–16을 KOSIS 쪽에만 적용(특허 창은 원판)", "V2a": "KSIC 9→10 비-1:1 코드 12개 중 34개 집합에 든 10개 제외(22개 코드·8그룹·44관측)"}
for v in ("V1a", "V1a_x", "V2a"):
    r = R[v]
    for k in ("26211", "26212", "26111", "29280"):
        for nm in ("출하액", "종사자수", "사업체수"):
            o, x = b["cases"][k][nm], r["cases"][k][nm]
            inset = r["cases"][k]["in_K34"]
            rows.append({"variant": v, "variant_desc": desc[v], "metric": f"SP-CASE-{k} KOSIS {nm} 연성장%", "orig": o, "var": x, "orig_p": None, "var_p": None,
                         "sign_flip": bool(np.sign(o) != np.sign(x)), "p_cross_0.05": None, "abs_diff": round(abs(x - o), 3),
                         "note": "" if inset else "V2a에서는 34개 분석 집합 밖(9차에 없는 분할 코드). 값 자체는 10차 소급 공식값으로 원판과 같다"})
    for key, lab in (("VAR_x_g_출하액", "SP-VAR 출하 그룹 안 분산 비중"), ("VAR_x_g_사업체수", "SP-VAR 사업체 그룹 안 분산 비중"), ("VAR_x_g_종사자수", "SP-VAR 종사자 그룹 안 분산 비중(원고 미인용)")):
        o, x = b[key], r[key]
        rows.append({"variant": v, "variant_desc": desc[v], "metric": lab, "orig": o[0], "var": x[0], "orig_p": o[2], "var_p": x[2], "sign_flip": None,
                     "p_cross_0.05": bool((o[2] < 0.05) != (x[2] < 0.05)), "abs_diff": round(abs(x[0] - o[0]), 4), "note": f"무작위 기대 {o[1]}→{x[1]}, n {o[3]}→{x[3]}"})
    for key, lab in (("FE_사업체수", "SP-FE 사업체 r"), ("FE_출하액", "SP-FE 출하 r"), ("FE_종사자수", "SP-FE 종사자 r")):
        o, x = b[key], r[key]
        rows.append({"variant": v, "variant_desc": desc[v], "metric": lab, "orig": o[0], "var": x[0], "orig_p": o[1], "var_p": x[1], "sign_flip": bool(np.sign(o[0]) != np.sign(x[0])),
                     "p_cross_0.05": bool((o[1] < 0.05) != (x[1] < 0.05)), "abs_diff": round(abs(x[0] - o[0]), 4), "note": f"n {o[2]}→{x[2]}"})
# R8-P2
P = {v: json.load(open(os.path.join(H, rf"cache\p2_{v}_results.json"), encoding="utf-8")) for v in ("B0", "V1", "V2a")}
pd2 = {"V1": "결과 창이 2016을 넘지 않는 원점(2009·2011·2012·2013)만", "V2a": "비-1:1 코드 제외(22개 단위)"}
for v in ("V1", "V2a"):
    for o in ("ship",):
        for m in ("sig_TXT", "sig_PROV_KR"):
            rows.append({"variant": "P2-" + v, "variant_desc": pd2[v], "metric": f"R8-P2 출하 몫 AUC {m}", "orig": round(P["B0"][o]["auc_mean"][m], 4), "var": round(P[v][o]["auc_mean"][m], 4),
                         "orig_p": None, "var_p": None, "sign_flip": None, "p_cross_0.05": None, "abs_diff": round(abs(P[v][o]["auc_mean"][m] - P["B0"][o]["auc_mean"][m]), 4), "note": "보고 항목(판정 대상 아님)"})
        c0, c1 = P["B0"][o]["compare"]["sig_PROV_KR-sig_TXT"], P[v][o]["compare"]["sig_PROV_KR-sig_TXT"]
        rows.append({"variant": "P2-" + v, "variant_desc": pd2[v], "metric": "R8-P2 출하 몫 AUC 제공KR−TXT", "orig": round(c0["mean"], 4), "var": round(c1["mean"], 4),
                     "orig_p": None, "var_p": None, "sign_flip": bool(np.sign(c0["mean"]) != np.sign(c1["mean"])), "p_cross_0.05": None, "abs_diff": round(abs(c1["mean"] - c0["mean"]), 4),
                     "note": f"CI {[round(x, 3) for x in c0['ci']]} {c0['pos']} → {[round(x, 3) for x in c1['ci']]} {c1['pos']}; CI가 0 포함으로 바뀜={bool(c1['ci'][0] < 0 < c1['ci'][1]) and not (c0['ci'][0] < 0 < c0['ci'][1])}"})
# V3 실질
RR = pd.read_csv(os.path.join(H, r"cache\s3b_real.csv"), dtype={"ksic": str})
for _, x in RR.iterrows():
    rows.append({"variant": "V3", "variant_desc": f"출하 실질화(ECOS {x.deflator}), 끝 창 {x.end_window}", "metric": f"SP-CASE-{x.ksic} KOSIS 출하 연성장%(실질)",
                 "orig": x["nominal_g%"], "var": x["real_g%"], "orig_p": None, "var_p": None, "sign_flip": bool(np.sign(x["nominal_g%"]) != np.sign(x["real_g%"])),
                 "p_cross_0.05": None, "abs_diff": round(abs(x["real_g%"] - x["nominal_g%"]), 2), "note": f"물가 연성장 {x['ppi_g%']}%; 보조(판정 대상 아님)"})
S = pd.DataFrame(rows)
S.to_csv(os.path.join(H, "sensitivity_313.csv"), index=False, encoding="utf-8-sig")
# 판정
verd = {}
for v in ("V1a", "V1a_x", "V2a"):
    q = S[S.variant == v]
    case = q[q.metric.str.startswith("SP-CASE")]
    case_in = case[case.note == ""]
    flips = case_in[case_in.sign_flip == True].metric.tolist()
    pcross = q[q["p_cross_0.05"] == True].metric.tolist()
    pcross = [m for m in pcross if "원고 미인용" not in m]
    big = (case_in.abs_diff >= 0.1).any() or (q[q.metric.str.contains("SP-FE")].abs_diff >= 0.01).any() or (q[q.metric.str.contains("SP-VAR") & ~q.metric.str.contains("미인용")].abs_diff >= 0.005).any()
    verd[v] = {"판정": "수정 필요" if (flips or pcross) else ("단서 필요" if big else "영향 없음"), "부호 뒤집힘": flips, "p 0.05 건넘": pcross,
               "34개 집합 밖으로 빠진 사례": case[case.note != ""].metric.str.extract(r"SP-CASE-(\d+)")[0].unique().tolist()}
print(json.dumps(verd, ensure_ascii=False, indent=1))
json.dump(verd, open(os.path.join(H, r"cache\s6_verdict.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
