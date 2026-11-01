# -*- coding: utf-8 -*-
"""C12-T1 s1b: 세대 플래그 정정 재집계.
s1/s4 1차 실행에서 2G–5G 열을 '빈칸 아님(!= "")'으로 읽었는데, 원본 값은 '0'/'1'/'' 이어서 '0'도 1로 셌다(5G=4G 비율이 같게 나온 원인).
여기서는 값 == '1'만 해당 세대로 본다. 또 프로젝트(ETPR_ACRONYM)·규격(Standard)으로 3GPP 셀룰러 여부를 정규식으로 분류한다.
출력: cache/etsi2/famgen_XXX.parquet(패밀리×기업 최초 반영: 전체·5G·4G, 셀룰러, KR 구성원), cache/etsi2/agg2_XXX.parquet, done.json"""
import os, sys, json, time, zipfile, re
import pandas as pd
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext\trade_etsi_ISLD-export.zip"
OUT = os.path.join(W, "cache", "etsi2"); os.makedirs(OUT, exist_ok=True)
cols = ["IPRD_ID", "Reflected_Date", "IPRD_SIGNATURE_DATE", "COMP_LEGAL_NAME", "DIPG_ID", "Standard", "ETPR_ACRONYM", "Country_Of_Registration",
        "DIPG_PATF_ID", "Ess_To_Standard", "2G", "3G", "4G", "5G"]
CELL = re.compile(r"3GPP|UMTS|LTE|GSM|GPRS|EDGE|E-?UTRA|UTRA|\bNR\b|NEW RADIO|5G|HSPA|HSDPA|HSUPA|WCDMA|W-CDMA|TD-SCDMA|IMT|GERAN|NB-?IOT|EPS|EPC|IMS|AMR|MCPTT|V2X|RELEASE", re.I)
SPEC3 = re.compile(r"\b(TS|TR)\s*(2[1-9]|3[0-8]|4[1-9]|5[0-5])\.\d{3}", re.I)   # 3GPP 규격 번호 체계(21–38 계열, 41–55)
dp = os.path.join(OUT, "done.json")
done = json.load(open(dp)) if os.path.exists(dp) else {"parts": [], "rows": 0}
z = zipfile.ZipFile(Z)
it = pd.read_csv(z.open("ISLD-export.csv"), sep=";", usecols=cols, dtype=str, chunksize=500_000, encoding="utf-8", encoding_errors="replace", keep_default_na=False)
for i, ch in enumerate(it):
    if i in done["parts"]:
        continue
    ch["cc"] = ch.Country_Of_Registration.str[:2]
    ch["refl_d"] = ch.Reflected_Date.str[:10]
    for g in ["2G", "3G", "4G", "5G"]:
        ch["g" + g[0]] = (ch[g] == "1").astype("int8")
    ch["gblank"] = ((ch["2G"] == "") & (ch["3G"] == "") & (ch["4G"] == "") & (ch["5G"] == "")).astype("int8")
    ch["cell"] = (ch.ETPR_ACRONYM.str.contains(CELL) | ch.Standard.str.contains(SPEC3)).astype("int8")
    ch["ess"] = (ch.Ess_To_Standard == "1").astype("int8")
    ch["gen"] = ch.g5.astype(str) + ch.g4.astype(str) + ch.g3.astype(str) + ch.g2.astype(str)
    ch.loc[ch.gblank == 1, "gen"] = "blank"
    ag = ch.groupby(["cc", ch.refl_d.str[:4].rename("refl_y"), "gen", "cell"]).agg(rows=("IPRD_ID", "size"), n_fam=("DIPG_PATF_ID", "nunique")).reset_index()
    ag.to_parquet(os.path.join(OUT, f"agg2_{i:03d}.parquet"), index=False)
    ch["refl5"] = ch.refl_d.where(ch.g5 == 1, "9999"); ch["refl4"] = ch.refl_d.where(ch.g4 == 1, "9999")
    ch["reflc"] = ch.refl_d.where(ch.cell == 1, "9999")
    ch["iskr"] = (ch.cc == "KR").astype("int8")
    fm = ch.groupby(["DIPG_PATF_ID", "COMP_LEGAL_NAME"]).agg(refl=("refl_d", "min"), refl5=("refl5", "min"), refl4=("refl4", "min"), reflc=("reflc", "min"),
                                                          kr=("iskr", "max"), ess=("ess", "max"), gblank=("gblank", "min")).reset_index()
    fm.to_parquet(os.path.join(OUT, f"famgen_{i:03d}.parquet"), index=False)
    done["parts"].append(i); done["rows"] += len(ch)
    json.dump(done, open(dp, "w"))
    print("part", i, len(ch), round(time.time() - t0), flush=True)
    if time.time() - t0 > 480:
        print("시간 제한 근접 — 재실행으로 이어서", flush=True); sys.exit(3)
done["complete"] = True
json.dump(done, open(dp, "w"))
print("완료 rows", done["rows"], round(time.time() - t0), flush=True)
