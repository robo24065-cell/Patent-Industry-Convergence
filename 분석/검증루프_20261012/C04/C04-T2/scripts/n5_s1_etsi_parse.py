# -*- coding: utf-8 -*-
"""C04-T2 N5: ETSI ISLD 원본(zip 안 CSV)을 직접 읽어 (1) KR 행 전부, (2) 전체 분포 집계를 만든다. 기존 trade_05/06 산출물은 쓰지 않는다.
청크별 저장(재개 가능). 출력: cache/etsi/kr_part_XXX.parquet, cache/etsi/agg_part_XXX.parquet, cache/etsi/done.json"""
import os, sys, json, time, zipfile
import pandas as pd
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext\trade_etsi_ISLD-export.zip"
OUT = os.path.join(W, "cache", "etsi"); os.makedirs(OUT, exist_ok=True)
cols = ["IPRD_ID", "IPRD_SIGNATURE_DATE", "Reflected_Date", "COMP_LEGAL_NAME", "DIPG_ID", "Standard", "WI_Type", "TGPP_NUMBER", "ETPR_ACRONYM",
        "Patent_Type", "PATT_APPLICATION_NUMBER", "PUBL_NUMBER", "Country_Of_Registration", "PBPA_APP_DATE", "DIPG_PATF_ID",
        "Original_Application_Number", "Original_Publication_Number", "Normalized_Patent", "Ess_To_Standard", "2G", "3G", "4G", "5G"]
done = json.load(open(os.path.join(OUT, "done.json"))) if os.path.exists(os.path.join(OUT, "done.json")) else {"parts": [], "rows": 0}
z = zipfile.ZipFile(Z)
it = pd.read_csv(z.open("ISLD-export.csv"), sep=";", usecols=cols, dtype=str, chunksize=500_000, encoding="utf-8", encoding_errors="replace", keep_default_na=False)
for i, ch in enumerate(it):
    if i in done["parts"]:
        continue
    ch["cc"] = ch.Country_Of_Registration.str[:2]
    kr = ch[ch.cc == "KR"]
    kr.to_parquet(os.path.join(OUT, f"kr_part_{i:03d}.parquet"), index=False)
    ch["refl_y"] = ch.Reflected_Date.str[:4]; ch["sig_y"] = ch.IPRD_SIGNATURE_DATE.str[:4]
    ch["wi"] = ch.WI_Type.where(ch.WI_Type != "", "NA"); ch["proj"] = ch.ETPR_ACRONYM.str.split("|").str[0].str.strip().where(ch.ETPR_ACRONYM != "", "NA")
    ag = ch.groupby(["cc", "refl_y", "sig_y", "wi", "proj", "COMP_LEGAL_NAME", "Patent_Type"], dropna=False).agg(
        rows=("IPRD_ID", "size"), n_decl=("IPRD_ID", "nunique"), n_dipg=("DIPG_ID", "nunique"), n_fam=("DIPG_PATF_ID", "nunique")).reset_index()
    ag.to_parquet(os.path.join(OUT, f"agg_part_{i:03d}.parquet"), index=False)
    # 특허(DIPG_ID)·패밀리 단위 최초 반영/서명, 기업(전 국가) — 집중도·지연 점검용
    pt = ch.groupby(["DIPG_ID"]).agg(cc=("cc", "first"), fam=("DIPG_PATF_ID", "first"), comp=("COMP_LEGAL_NAME", "first"),
                                     refl=("Reflected_Date", "min"), sig=("IPRD_SIGNATURE_DATE", "min"), n_decl=("IPRD_ID", "nunique")).reset_index()
    pt.to_parquet(os.path.join(OUT, f"dipg_part_{i:03d}.parquet"), index=False)
    done["parts"].append(i); done["rows"] += len(ch)
    json.dump(done, open(os.path.join(OUT, "done.json"), "w"))
    print("part", i, len(ch), "kr", len(kr), round(time.time() - t0), flush=True)
    if time.time() - t0 > 500:
        print("시간 제한 근접 — 재실행으로 이어서", flush=True); sys.exit(3)
done["complete"] = True
json.dump(done, open(os.path.join(OUT, "done.json"), "w"))
print("완료 rows", done["rows"], round(time.time() - t0), flush=True)
