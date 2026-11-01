# -*- coding: utf-8 -*-
"""C12-T1 s1: ETSI ISLD 원본 zip을 직접 읽어 (1) KR 행 전부, (2) 분포 집계, (3) 패밀리×기업 세대별 최초 반영일, (4) DIPG 단위 요약을 만든다.
기존 trade_05/06 산출물·C04-T2 캐시는 쓰지 않는다. 청크 단위 저장, 재실행 시 이어서."""
import os, sys, json, time, zipfile
import pandas as pd
t0 = time.time()
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext\trade_etsi_ISLD-export.zip"
OUT = os.path.join(W, "cache", "etsi"); os.makedirs(OUT, exist_ok=True)
cols = ["IPRD_ID", "IPRD_REFERENCE", "IPRD_SIGNATURE_DATE", "Reflected_Date", "COMP_LEGAL_NAME", "DIPG_ID", "Standard", "WI_Type", "ETPR_ACRONYM",
        "Patent_Type", "PATT_APPLICATION_NUMBER", "PUBL_NUMBER", "Country_Of_Registration", "PBPA_APP_DATE", "PBPA_PRIORITY_NUMBERS", "DIPG_PATF_ID",
        "Original_Application_Number", "Original_Publication_Number", "Normalized_Patent", "Ess_To_Standard", "2G", "3G", "4G", "5G"]
dp = os.path.join(OUT, "done.json")
done = json.load(open(dp)) if os.path.exists(dp) else {"parts": [], "rows": 0}
z = zipfile.ZipFile(Z)
info = z.getinfo("ISLD-export.csv")
done["csv_bytes"] = info.file_size; done["zip_time"] = "%04d-%02d-%02d %02d:%02d" % info.date_time[:5]
it = pd.read_csv(z.open("ISLD-export.csv"), sep=";", usecols=cols, dtype=str, chunksize=500_000, encoding="utf-8", encoding_errors="replace", keep_default_na=False)
for i, ch in enumerate(it):
    if i in done["parts"]:
        continue
    ch["cc"] = ch.Country_Of_Registration.str[:2]
    ch[ch.cc == "KR"].to_parquet(os.path.join(OUT, f"kr_{i:03d}.parquet"), index=False)
    ch["refl_d"] = ch.Reflected_Date.str[:10]; ch["sig_d"] = ch.IPRD_SIGNATURE_DATE.str[:10]
    ch["refl_y"] = ch.refl_d.str[:4]; ch["sig_y"] = ch.sig_d.str[:4]
    ch["proj"] = ch.ETPR_ACRONYM.str.split("|").str[0].str.strip().where(ch.ETPR_ACRONYM != "", "NA")
    for g in ["2G", "3G", "4G", "5G"]:
        ch["g" + g[0]] = (ch[g] != "").astype("int8")
    ch["gen"] = ch.g5.astype(str) + ch.g4.astype(str) + ch.g3.astype(str) + ch.g2.astype(str)
    ch["rb"] = (ch.refl_d == "").astype("int32"); ch["sb"] = (ch.sig_d == "").astype("int32")
    ag = ch.groupby(["cc", "refl_y", "sig_y", "proj", "gen", "COMP_LEGAL_NAME"], dropna=False).agg(
        rows=("IPRD_ID", "size"), n_decl=("IPRD_ID", "nunique"), n_dipg=("DIPG_ID", "nunique"), n_fam=("DIPG_PATF_ID", "nunique"),
        refl_blank=("rb", "sum"), sig_blank=("sb", "sum")).reset_index()
    ag.to_parquet(os.path.join(OUT, f"agg_{i:03d}.parquet"), index=False)
    # 패밀리 × 기업: 최초 반영일(전체·5G·4G), KR 구성원 여부
    ch["refl5"] = ch.refl_d.where(ch.g5 == 1, "9999"); ch["refl4"] = ch.refl_d.where(ch.g4 == 1, "9999")
    ch["iskr"] = (ch.cc == "KR").astype("int8")
    fm = ch.groupby(["DIPG_PATF_ID", "COMP_LEGAL_NAME"]).agg(refl=("refl_d", "min"), refl5=("refl5", "min"), refl4=("refl4", "min"),
                                                          sig=("sig_d", "min"), kr=("iskr", "max"), n_rows=("IPRD_ID", "size")).reset_index()
    fm.to_parquet(os.path.join(OUT, f"fam_{i:03d}.parquet"), index=False)
    # DIPG(선언 안 특허 묶음) 단위: 반영일 개수·최소·최대(사후 수정·재선언 흔적), 서명, 선언 수
    dg = ch.groupby("DIPG_ID").agg(cc=("cc", "first"), fam=("DIPG_PATF_ID", "first"), comp=("COMP_LEGAL_NAME", "first"),
                                  refl_min=("refl_d", "min"), refl_max=("refl_d", "max"), n_refl=("refl_d", "nunique"),
                                  sig_min=("sig_d", "min"), n_decl=("IPRD_ID", "nunique"), n_rows=("IPRD_ID", "size")).reset_index()
    dg.to_parquet(os.path.join(OUT, f"dipg_{i:03d}.parquet"), index=False)
    done["parts"].append(i); done["rows"] += len(ch)
    json.dump(done, open(dp, "w"))
    print("part", i, len(ch), round(time.time() - t0), flush=True)
    if time.time() - t0 > 480:
        print("시간 제한 근접 — 재실행으로 이어서", flush=True); sys.exit(3)
done["complete"] = True
json.dump(done, open(dp, "w"))
print("완료 rows", done["rows"], round(time.time() - t0), flush=True)
