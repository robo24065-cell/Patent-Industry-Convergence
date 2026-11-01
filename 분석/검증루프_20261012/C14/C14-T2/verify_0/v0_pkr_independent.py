# -*- coding: utf-8 -*-
r"""C14-T2 verify_0: PKR-IPC 대안 A 독립 재현 — duckdb 미사용, pyarrow+pandas groupby/sort/drop_duplicates로 최빈 계산.
s5_pkr_altA.py와 코드 공유 없음. 정의는 c1_assign.py를 다시 읽고 직접 구현:
 - IPC 표: KR 문서(appl_year<=2019) 제공 라벨, 키(sg/mg/sc)별 문서 수 합 >=3, 최빈(동률은 KSIC 코드 숫자 작은 것)
 - 해외 문서: 등록번호 연결(REG) KR 문서가 있으면 그 KR 제공 라벨 최빈(같은 동률 규칙), 없으면 sg→mg→sc 표
 - 창: is_inv & 2015<=appl_year<=2019, 일치 = 배정==제공(배정 없음은 불일치)"""
import sys, json, time, glob
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.dataset as ds
sys.stdout.reconfigure(encoding="utf-8")
BASE = r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"
ALTD = MC + "/_shared_report/fusion_advantage/c5_R10/alt_assign"
RC = MC + "/T3_기술셀부상예측/ESSENTIAL_R7/rec_conc.parquet"
LINKS = BASE + "/분석/검증루프_20261012/C10/C10-T3/cache/links.parquet"
OUT = BASE + "/분석/검증루프_20261012/C14/C14-T2/verify_0/v0_out.json"
t0 = time.time(); log = lambda *a: print(*a, f"[{time.time()-t0:.0f}s]", flush=True)

def mode_pick(df, keycols):
    """df: keycols + k(문자열 KSIC) + n. 키별 최대 n, 동률은 int(k) 최소."""
    df = df.assign(ki=df["k"].astype(int)).sort_values(keycols + ["n", "ki"], ascending=[True] * len(keycols) + [False, True])
    return df.drop_duplicates(keycols)[keycols + ["k"]]

alt = {}
for cc in ["KR", "US", "JP", "CN", "EP"]:
    t = pq.read_table(f"{ALTD}/alt_{cc}.parquet", columns=["cc", "appl_no", "appl_year", "is_inv", "ksic_prov", "k_pkr", "m_pkr"]).to_pandas()
    alt[cc] = t
log("alt read", {k: len(v) for k, v in alt.items()})
rc = ds.dataset(RC)
need = pd.concat([alt["KR"].loc[alt["KR"].appl_year <= 2019, ["cc", "appl_no"]]] +
                 [alt[c].loc[alt[c].is_inv & alt[c].appl_year.between(2015, 2019), ["cc", "appl_no"]] for c in ["US", "JP", "CN", "EP"]])
import pyarrow as pa
parts = []
for cc in ["KR", "US", "JP", "CN", "EP"]:
    ids = pa.array(need.loc[need.cc == cc, "appl_no"].unique())
    parts.append(rc.to_table(columns=["cc", "appl_no", "sg"], filter=(ds.field("cc") == cc) & ds.field("appl_no").isin(ids)).to_pandas())
sg = need.merge(pd.concat(parts), on=["cc", "appl_no"], how="left")
del rc, parts
log("sg merged", len(sg), sg.sg.isna().mean())
sg["sg"] = sg["sg"].fillna("")
sg["mg"] = sg["sg"].str.split("/").str[0]
sg["sc"] = sg["sg"].str[:4]
kr = alt["KR"]
krk = kr.loc[kr.appl_year <= 2019, ["appl_no", "ksic_prov"]].merge(sg[sg.cc == "KR"], on="appl_no").rename(columns={"ksic_prov": "k"})
tabs = {}
for lvl in ["sg", "mg", "sc"]:
    g = krk[krk[lvl] != ""].groupby([lvl, "k"]).size().rename("n").reset_index()
    tot = g.groupby(lvl).n.sum(); g = g[g[lvl].isin(tot[tot >= 3].index)]
    tabs[lvl] = mode_pick(g, [lvl]).set_index(lvl)["k"]
log("tables", {k: len(v) for k, v in tabs.items()})
lk = pq.read_table(LINKS).to_pandas()
krlab = kr[["appl_no", "ksic_prov"]].rename(columns={"appl_no": "kr_appl_no", "ksic_prov": "k"})
res = {}
for link in ["REG", "R8", "NONE"]:
    if link != "NONE":
        l = lk[lk.link == link].drop_duplicates(["cc", "appl_no", "kr_appl_no"]).merge(krlab, on="kr_appl_no")
        g = l.groupby(["cc", "appl_no", "k"]).size().rename("n").reset_index()
        fam = mode_pick(g, ["cc", "appl_no"]).rename(columns={"k": "k_fam"})
    out = {}
    for cc in ["US", "JP", "CN", "EP"]:
        a = alt[cc]; a = a[a.is_inv & a.appl_year.between(2015, 2019)][["cc", "appl_no", "ksic_prov", "k_pkr"]]
        a = a.merge(sg[sg.cc == cc], on=["cc", "appl_no"], how="left")
        ipc = a["sg"].map(tabs["sg"])
        ipc = ipc.fillna(a["mg"].map(tabs["mg"])).fillna(a["sc"].map(tabs["sc"]))
        if link != "NONE":
            a = a.merge(fam, on=["cc", "appl_no"], how="left"); kfin = a["k_fam"].fillna(pd.Series(ipc.values, index=a.index))
            famshare = float(a["k_fam"].notna().mean())
        else:
            kfin = ipc; famshare = 0.0
        agree = float((kfin == a["ksic_prov"]).mean())
        out[cc] = dict(n=int(len(a)), pkr5=round(agree, 6), famshare=round(famshare, 6),
                       same_as_stored=round(float((kfin.values == a["k_pkr"].values).mean()), 6) if link == "R8" else None)
    res[link] = out; log(link, out)
json.dump(res, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
log("done")
