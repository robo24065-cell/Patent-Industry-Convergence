# -*- coding: utf-8 -*-
"""I5 '표준특허(SEP) 선언 예측 — 표준화 레이더' 학습표. 단위 = KSIC 26429(기타 무선통신장비) KR 출원(출원 t0-5..t0, t0까지 공개),
t0까지 ETSI 선언 미반영. 결과 = ETSI DB 최초 반영일(first_refl)이 t0+1..t0+2.
🔒 최종시험 t0=2022(반영 2023–24, F=2023) → 개발은 t0 ≤ 2020(반영 ≤2022)만 레이블 생성.
해외 패밀리: 제공 US·EP·JP 기록의 family 토큰으로 KR 출원과 연결(신원만 사용, 시점은 해외 공개일·반영일)."""
import os, time, sqlite3
import duckdb, pandas as pd, numpy as np
from i00_common import *

t = time.time()
def log(*a):
    print(*a, round(time.time() - t), flush=True)

SEAL_I5_REFL_YEAR = 2023
KS5 = "26429"
L = pd.read_parquet(os.path.join(EXT, "trade_etsi_sep_link.parquet"))
L["refl"] = L.first_refl.dt.strftime("%Y%m%d").astype(float)
Lkr = L[L.cc == "KR"].groupby("appl_no").refl.min()
Lfo = L[L.cc.isin(["US", "EP", "JP"])].copy()
Lfo["fkey"] = Lfo.cc + Lfo.appl_no.astype(str).str.split(".").str[0]
Lfo = Lfo.groupby("fkey").refl.min()

k = pd.read_parquet(os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet"),
                    columns=["출원번호", "출원연도", "등록번호", "공개번호", "공개일자", "등록일자", "출원인명", "메인IPC", "세세분류코드"])
k.columns = ["appl_no", "ay", "rg", "pb", "pub", "rgd", "names", "ipc", "ksic"]
k = k[k.ksic.astype(str) == KS5].copy()
k["appl_no"] = k.appl_no.astype(str)
for c in ["pub", "rgd"]:
    k[c] = pd.to_numeric(k[c].astype(str).str.replace(r"\D", "", regex=True).str[:8], errors="coerce")
k["known"] = k[["pub", "rgd"]].min(axis=1)
k["app"] = k.names.fillna("").str.split("|").str[0].map(norm)
k["mg"] = k.ipc.fillna("").str.split("/").str[0].str.replace(" ", "")      # IPC 메인그룹
k["refl"] = k.appl_no.map(Lkr)
# 토큰
tok = {}
for c, f in [("appl_no", lambda s: "KR" + s), ("rg", lambda s: "KR" + s.str[:9]), ("pb", lambda s: "KR" + s.str[2:])]:
    s = k[c].fillna("").astype(str).str.replace(r"\D", "", regex=True)
    ok = s.str.len() >= 9
    tok.update(dict(zip(f(s[ok]), k.appl_no[ok])))
meta = os.path.join(MC, "_shared", "meta", "*.parquet").replace("\\", "/")
fo = duckdb.sql(f"select distinct cc, appl_no, pub_date, family from read_parquet('{meta}') where cc in ('US','EP','JP') and ksic='{KS5}' and family like '%KR%'").df()
fo["fkey"] = fo.cc + fo.appl_no.astype(str)
fo["fknown"] = pd.to_numeric(fo.pub_date, errors="coerce")
fo["frefl"] = fo.fkey.map(Lfo)
fo["tok"] = fo.family.fillna("").str.split("|")
ex = fo[["fkey", "cc", "fknown", "frefl", "tok"]].explode("tok")
ex = ex[ex.tok.str.startswith("KR", na=False)]
ex["kr"] = ex.tok.map(tok)
ex = ex.dropna(subset=["kr"]).drop_duplicates(["fkey", "kr"])
log("KR 26429 출원", len(k), "해외기록", len(fo), "연결쌍", len(ex), "해외 선언 매칭", int(ex.frefl.notna().sum()))
c = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'registry_full.sqlite')}?mode=ro", uri=True)
cl = pd.read_sql(f"select applNo appl_no, claimCount from patent where ksic='{KS5}'", c)
k = k.merge(cl.drop_duplicates("appl_no").assign(claimCount=lambda d: pd.to_numeric(d.claimCount, errors="coerce")), on="appl_no", how="left")


def sm_rate(num, den, prior, a=10):
    return (num + a * prior) / (den + a)


rows = []
for t0 in range(2012, 2021):
    assert t0 + 2 < SEAL_I5_REFL_YEAR
    end = t0 * 10000 + 1231
    kn = k[k.known <= end]
    dec = kn.refl <= end
    prior = dec.mean()
    ga = kn.assign(d=dec).groupby("app").agg(n=("d", "size"), d=("d", "sum"))
    gm = kn.assign(d=dec).groupby("mg").agg(n=("d", "size"), d=("d", "sum"))
    rec = kn[(kn.refl > (t0 - 2) * 10000 + 1231) & (kn.refl <= end)]
    U = kn[kn.ay.between(t0 - 5, t0) & ~dec].copy()
    U["app_age"] = t0 - U.ay
    U["reg_t0"] = (U.rgd <= end).astype(float)
    U["claims_t0"] = U.claimCount.where(U.rgd <= end)
    U["app_rate"] = sm_rate(U.app.map(ga.d).fillna(0), U.app.map(ga.n).fillna(0), prior)
    U["app_logn"] = np.log1p(U.app.map(ga.n).fillna(0))
    U["mg_rate"] = sm_rate(U.mg.map(gm.d).fillna(0), U.mg.map(gm.n).fillna(0), prior)
    U["mg_logn"] = np.log1p(U.mg.map(gm.n).fillna(0))
    # 5개국 패밀리(t0까지 공개된 해외 기록)
    e = ex[ex.fknown <= end]
    for cc in ["US", "EP", "JP"]:
        U["fam_" + cc] = U.appl_no.isin(set(e.loc[e.cc == cc, "kr"])).astype(float)
    U["fam_n"] = U.appl_no.map(e.groupby("kr").fkey.nunique()).fillna(0)
    # ETSI × 패밀리 전이 + 선언 동향
    U["fam_declared"] = U.appl_no.isin(set(ex.loc[ex.frefl <= end, "kr"])).astype(float)
    U["app_recent_dec"] = np.log1p(U.app.map(rec.groupby("app").size()).fillna(0))
    U["mg_recent_dec"] = np.log1p(U.mg.map(rec.groupby("mg").size()).fillna(0))
    U["y"] = ((U.refl > end) & (U.refl <= (t0 + 2) * 10000 + 1231)).astype(int)
    U["t0"] = t0
    rows.append(U.drop(columns=["names", "rg", "pb"]))
    log(t0, len(U), int(U.y.sum()), round(U.y.mean(), 4), "fam_declared", int(U.fam_declared.sum()))
D = pd.concat(rows, ignore_index=True)
D.to_parquet(os.path.join(CACHE, "I5_panel.parquet"), index=False)
