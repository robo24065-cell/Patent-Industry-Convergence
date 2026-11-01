# -*- coding: utf-8 -*-
"""PROTOCOL_R7 §3 외부 타당성(기업 단위) — 새로 받은 KIND 상장목록의 '업종'(KSIC 명칭)을 외부 기준으로 사용.
상장 법인(KIND 1999–2026, 국내 주권)을 출원인코드에 잇고(p1), 그 법인의 제공범위 KR 출원(≥3건)으로
 (a) 제공 KSIC 라벨 최빈 산업, (b) 공식 IPC-KSIC 연계표(2013) 최빈 산업을 구해 KIND 업종과 일치율을 비교한다.
비교 수준: 중분류 2자리(두 방법 모두 가능), 소분류 3자리(연계표가 3자리 이상 해상도를 가진 산업만).
업종 명칭→코드: r7_data_ksic_ksco_revision_link_raw.csv의 KSIC 10차·11차 항목명(정규화 완전 일치).
연계표 산업코드는 KSIC 9차 기준(C2610 등) — 제조업 중분류 코드는 9·10차가 같다(26~33 범위 확인)."""
import os, re, json
import numpy as np, pandas as pd, duckdb
R7 = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(R7); BASE = os.path.dirname(os.path.dirname(ROOT))
EXT = os.path.join(ROOT, "_shared", "ext"); META = os.path.join(ROOT, "_shared", "meta", "*.parquet").replace("\\", "/")
nn = lambda s: re.sub(r"[\s,·\.;]", "", str(s)).replace("및", "")
L = pd.read_csv(os.path.join(EXT, "r7_data_ksic_ksco_revision_link_raw.csv"), encoding="cp949", dtype=str)
L = L[L["분류종류"] == "KSIC"]
names = pd.concat([L[["신분류코드", "신분류항목명"]].set_axis(["code", "nm"], axis=1), L[["구분류코드", "구분류항목명"]].set_axis(["code", "nm"], axis=1)])
names = names[names.code.str.fullmatch(r"\d{2,4}")]
names["k"] = names.nm.map(nn)
nm2code = names.groupby("k").code.agg(lambda s: s.value_counts().index[0])
K = pd.read_parquet(os.path.join(R7, "cache", "kind_linked.parquet"))
K["ind_code"] = K["업종"].map(nn).map(nm2code)
E = pd.read_parquet(os.path.join(R7, "cache", "ipo_events.parquet")); E["applicantCd"] = E.applicantCd.astype(str)
E = E.merge(K[["종목코드", "listing_date", "ind_code", "업종"]].drop_duplicates(["종목코드", "listing_date"]), on=["종목코드", "listing_date"])
E = E[E.listing_date <= "2023-12-31"]  # IPO 레이더 최종시험 창(2024–25) 상장 기업은 건드리지 않는다
E = E.dropna(subset=["ind_code"]).drop_duplicates(["applicantCd", "종목코드"])
A = pd.read_parquet(os.path.join(ROOT, "N4_특허행태_기업폐업조기경보", "VERIFY_R5", "cache", "app_firm.parquet"))
A["applicantCd"] = A.applicantCd.astype(str)
kr = duckdb.sql(f"select appl_no, main_ipc from read_parquet('{META}') where cc='KR'").df()
A = A[A.applicantCd.isin(set(E.applicantCd))].merge(kr, on="appl_no", how="left")
X = pd.read_excel(os.path.join(BASE, r"공모전데이터\외부데이터\20131002_IPC-KSIC\20131002_IPC-KSIC.xlsx"), sheet_name="전체연계표", dtype=str)
X.columns = ["cls", "ipc", "cnt"]; X["cnt"] = pd.to_numeric(X.cnt); X["ipc"] = X.ipc.str.replace(" ", "")
X["mg"] = X.ipc.str.split("/").str[0]; X["sub"] = X.ipc.str[:4]
def best(key):
    s = X.groupby([key, "cls"]).cnt.sum().reset_index().sort_values("cnt", ascending=False); return s.drop_duplicates(key).set_index(key).cls
bs, bm, bb = best("ipc"), best("mg"), best("sub")
ip = A.main_ipc.fillna("").str.replace(" ", "")
A["alt"] = ip.map(bs).fillna(ip.str.split("/").str[0].map(bm)).fillna(ip.str[:4].map(bb))
# 연계표 코드 → 숫자 접두(끝 0 제거, 최소 2자리): C2610→261, C2041→2041, C1000→10, C3000→30
def cls_prefix(c):
    if not isinstance(c, str) or not c.startswith("C"): return None
    d = c[1:].rstrip("0"); return d if len(d) >= 2 else c[1:3]
A["alt_p"] = A.alt.map(cls_prefix)
firm = E.groupby("applicantCd").agg(ind=("ind_code", "first"), 업종=("업종", "first"))
cnt = A.groupby("applicantCd").size(); firm = firm[firm.index.isin(cnt[cnt >= 3].index)]
mk = A.groupby("applicantCd").ksic.agg(lambda s: s.value_counts().index[0])
ma = A.dropna(subset=["alt_p"]).groupby("applicantCd").alt_p.agg(lambda s: s.value_counts().index[0])
firm["ksic"] = mk; firm["alt"] = ma
firm = firm.dropna(subset=["ksic", "alt"])
r = {"법인 수(제공범위 KR 출원 ≥3, KIND 업종 코드화)": int(len(firm)),
     "KIND 업종 코드화율": round(float(K.ind_code.notna().mean()), 3)}
f2 = firm.copy()
r["중분류 2자리 일치 — 제공 KSIC 라벨"] = round(float((f2.ksic.str[:2] == f2.ind.str[:2]).mean()), 3)
r["중분류 2자리 일치 — 공식 연계표"] = round(float((f2.alt.str[:2] == f2.ind.str[:2]).mean()), 3)
f3 = firm[(firm.alt.str.len() >= 3) & (firm.ind.str.len() >= 3)]
r["소분류 3자리 비교 법인 수(연계표 3자리 이상 & 업종 3자리 이상)"] = int(len(f3))
r["소분류 3자리 일치 — 제공 KSIC 라벨"] = round(float((f3.ksic.str[:3] == f3.ind.str[:3]).mean()), 3)
r["소분류 3자리 일치 — 공식 연계표"] = round(float((f3.alt.str[:3] == f3.ind.str[:3]).mean()), 3)
# 업종이 50 KSIC의 3자리 집합 안인 법인만(제공 라벨이 맞힐 수 있는 경우)
s50 = set(A.ksic.str[:3])
f4 = f3[f3.ind.str[:3].isin(s50)]
r["  (업종 3자리가 50 KSIC 범위 안인 법인) n"] = int(len(f4))
r["  제공 KSIC 일치"] = round(float((f4.ksic.str[:3] == f4.ind.str[:3]).mean()), 3)
r["  공식 연계표 일치"] = round(float((f4.alt.str[:3] == f4.ind.str[:3]).mean()), 3)
# 부트스트랩 CI(법인 재추출) — 3자리 차이
rng = np.random.default_rng(0); a = (f3.ksic.str[:3] == f3.ind.str[:3]).values.astype(float); b = (f3.alt.str[:3] == f3.ind.str[:3]).values.astype(float)
bs_ = [np.mean(a[i] - b[i]) for i in (rng.integers(0, len(a), len(a)) for _ in range(1000))]
r["소분류 3자리 일치율 차이(제공−연계표) 95% CI"] = np.percentile(bs_, [2.5, 97.5]).round(3).tolist()
# 50개 세세분류가 연계표 61개 산업 중 몇 개로 뭉치는가(9→10차 연계 후 접두 규칙)
v10 = L[(L["분류차순정보"] == "10") & (L["분류범위"] == "5")]
k50 = sorted(duckdb.sql(f"select distinct ksic from read_parquet('{META}')").df().ksic)
cls_codes = sorted({c for c in X.cls.unique() if c.startswith("C")}, key=lambda c: -len(c[1:].rstrip("0")))
def to_cls(k9):
    for c in cls_codes:
        p = c[1:].rstrip("0"); p = p if len(p) >= 2 else c[1:3]
        if k9.startswith(p): return c
    if k9.startswith("291"): return "C2919"   # 사무용 이외의 일반 기계(291 − 2918)
    if k9.startswith("319"): return "C3199"
    return None
m = {k: sorted({to_cls(o) for o in v10[v10["신분류코드"] == k]["구분류코드"] if to_cls(o)}) for k in k50}
grp = pd.Series({k: "|".join(v) for k, v in m.items()})
r["50 세세분류 → 연계표 산업(9→10차 연계 경유) 고유 조합 수"] = int(grp.nunique())
r["연계표 산업 하나에 2개 이상 세세분류가 뭉치는 경우(예)"] = {c: list(g.index) for c, g in grp.groupby(grp) if len(g) >= 3}
json.dump(r, open(os.path.join(R7, "validity_kind_industry.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(r, ensure_ascii=False, indent=1))
