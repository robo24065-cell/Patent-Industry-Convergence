# -*- coding: utf-8 -*-
"""N2 재무 파싱. 두 가지 판을 만든다.
(1) fin_first.parquet — 파일럿 f05 방식: 파일럿 수집 4개 보고서(2015·18·21·23)만, 같은 회계연도 값은 당기>전기>전전기 우선.
    회계연도 2014·16·17·19·20·22 는 이후 보고서의 비교수치(재작성 가능·생존 편향).
(2) fin_rt.parquet — 실시간판: 회계연도 t 성장 = 't년 사업보고서' 한 건의 당기(t)/전기(t−1). t+1년 3월 공시 → 4월 나우캐스트 시점에 알 수 있는 값.
    보고서 연도 2014~2023 전부(파일럿 4개 + N2 추가 6개). 별도(OFS) 우선, 매출 없으면 연결(CFS).
    주의: fnlttMultiAcnt는 정정보고서가 있으면 정정본을 돌려준다(rcept_no로 확인 가능) → '최초 공시값'과 다를 수 있음.
출력 컬럼: corp_code, year, fs, rev, rev_prev, assets, as_prev, op, op_prev, ... (fin_rt), rcept_no"""
import os, json, glob, sys
import numpy as np, pandas as pd
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "cache")
EXT = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
MAXREP = int(sys.argv[1]) if len(sys.argv) > 1 else 2023   # 개발 2023 고정. final_run_r2.py 만 2024/2025 전달
SUF = "" if MAXREP == 2023 else f"_{MAXREP}"
ACC = {"매출액": "rev", "영업이익": "op", "당기순이익(손실)": "ni", "자산총계": "assets", "부채총계": "liab",
       "자본총계": "equity", "자본금": "capital", "유동자산": "cur_assets", "이익잉여금": "re"}

def num(v):
    v = (v or "").replace(",", "").strip()
    if v in ("", "-"):
        return np.nan
    try:
        return float(v)
    except ValueError:
        return np.nan

files = [os.path.join(EXT, "R2firm_dart_fnltt_raw.jsonl")] + sorted(glob.glob(os.path.join(EXT, "R2N2_dart_fnltt_*.jsonl")))
rows = []
for fn in files:
    src = "pilot" if "R2firm" in fn else "n2"
    for line in open(fn, encoding="utf-8"):
        x = json.loads(line)
        a = ACC.get(x["account_nm"])
        if a is None:
            continue
        by = int(x["bsns_year"])
        if by > MAXREP:
            continue
        rows.append((x["corp_code"], by, x["fs_div"], a, num(x.get("thstrm_amount")), num(x.get("frmtrm_amount")),
                     num(x.get("bfefrmtrm_amount")), x.get("rcept_no"), src))
R = pd.DataFrame(rows, columns=["corp_code", "rep", "fs", "acc", "cur", "prev", "prev2", "rcept_no", "src"])
R = R.drop_duplicates(["corp_code", "rep", "fs", "acc"])
print("reports by year", R.groupby("rep").corp_code.nunique().to_dict())

# (1) 파일럿판
P = R[R.src == "pilot"]
L = pd.concat([P.assign(year=P.rep, v=P.cur, pr=0), P.assign(year=P.rep - 1, v=P.prev, pr=1), P.assign(year=P.rep - 2, v=P.prev2, pr=2)])
L = L[L.v.notna()].sort_values("pr").drop_duplicates(["corp_code", "year", "fs", "acc"])
W = L.pivot_table(index=["corp_code", "year", "fs"], columns="acc", values="v").reset_index()
W["fsr"] = (W.fs != "OFS").astype(int)
W = W.sort_values("fsr").drop_duplicates(["corp_code", "year"]).drop(columns="fsr")
assert W.year.max() <= 2023
if MAXREP == 2023:
    W.to_parquet(os.path.join(C, "fin_first.parquet"), index=False)

# (2) 실시간판: 보고서 rep 한 건의 당기·전기
cur = R.pivot_table(index=["corp_code", "rep", "fs"], columns="acc", values="cur").add_suffix("")
prv = R.pivot_table(index=["corp_code", "rep", "fs"], columns="acc", values="prev").add_suffix("_prev")
Q = cur.join(prv, how="outer").reset_index()
# OpenDART 재무 API는 2015 사업연도부터 제공 → 2014 보고서 없음. 회계연도 2014 성장(학습행 전용, 개발 원점 아님)은
# 2015 보고서의 전기(2014)/전전기(2013) 비교수치로 대체(주석으로 명시).
if not (Q.rep == 2014).any():
    P15 = R[R.rep == 2015]
    c14 = P15.pivot_table(index=["corp_code", "fs"], columns="acc", values="prev")
    p14 = P15.pivot_table(index=["corp_code", "fs"], columns="acc", values="prev2").add_suffix("_prev")
    Q14 = c14.join(p14, how="outer").reset_index(); Q14["rep"] = 2014
    Q = pd.concat([Q, Q14], ignore_index=True)
Q["has_rev"] = (Q.rev.notna() & Q.rev_prev.notna()).astype(int)
Q["fsr"] = (Q.fs != "OFS").astype(int)
Q = Q.sort_values(["has_rev", "fsr"], ascending=[False, True]).drop_duplicates(["corp_code", "rep"]).drop(columns=["fsr", "has_rev"])
Q = Q.rename(columns={"rep": "year"})
Q = Q.rename(columns={"assets_prev": "as_prev"})
assert Q.year.max() <= MAXREP
Q.to_parquet(os.path.join(C, f"fin_rt{SUF}.parquet"), index=False)
print("fin_first firms/year", W.groupby("year").rev.count().to_dict())
print("fin_rt firms/year (rev&rev_prev)", Q[Q.rev.notna() & Q.rev_prev.notna()].groupby("year").size().to_dict())
print("fin_rt OFS share", Q.groupby("year").fs.apply(lambda s: round((s == "OFS").mean(), 3)).to_dict())

# (3) 실시간판2(rt2): rev_t = t년 보고서 당기, rev_{t−1} = (t−1)년 보고서 당기(최초 보고값). 둘 다 t+1년 4월에 알 수 있음.
#     t−1년 보고서가 없으면(2014, 신규 공시기업) t년 보고서 전기 값으로 대체.
C1 = R.pivot_table(index=["corp_code", "rep", "fs"], columns="acc", values="cur").reset_index()
C1["fsr"] = (C1.fs != "OFS").astype(int)
C1 = C1[C1.rev.notna()].sort_values("fsr").drop_duplicates(["corp_code", "rep"])
C1 = C1[["corp_code", "rep", "fs", "rev", "assets", "op"]].rename(columns={"rep": "year"})
prev_own = C1.assign(year=C1.year + 1).rename(columns={"rev": "rev_prev1", "assets": "as_prev1", "op": "op_prev1", "fs": "fs_prev"})
Q2 = Q[["corp_code", "year", "rev", "rev_prev", "assets", "as_prev", "op", "op_prev", "fs"]].merge(
    prev_own[["corp_code", "year", "rev_prev1", "as_prev1", "op_prev1", "fs_prev"]], on=["corp_code", "year"], how="left")
use = Q2.rev_prev1.notna() & (Q2.fs_prev == Q2.fs)
Q2["prev_src"] = np.where(use, "prev_report", "same_report")
for a, b in [("rev_prev", "rev_prev1"), ("as_prev", "as_prev1"), ("op_prev", "op_prev1")]:
    Q2[a] = np.where(use, Q2[b], Q2[a])
Q2 = Q2.drop(columns=["rev_prev1", "as_prev1", "op_prev1", "fs_prev"])
assert Q2.year.max() <= MAXREP
Q2.to_parquet(os.path.join(C, f"fin_rt2{SUF}.parquet"), index=False)
print("fin_rt2 prev from previous report share", Q2.groupby("year").prev_src.apply(lambda s: round((s == "prev_report").mean(), 3)).to_dict())
d = Q2[(Q2.rev > 0) & (Q2.rev_prev > 0)]
print("rt vs rt2 growth diff |>1%| share", round((np.abs(np.log(d.rev / d.rev_prev) - np.log(Q.set_index(['corp_code','year']).reindex(pd.MultiIndex.from_frame(d[['corp_code','year']])).rev.values / Q.set_index(['corp_code','year']).reindex(pd.MultiIndex.from_frame(d[['corp_code','year']])).rev_prev.values)) > 0.01).mean(), 3))
