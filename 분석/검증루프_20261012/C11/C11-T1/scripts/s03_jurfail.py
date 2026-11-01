# -*- coding: utf-8 -*-
"""a1 법인번호 체크섬 실패의 성격: 코드 발급연도·법인 종류·DART 공식값 대조·국세청 상태."""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
from lib_ck import *
import pandas as pd, numpy as np

ca = pd.read_parquet(os.path.join(CACHE, "ca.parquet"))
O = {}
J = ca[ca.jur != ""].copy()
J["fail"] = J.jur_ok == False
J["iy"] = J.cd.str[1:5]
J["serial"] = J.jur.str[6:12].astype(int)
O["fail_by_issue_year"] = J.groupby("iy").fail.mean().round(4).to_dict()
O["fail_by_kind"] = J.groupby("jkind").fail.agg(["mean", "size"]).query("size>=100").round(4).to_dict("index")
J["sbin"] = pd.cut(J.serial, [-1, 999, 9999, 49999, 99999, 199999, 499999, 999999])
O["fail_by_serial_bin"] = J.groupby("sbin", observed=True).fail.agg(["mean", "size"]).round(4).rename(index=str).to_dict("index")
# DART 공식값과 같은 번호인가 (DART는 금감원 공시 회사 고유정보)
dc = ro(DART_DB)
D = pd.read_sql("select corp_name, jurir_no, bizr_no, est_dt from company", dc)
D["jur"] = D.jurir_no.fillna("").str.replace(r"\D", "", regex=True)
D["ok"] = D.jur.map(jur_ok)
O["dart_fail_rate"] = float((D.ok == False).sum() / D.ok.notna().sum())
D["est_y"] = D.est_dt.fillna("").str[:4]
D["ey"] = pd.cut(pd.to_numeric(D.est_y, errors="coerce"), [0, 1970, 1990, 2000, 2010, 2030])
O["dart_fail_by_est"] = D.groupby("ey", observed=True).ok.apply(lambda s: float((s == False).mean())).round(4).to_dict()
O["dart_fail_by_est"] = {str(k): v for k, v in O["dart_fail_by_est"].items()}
fj = set(J.jur[J.fail]); dj = set(D.jur[D.ok == False])
O["ca_fail_in_dart_same_number"] = len(fj & set(D.jur))
O["ca_fail_in_dart_and_dart_fail"] = len(fj & dj)
# 국세청 상태: 체크섬 실패 법인번호 행의 사업자번호가 등록된 사업자인가
S = pd.read_parquet(os.path.join(EXT, "r3nts_status.parquet"))
S["biz"] = S.bizno.astype(str).str.replace(r"\D", "", regex=True)
st = S.set_index("biz").b_stt.fillna("미등록")
for lab, msk in [("jur_fail", J.fail), ("jur_ok", ~J.fail)]:
    b = J.loc[msk & (J.biz != ""), "biz"]
    O[f"nts_{lab}"] = dict(n=int(len(b)), dist=b.map(st).fillna("조회없음").value_counts(normalize=True).round(4).to_dict(),
                          biz_share=float((J.loc[msk, "biz"] != "").mean()))
# 사업자번호 체크섬 통과 비율 비교
O["biz_ok_given_jur_fail"] = float((J.loc[J.fail & (J.biz != ""), "biz_ok"] == True).mean())
# 실패 번호의 검증자리 차이 분포 (오타면 무작위, 다른 규칙이면 특정 패턴)
def diff(d):
    s = sum(int(d[i]) * (1 if i % 2 == 0 else 2) for i in range(12))
    return (int(d[12]) - (10 - s % 10) % 10) % 10
O["fail_checkdigit_offset"] = J.loc[J.fail, "jur"].map(diff).value_counts().sort_index().to_dict()
# 대안 규칙: 가중합 대신 자릿수합(2배 곱 결과의 각 자리 합, Luhn식)
def luhn_like(d):
    s = 0
    for i in range(12):
        v = int(d[i]) * (1 if i % 2 == 0 else 2)
        s += v // 10 + v % 10
    return (10 - s % 10) % 10 == int(d[12])
O["fail_pass_luhn_like"] = float(J.loc[J.fail, "jur"].map(luhn_like).mean())
O["ok_pass_luhn_like"] = float(J.loc[~J.fail, "jur"].map(luhn_like).mean())
json.dump(O, open(guard_path(os.path.join(CACHE, "s03_out.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=0, default=str))
