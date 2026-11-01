# -*- coding: utf-8 -*-
"""(d1) 합병·분할 처리: 등록원부 owner '법인합병' 이전의 소멸 법인(직전 권리자)과 존속 법인(새 권리자)을 출원인코드·CA 번호로 대조.
출력 cache/s07_out.json, cache/s07_merger_pairs.parquet"""
import os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from paths import *
from lib_ck import *
import pandas as pd, numpy as np
ca = pd.read_parquet(os.path.join(CACHE, "ca.parquet"))
A = pd.read_parquet(os.path.join(CACHE, "reg_app.parquet"))
r = ro(REG_DB)
own = pd.read_sql("select rgst_no, seq, ownerName, ownerRgstCsDate, ownerRgstCsName from owner", r)
own["dt"] = pd.to_datetime(own.ownerRgstCsDate, format="%Y%m%d", errors="coerce")
# 같은 날짜에 권리자 1명인 특허만(단독 권리 체인)
own["n_same_dt"] = own.groupby(["rgst_no", "ownerRgstCsDate"]).seq.transform("size")
own = own.sort_values(["rgst_no", "seq"])
own["prev_name"] = own.groupby("rgst_no").ownerName.shift(1)
own["prev_n_same"] = own.groupby("rgst_no").n_same_dt.shift(1)
O = {}
ev = {}
for kind, names in [("merger", ["법인합병", "합병"]), ("split", ["법인분할", "분할"])]:
    M = own[own.ownerRgstCsName.isin(names) & (own.n_same_dt == 1) & (own.prev_n_same == 1) & own.prev_name.notna()].copy()
    O[f"{kind}_events_single_chain"] = len(M)
    # 이름 → 국내 법인 코드(등록원부 현재 이름 기준 + CA 이름 기준): 정규화 이름이 유일한 코드
    ca["nn"] = ca.name.map(norm)
    dom = ca[ca.cd.str[:1] == "1"]
    nn_cnt = dom.groupby("nn").cd.nunique()
    uniq = dom[dom.nn.map(nn_cnt) == 1].set_index("nn")
    M["prev_nn"] = M.prev_name.map(norm); M["new_nn"] = M.ownerName.map(norm)
    # 원출원인 코드: 이 특허의 등록원부 출원인 중 이름이 직전 권리자와 같은 코드(우선), 없으면 CA 이름 유일 매칭
    app = A[["rgst_no", "cd", "nm"]].copy(); app["nn"] = app.nm.map(norm)
    M = M.merge(app.rename(columns={"cd": "cd_app", "nn": "prev_nn"})[["rgst_no", "prev_nn", "cd_app"]], on=["rgst_no", "prev_nn"], how="left")
    M["cd_prev"] = M.cd_app.fillna(M.prev_nn.map(uniq.cd))
    M["cd_new"] = M.new_nn.map(uniq.cd)
    M = M[M.cd_prev.notna() & M.cd_new.notna() & (M.cd_prev != M.cd_new)]
    E = M.drop_duplicates(["cd_prev", "cd_new"])[["cd_prev", "cd_new", "prev_name", "ownerName", "dt"]]
    c = ca.set_index("cd")
    E["jur_prev"] = E.cd_prev.map(c.jur); E["jur_new"] = E.cd_new.map(c.jur)
    E["biz_prev"] = E.cd_prev.map(c.biz); E["biz_new"] = E.cd_new.map(c.biz)
    E["ca_name_prev"] = E.cd_prev.map(c.name)
    E["prev_name_kept"] = [sim(a, b) >= 0.5 if isinstance(a, str) else None for a, b in zip(E.ca_name_prev, E.prev_name)]
    E["same_jur"] = (E.jur_prev != "") & (E.jur_prev == E.jur_new)
    E["same_biz"] = (E.biz_prev != "") & (E.biz_prev == E.biz_new)
    S = pd.read_parquet(os.path.join(EXT, "r3nts_status.parquet"))
    S["biz"] = S.bizno.astype(str).str.replace(r"\D", "", regex=True)
    st = S.drop_duplicates("biz").set_index("biz")
    E["nts_prev"] = E.biz_prev.map(st.b_stt); E["end_prev"] = pd.to_datetime(E.biz_prev.map(st.end_dt), errors="coerce")
    E["gap_days"] = (E.end_prev - E.dt).dt.days
    n = len(E)
    O[f"{kind}_pairs"] = n
    O[f"{kind}_prev_code_jur_blank"] = int((E.jur_prev == "").sum())
    O[f"{kind}_prev_code_biz_blank"] = int((E.biz_prev == "").sum())
    O[f"{kind}_same_jur_as_survivor"] = int(E.same_jur.sum())
    O[f"{kind}_same_biz_as_survivor"] = int(E.same_biz.sum())
    O[f"{kind}_prev_ca_name_still_old"] = int((E.prev_name_kept == True).sum())
    O[f"{kind}_prev_nts_dist"] = E.nts_prev.fillna("사업자번호 없음/미조회").value_counts().to_dict()
    cl = E[E.nts_prev == "폐업자"]
    O[f"{kind}_prev_closed_gap_days_quantiles"] = cl.gap_days.quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(0).to_dict() if len(cl) else {}
    O[f"{kind}_prev_closed_within_1y_of_event"] = int((cl.gap_days.abs() <= 365).sum())
    O[f"{kind}_prev_closed_after_2016"] = int((cl.end_prev >= "2016-01-01").sum())
    O[f"{kind}_event_year_dist"] = E.dt.dt.year.value_counts().sort_index().to_dict()
    ev[kind] = E
    E.drop(columns=["jur_prev", "jur_new", "biz_prev", "biz_new"]).to_parquet(guard_path(os.path.join(CACHE, f"s07_{kind}_pairs.parquet")))
# N4 공식 시험 모집단(t0=2023)에 합병 소멸 코드가 들어가 폐업 양성으로 잡혔는가
try:
    P = pd.read_parquet(os.path.join(MC, r"N4_특허행태_기업폐업조기경보\final_test_pred_r4.parquet"))
    P["applicantCd"] = P.applicantCd.astype(str)
    E = ev["merger"]
    pm = P[P.applicantCd.isin(set(E.cd_prev))]
    O["N4_test_codes"] = len(P); O["N4_test_pos"] = int(P.y.sum())
    O["N4_merger_prev_codes_in_test"] = len(pm); O["N4_merger_prev_codes_pos"] = int(pm.y.sum())
    E2 = E.merge(P[["applicantCd", "y"]], left_on="cd_prev", right_on="applicantCd")
    O["N4_merger_prev_pos_event_year"] = E2[E2.y == 1].dt.dt.year.value_counts().sort_index().to_dict()
    ps = P[P.applicantCd.isin(set(ev["split"].cd_prev))]
    O["N4_split_prev_codes_in_test"] = len(ps); O["N4_split_prev_codes_pos"] = int(ps.y.sum())
except Exception as e:
    O["N4_err"] = repr(e)
json.dump(O, open(guard_path(os.path.join(CACHE, "s07_out.json")), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(O, ensure_ascii=False, indent=0, default=str))
