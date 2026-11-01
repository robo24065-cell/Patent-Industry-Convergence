# -*- coding: utf-8 -*-
r"""[사후, PREREG 밖] P1 결과를 본 뒤 추가한 점검. 출력 res_posthoc.json
 (a) KR-해외 라벨 일치가 내용 차이(메인 IPC 서브그룹 일치 여부)에 따라 달라지나 — 라벨 '복사' 여부 점검
 (b) KR-해외 일치율을 출원인 유형별로: 등록업종(NPS/DART 50개 안) 있는 법인 vs 그 밖 법인 vs 대학·연구기관 vs 개인
     (등록업종이 KR 라벨에 쓰였다면 등록업종 법인에서 KR-해외 불일치가 커야 함)
 (c) P1 국가별(US·JP·EP만) 민감도"""
import os, sys, json
sys.dont_write_bytecode = True
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
OUT = os.path.join(MC, r"_shared_report\label_origin_R10"); CACHE = os.path.join(OUT, "cache")
KSIC50 = set("15220 20202 20411 20423 20491 21220 23992 24191 24219 25914 25922 25924 25932 26111 26112 26121 26129 26211 26212 26223 26224 26291 26322 26323 26422 26429 26519 26529 27199 27211 27212 27301 28111 28122 28511 28902 29131 29132 29133 29150 29162 29210 29229 29241 29280 30393 31202 31312 33301 33309".split())
rng = np.random.default_rng(7)
kr = pd.read_parquet(os.path.join(CACHE, "kr_lo.parquet"), columns=["appl_no", "ksic", "sg", "app"]); kr["appl_no"] = kr.appl_no.astype(str)
fam = pd.read_parquet(os.path.join(CACHE, "fam_reg.parquet")); fam["kr_appl_no"] = fam.kr_appl_no.astype(str)
RT = pq.read_table(os.path.join(MC, r"_ideas_R8\cache\rec.parquet"), columns=["cc", "appl_no", "sg"])
RT = RT.filter(pc.is_in(RT["appl_no"], value_set=pa.array(fam.appl_no.astype(str).unique())))
fs = RT.to_pandas().rename(columns={"sg": "fsg"}); fs = fs[fs.cc != "KR"]
F = fam.merge(fs, on=["cc", "appl_no"], how="left").merge(kr.rename(columns={"appl_no": "kr_appl_no"}), on="kr_appl_no")
F["agree"] = F.fk == F.ksic; F["same_sg"] = F.fsg == F.sg
F["same_sc"] = F.fsg.str[:4] == F.sg.str[:4]
J = {}
J["a_agree_by_ipc"] = {"메인IPC서브그룹 같음": [round(float(F.agree[F.same_sg].mean()), 4), int(F.same_sg.sum())],
                       "서브그룹 다름·서브클래스 같음": [round(float(F.agree[~F.same_sg & F.same_sc].mean()), 4), int((~F.same_sg & F.same_sc).sum())],
                       "서브클래스도 다름": [round(float(F.agree[~F.same_sc].mean()), 4), int((~F.same_sc).sum())]}
pf = pd.read_parquet(os.path.join(CACHE, "pat_firm.parquet")); reg = pd.read_parquet(os.path.join(CACHE, "firm_reg.parquet"))
regd = set(reg.applicantCd[reg.nps.isin(KSIC50) | reg.dart.isin(KSIC50)])
corp = pf.drop_duplicates("appl_no").set_index("appl_no").applicantCd
F["cd"] = F.kr_appl_no.map(corp)
uni = F.app.str.contains("대학|산학협력|연구원|연구소|INSTITUTE|UNIV", regex=True, na=False)
F["유형"] = np.select([F.cd.isin(regd), F.cd.notna() & ~uni, uni], ["등록업종 50개 안 법인(KR)", "그 밖 KR 법인", "대학·연구기관"], "그 밖(개인·해외 출원인 등)")
J["b_agree_by_type"] = {k: [round(float(v.agree.mean()), 4), int(len(v))] for k, v in F.groupby("유형")}
# 등록 법인 vs 그 밖 KR 법인 차이 CI(출원인 군집)
G = F[F["유형"].isin(["등록업종 50개 안 법인(KR)", "그 밖 KR 법인"])].groupby(["유형", "cd"]).agree.agg(["sum", "size"]).reset_index()
d = []
for _ in range(1000):
    s = [G[G["유형"] == t].sample(frac=1, replace=True, random_state=int(rng.integers(1e9))) for t in ("등록업종 50개 안 법인(KR)", "그 밖 KR 법인")]
    d.append(s[0]["sum"].sum() / s[0]["size"].sum() - s[1]["sum"].sum() / s[1]["size"].sum())
J["b_diff_registered_minus_other_corp"] = [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]
# (c) 국가별 P1
X = pf.merge(F[["kr_appl_no", "cc", "fk", "ksic"]].rename(columns={"kr_appl_no": "appl_no"}), on="appl_no")
out = []
for src in ("nps", "dart"):
    R = reg[["applicantCd", src]].dropna(); R = R[R[src].isin(KSIC50)]
    for cc in ("US", "JP", "EP"):
        Y = X[X.cc == cc].drop_duplicates(["applicantCd", "appl_no"]).merge(R.rename(columns={src: "r5"}), on="applicantCd")
        g = Y.assign(k=(Y.ksic == Y.r5), f=(Y.fk == Y.r5)).groupby("applicantCd")[["k", "f"]].mean()
        out.append({"기준": src.upper(), "국가": cc, "기업": len(g), "특허": len(Y), "KR라벨": round(float(g.k.mean()), 4), "해외라벨": round(float(g.f.mean()), 4)})
J["c_p1_by_cc"] = out
json.dump(J, open(os.path.join(OUT, "res_posthoc.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(J, ensure_ascii=False, indent=1))
