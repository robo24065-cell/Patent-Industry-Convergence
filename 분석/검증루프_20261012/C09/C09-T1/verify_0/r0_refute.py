# -*- coding: utf-8 -*-
# 반박 검증(독립): registry_full.sqlite 직접(mode=ro) + kr_patents.parquet. C09 캐시 미사용.
import sys, os, json, time, sqlite3
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, numpy as np
t0 = time.time(); res = {}
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C09\C09-T1\verify_0")
HY, KIA = "119980045675", "119980003181"
kp = pd.read_parquet(os.path.join(ROOT, r"공모전데이터\작업데이터\kr_patents.parquet"))
cols = list(kp.columns)
res["kp_cols"] = cols
# 열 위치: 0 출원번호, 1 출원연도, 4 출원인명(추정) -> 이름으로 확인
kp = kp.rename(columns={cols[0]: "appl", cols[1]: "ay"})
name_col = [c for c in cols if c == "출원인명"][0]
ksic_col = [c for c in cols if c == "세세분류코드"][0]
kp["appl"] = kp["appl"].astype(str).str.replace("-", "", regex=False)
kp = kp.rename(columns={name_col: "pname", ksic_col: "ksic"})
kp = kp[kp.appl.str[:2] == "10"][["appl", "ay", "pname", "ksic"]]
res["kp_rows_10"] = len(kp); res["kp_dup_appl"] = int(kp.appl.duplicated().sum())
# 이름 기준 연도별
nm = kp[kp.pname.isin(["현대자동차주식회사", "기아 주식회사"])].groupby(["ay", "pname"]).size().unstack(fill_value=0)
res["name_year"] = {int(k): v for k, v in nm.loc[2012:2023].to_dict("index").items()}
# '현대자동차' 포함 모든 이름 변형
res["names_with_hyundai_motor"] = kp[kp.pname.fillna("").str.contains("현대자동차")].pname.value_counts().head(10).to_dict()
c = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite").replace("\\", "/") + "?mode=ro", uri=True)
res["rf_patent_rows"] = c.execute("select count(*) from patent").fetchone()[0]
pat = pd.DataFrame(c.execute("select rgst_no, applNo, substr(applDate,1,4), ksic from patent").fetchall(), columns=["rgst_no", "applNo", "ay_rf", "ksic_rf"])
pat["applNo"] = pat.applNo.astype(str).str.replace("-", "", regex=False)
app = pd.DataFrame(c.execute("select rgst_no, seq, applicantCd, applicantName, rpstrYn from applicant").fetchall(), columns=["rgst_no", "seq", "cd", "aname", "rp"])
hyk = app[app.cd.isin([HY, KIA])]
g = hyk.groupby("rgst_no").cd.agg(lambda s: (HY in set(s), KIA in set(s)))
flags = pd.DataFrame(g.tolist(), index=g.index, columns=["hy", "kia"]).reset_index()
nap = app.groupby("rgst_no").size().rename("n_app").reset_index()
f = flags.merge(nap, on="rgst_no").merge(pat, on="rgst_no", how="left")
f = f.merge(kp, left_on="applNo", right_on="appl", how="left")
f["inscope"] = f.appl.notna()
f["combo"] = np.select([f.hy & ~f.kia, f.hy & f.kia], ["HY_only", "HY_KIA"], "KIA_only")
f["ay_rf"] = pd.to_numeric(f.ay_rf, errors="coerce")
res["rf_hy_rows_inscope_share"] = float(f[f.hy].inscope.mean())
# registry_full이 제공 범위만 담는지(전체 포트폴리오 아님) 확인
res["rf_patent_inscope_share_all"] = float(pat.applNo.isin(set(kp.appl)).mean())
t = f[f.inscope].groupby(["ay_rf", "combo"]).size().unstack(fill_value=0)
res["inscope_combo_by_year"] = {int(k): v for k, v in t.loc[2012:2022].to_dict("index").items()}
# 순수 쌍(출원인이 정확히 현대+기아 2명)과 제공 이름
pair = f[f.inscope & (f.combo == "HY_KIA")]
res["pair_name_by_year"] = {int(y): grp.pname.value_counts().head(3).to_dict() for y, grp in pair.groupby("ay_rf") if 2012 <= y <= 2021}
pp = pair[pair.n_app == 2]
res["pure_pair_n"] = len(pp); res["pure_pair_kia_named"] = int((pp.pname == "기아 주식회사").sum())
res["pure_pair_kia_named_pre2017"] = [int(((pp.ay_rf < 2017) & (pp.pname == "기아 주식회사")).sum()), int((pp.ay_rf < 2017).sum())]
res["pure_pair_kia_named_2017plus"] = [int(((pp.ay_rf >= 2017) & (pp.pname == "기아 주식회사")).sum()), int((pp.ay_rf >= 2017).sum())]
# 현대만(범위 안) 제공 이름
ho = f[f.inscope & (f.combo == "HY_only")]
res["hy_only_name_by_year"] = {int(y): grp.pname.value_counts().head(3).to_dict() for y, grp in ho.groupby("ay_rf") if 2012 <= y <= 2021}
# 쌍 안 seq·rpstrYn: 기아가 seq0인지, 대표자인지(선택 규칙 단서)
pa_ = app[app.rgst_no.isin(pp.rgst_no)]
s0 = pa_.sort_values("seq").groupby("rgst_no").head(1)
s0 = s0.merge(pp[["rgst_no", "ay_rf"]], on="rgst_no")
res["pure_pair_seq0_is_kia_pre2017"] = float((s0[s0.ay_rf < 2017].cd == KIA).mean())
res["pure_pair_seq0_is_kia_2017plus"] = float((s0[s0.ay_rf >= 2017].cd == KIA).mean())
rp = pa_[pa_.rp == "Y"].merge(pp[["rgst_no", "ay_rf"]], on="rgst_no")
res["pure_pair_rpstr_counts"] = rp.groupby([rp.ay_rf >= 2017, "cd"]).size().astype(int).reset_index().astype(str).values.tolist()
res["rpstrYn_values"] = app.rp.value_counts(dropna=False).head(5).astype(int).to_dict()
# 일반 규칙 후보: 공동출원(2명 이상, 범위 안) 중 제공 이름 = 최소 코드 출원인 이름 비율
fa = app[app.rgst_no.isin(set(pat[pat.applNo.isin(set(kp.appl))].rgst_no))]
multi = fa.groupby("rgst_no").filter(lambda x: len(x) >= 2) if False else None
cnt = fa.groupby("rgst_no").size(); mr = set(cnt[cnt >= 2].index)
fm = fa[fa.rgst_no.isin(mr) & fa.cd.notna()]
mn = fm.sort_values("cd").groupby("rgst_no").head(1)[["rgst_no", "aname"]].rename(columns={"aname": "min_cd_name"})
mx = fm.sort_values("cd").groupby("rgst_no").tail(1)[["rgst_no", "aname"]].rename(columns={"aname": "max_cd_name"})
z = mn.merge(mx, on="rgst_no").merge(pat[["rgst_no", "applNo", "ay_rf"]], on="rgst_no").merge(kp[["appl", "pname"]], left_on="applNo", right_on="appl")
z["ay_rf"] = pd.to_numeric(z.ay_rf, errors="coerce")
zz = z[z.ay_rf.between(2015, 2019)]
res["joint_1519_n"] = len(zz)
res["joint_1519_provided_eq_min_cd"] = float((zz.pname == zz.min_cd_name).mean())
res["joint_1519_provided_eq_max_cd"] = float((zz.pname == zz.max_cd_name).mean())
# 시점 정합: 쌍 문서의 '기아' 포함이 출원 이후 이전(owner 변경)인가 — owner 표 등록원인
ow = pd.DataFrame(c.execute("select rgst_no, ownerName, ownerRgstCsReason, ownerRgstCsDate from owner where rgst_no in (%s)" % ",".join("'%s'" % r for r in pp.rgst_no.head(3000))).fetchall(),
                  columns=["rgst_no", "oname", "reason", "odate"])
res["pair_owner_reason_counts"] = ow.reason.fillna("NA").value_counts().head(8).astype(int).to_dict()
res["pair_owner_names"] = ow.oname.value_counts().head(5).astype(int).to_dict()
# 자동차 KSIC(29·30)가 제공 KSIC 집합에 있는가
ks = kp.ksic.astype(str)
res["provided_ksic_29_30"] = sorted(set(ks[ks.str[:2].isin(["29", "30"])]))
res["provided_n_ksic"] = int(ks.nunique())
# 현대 코드 범위 안(전 조합) 문서의 KSIC 2자리 29·30 비율
hin = f[f.hy & f.inscope]
hin2 = hin.assign(k2=hin.ksic.astype(str).str[:2])
res["hy_inscope_share_29_30"] = hin2[hin2.ay_rf.between(2010, 2021)].groupby("ay_rf").k2.apply(lambda s: round(float(s.isin(["29", "30"]).mean()), 3)).to_dict()
# 2016 현대 단독 범위 안의 KSIC·IPC 쏠림 (453 중 이상 배치인가)
res["sec"] = round(time.time() - t0)
json.dump(res, open(os.path.join(OUT, "r0_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
for k, v in res.items():
    print(k, json.dumps(v, ensure_ascii=False, default=str)[:1200])
