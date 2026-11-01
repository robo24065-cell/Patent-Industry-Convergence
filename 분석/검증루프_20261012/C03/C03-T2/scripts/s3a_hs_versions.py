# -*- coding: utf-8 -*-
"""s3a: UNSD HS2017↔HS2012/HS2007 대응표 무결성 + 대응표(1,019 HS6) 버전 진단 + 보충 수집 대상 구버전 코드."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
D = os.path.join(DL, "unsd_hs")


def corr(fname, sheet, newc, oldc):
    df = pd.read_excel(os.path.join(D, fname), sheet, header=None, dtype=str)
    hdr = df.index[df.apply(lambda r: r.astype(str).str.contains("Relationship").any(), axis=1)][0]
    body = df.iloc[hdr + 1:].copy()
    if body.shape[1] == 4:
        body.columns = ["new", "exflag", "old", "rel"]
    else:
        body.columns = ["new", "old", "rel"]; body["exflag"] = np.nan
    body = body.dropna(subset=["new", "old"])
    body["new"] = body.new.str.strip().str.zfill(6); body["old"] = body.old.str.strip().str.zfill(6)
    return body


def conv(fname, sheet):
    df = pd.read_excel(os.path.join(D, fname), sheet, header=None, dtype=str).iloc[1:]
    df.columns = ["new", "old"]
    return df.dropna().assign(new=lambda d: d.new.str.strip().str.zfill(6), old=lambda d: d.old.str.strip().str.zfill(6))


c12 = corr("HS2017toHS2012ConversionAndCorrelationTables.xlsx", "Correlation HS17-HS12", "HS2017", "HS2012")
c07 = corr("HS2017toHS2007ConversionAndCorrelationTables.xlsx", "Correlation HS17-HS07", "HS2017", "HS2007")
v12 = conv("HS2017toHS2012ConversionAndCorrelationTables.xlsx", "Conversion HS17-HS12")
v07 = conv("HS2017toHS2007ConversionAndCorrelationTables.xlsx", "Conversion HS17-HS07")
chk = {}
for nm, c, v in [("HS17-HS12", c12, v12), ("HS17-HS07", c07, v07)]:
    n_new = c.new.nunique(); n_old = c.old.nunique()
    # 관계 표기 일관성: 1:1이면 new·old 모두 한 번씩
    cn = c.groupby("new").old.nunique(); co = c.groupby("old").new.nunique()
    c["calc"] = np.where(c.new.map(cn) == 1, np.where(c.old.map(co) == 1, "1:1", "n:1"), np.where(c.old.map(co) == 1, "1:n", "n:n"))
    chk[nm] = dict(rows=int(len(c)), new_codes=int(n_new), old_codes=int(n_old), dup_rows=int(c.duplicated(["new", "old"]).sum()),
                   rel_counts=c.rel.value_counts().to_dict(), rel_label_mismatch=int((c.rel.str.strip() != c.calc).sum()),
                   conversion_rows=int(len(v)), conversion_new_unique=int(v.new.nunique()), conversion_new_in_corr=float(v.new.isin(c.new).mean()),
                   bad_format=int((~c.new.str.match(r"^\d{6}$")).sum() + (~c.old.str.match(r"^\d{6}$")).sum()))
print(json.dumps(chk, indent=1))
mp = load_map(); codes = sorted(mp.hs6.unique())
ex = load_ex()
res = {"unsd_integrity": chk}
res["map_codes_in_HS2017"] = int(pd.Series(codes).isin(c12.new).sum())
res["map_codes_not_in_HS2017"] = sorted(set(codes) - set(c12.new))
res["map_codes_in_HS2012_as_old"] = int(pd.Series(codes).isin(c12.old).sum())
# 대응표 코드별 HS2012·HS2007 관계
rows = []
for nm, c in [("H4", c12), ("H3", c07)]:
    cm = c[c.new.isin(codes)]
    for t, g in cm.groupby("new"):
        rows.append(dict(ver=nm, hs6=t, rel=",".join(sorted(g.rel.str.strip().unique())), olds=" ".join(sorted(g.old)), same_code=t in set(g.old) and len(g) == 1 and g.rel.str.strip().iloc[0] == "1:1"))
rv = pd.DataFrame(rows)
rv.to_csv(os.path.join(CACHE, "s3a_map_code_versions.csv"), index=False)
for nm in ["H4", "H3"]:
    r = rv[rv.ver == nm]
    res[f"map_vs_{nm}"] = dict(n=int(len(r)), identical_1to1=int(r.same_code.sum()), rel=r.rel.value_counts().to_dict())
# 보충 수집 대상: 구버전 코드 중 대응표 대상 신코드와 연결되지만 원 요청 목록(1,019)에 없는 것
need = {}
for nm, c in [("H4", c12), ("H3", c07)]:
    olds = set(c[c.new.isin(codes)].old)
    need[nm] = sorted(olds - set(codes))
    res[f"old_codes_linked_{nm}"] = len(olds); res[f"old_codes_to_fetch_{nm}"] = len(need[nm])
    # 원 요청 코드 중 구버전에 그 문자열이 존재하지 않는 코드(=수집 0이어야 정상)
    oldset = set(c.old)
    res[f"requested_codes_absent_in_{nm}"] = sorted(set(codes) - oldset)
json.dump(need, open(os.path.join(CACHE, "s3a_fetch_need.json"), "w"), indent=0)
c12.to_csv(os.path.join(CACHE, "unsd_corr_17_12.csv"), index=False); c07.to_csv(os.path.join(CACHE, "unsd_corr_17_07.csv"), index=False)
# 실제 보고 확인: 구버전에 없는 요청 코드가 H4/H3 연도에 보고됐는가
for nm, yrs in [("H4", [2015, 2016]), ("H3", [2008, 2009, 2010])]:
    ab = set(res[f"requested_codes_absent_in_{nm}"])
    e = ex[ex.year.isin(yrs)]
    res[f"absent_codes_reported_rows_{nm}"] = int(e.hs6.isin(ab).sum())
json.dump(res, open(os.path.join(CACHE, "s3a_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "unsd_integrity"}, ensure_ascii=False, indent=1)[:4000])
