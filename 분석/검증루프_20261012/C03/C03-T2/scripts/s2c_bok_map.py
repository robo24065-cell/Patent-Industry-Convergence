# -*- coding: utf-8 -*-
"""s2c: 한국은행 IO–KSIC–HS 대조표 원본에서 대응표 구조 진단(비대상 KSIC와 공유하는 IO 코드, 동일 HS 묶음 KSIC쌍, 미연결 KSIC 2개 사유)."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import *
conc = os.path.join(ROOT, "공모전데이터", "외부데이터", "산업연관표", "IO코드와 KSIC 및 HS코드 대조표.xlsx")
x = pd.ExcelFile(conc); print(x.sheet_names)
for s in x.sheet_names:
    d = pd.read_excel(x, s, header=None, dtype=str, nrows=6); print("--", s); print(d.to_string()[:800])
k = pd.read_excel(conc, sheet_name="IO코드와 KSIC 비교표", header=1, dtype=str)
h = pd.read_excel(conc, sheet_name="IO코드와 HS코드 비교표", header=1, dtype=str)
k.columns = ["io", "nm", "ksic"]; h.columns = ["io", "nm", "hs"]
k["io"] = k.io.ffill().str.strip(); h["io"] = h.io.ffill().str.strip()
k = k.dropna(subset=["ksic"]).copy(); h = h.dropna(subset=["hs"]).copy()
k["ksic"] = k.ksic.str.strip(); h["hs"] = h.hs.str.strip()
print("ksic len dist", k.ksic.str.len().value_counts().to_dict(), "hs len dist", h.hs.str.len().value_counts().to_dict())
k["k5"] = k.ksic.str[:5]; h["hs6"] = h.hs.str.zfill(6)
prof = pd.read_csv(os.path.join(MAP5, "m2_ksic_profile.csv"), dtype={"ksic": str})
tgt = set(prof.ksic)
mp = load_map()
res = {}
# 미연결 2개
for kk in ["25914", "25922"]:
    rows = k[k.k5 == kk]
    res[f"unlinked_{kk}"] = {"io_rows": rows[["io", "nm", "ksic"]].values.tolist(),
                              "io_hs_count": {io: int((h.io == io).sum()) for io in rows.io.unique()},
                              "io_other_target_ksic": {io: sorted(set(k[(k.io == io)].k5) & tgt - {kk}) for io in rows.io.unique()}}
# 대상 IO 코드별 전체 KSIC(비대상 포함) 수
rows = []
for io, g in k[k.io.isin(mp.io.unique())].groupby("io"):
    allk = sorted(set(g.k5)); t = sorted(set(allk) & tgt)
    rows.append(dict(io=io, io_name=g.nm.dropna().iloc[0] if g.nm.notna().any() else "", n_ksic5_all=len(allk), n_ksic5_target=len(t),
                     target=" ".join(t), nontarget=" ".join(sorted(set(allk) - tgt)), n_hs6=int(h[h.io == io].hs6.nunique())))
io_df = pd.DataFrame(rows)
io_df.to_csv(os.path.join(CACHE, "s2c_io_structure.csv"), index=False, encoding="utf-8-sig")
print(io_df.to_string())
# KSIC별: HS 묶음이 IO 단위라 같은 IO의 대상 KSIC는 수출액이 동일
same = io_df[io_df.n_ksic5_target > 1][["io", "target"]].values.tolist()
res["io_with_multiple_target_ksic"] = same
# KSIC별 IO 수 / 비대상 공유 비율
kio = mp.groupby("k5").io.nunique()
res["ksic_io_count_dist"] = kio.value_counts().to_dict()
nt = io_df.set_index("io")
mp2 = mp.merge(io_df[["io", "n_ksic5_all", "n_ksic5_target"]], on="io")
kshare = mp2.groupby("k5").apply(lambda g: float((g.n_ksic5_all > g.n_ksic5_target).mean()))
res["ksic_share_hs6_in_io_with_nontarget_ksic"] = kshare.round(3).to_dict()
res["n_ksic_all_hs6_shared_with_nontarget"] = int((kshare == 1).sum())
json.dump(res, open(os.path.join(CACHE, "s2c_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)[:5000])
