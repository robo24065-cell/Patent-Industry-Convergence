# verify_4: 형식 정규화(공백·'000000' 자리표시 제외, 앞 0 채움) 후 실제 업종코드 변경 수
import os, pandas as pd
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
EXT = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\ext")
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C01\C01-T4\verify_4")
d = pd.read_parquet(os.path.join(EXT, "firm_nps_firm_panel.parquet"), columns=["applicantCd", "snap", "active", "indcd"])
d = d[d.active]
raw = d.indcd.fillna("")
d["blank"] = raw.str.strip().isin(["", "000000"])
d["norm"] = raw.str.strip().str.zfill(6)
print(d.groupby("snap").agg(blank_space=("indcd", lambda s: (s == " ").sum()), z000000=("indcd", lambda s: (s == "000000").sum()), empty=("indcd", lambda s: (s.fillna("") == "").sum()), len5=("indcd", lambda s: (s.fillna("").str.len() == 5).sum())).to_string())
snaps = sorted(d.snap.unique()); rows = []
for s0, s1 in zip(snaps[:-1], snaps[1:]):
    x = d[d.snap == s0].merge(d[d.snap == s1], on="applicantCd")
    raw_ne = x.indcd_x.fillna("").ne("") & x.indcd_y.fillna("").ne("")
    rawch = (raw_ne & (x.indcd_x != x.indcd_y)).sum()
    v = x[~x.blank_x & ~x.blank_y]
    ch = v.norm_x != v.norm_y
    rows.append(dict(s0=s0, s1=s1, raw_changes_as_in_data_check=int(rawch), valid_pairs=len(v), real_changes=int(ch.sum()), real_pct=round(ch.mean()*100, 3), real_2digit=int((v.norm_x.str[:2] != v.norm_y.str[:2]).sum())))
r = pd.DataFrame(rows); print(r.to_string()); r.to_csv(os.path.join(OUT, "v2_clean_changes.csv"), index=False, encoding="utf-8-sig")
