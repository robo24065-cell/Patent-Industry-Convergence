# Part3 독립 재계산: NPS 업종코드(indcd) 연속 스냅샷 변경 수와 형식 변화(공백→000000, 5자리→앞0 채움) 분리
import sys, json, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T3"
p = pd.read_parquet(MC + r"\_shared\ext\firm_nps_firm_panel.parquet", columns=["applicantCd", "snap", "active", "indcd"])
snaps = sorted(p.snap.unique())
w = p[p.active].pivot_table(index="applicantCd", columns="snap", values="indcd", aggfunc="first")
def norm(s):
    s = s.astype("string")
    s2 = s.str.strip()
    s2 = s2.mask(s2.isin(["", "000000"]))
    return s2.where(s2.isna() | (s2.str.len() != 5), "0" + s2)
rows = []
for a, b in zip(snaps[:-1], snaps[1:]):
    x = w[[a, b]].dropna(how="any")
    raw = (x[a] != x[b])
    sp = (x[a].astype(str) == " ") & (x[b].astype(str) == "000000")
    pad = (x[a].astype(str).str.len() == 5) & (("0" + x[a].astype(str)) == x[b].astype(str))
    na, nb = norm(x[a]), norm(x[b])
    real = (na.notna() & nb.notna() & (na != nb))
    rows.append(dict(pair=f"{a}->{b}", n_both_active=len(x), raw_changes=int(raw.sum()), raw_rate=round(raw.mean() * 100, 2),
                     blank_to_000000=int(sp.sum()), zero_pad=int(pad.sum()), real_changes=int(real.sum()), real_rate=round(real.mean() * 100, 2)))
df = pd.DataFrame(rows); print(df.to_string())
df.to_csv(OUT + r"\nps_indcd_changes_C02.csv", index=False, encoding="utf-8-sig")
