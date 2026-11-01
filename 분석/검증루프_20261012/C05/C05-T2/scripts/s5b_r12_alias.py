# -*- coding: utf-8 -*-
"""s5b: R12 NPS5(원고 3.6.2 '제공 − 공개 최강 −0.6%p') 재현 후, 엄격 규칙에서 탈락하는 2019-12 별칭으로 NPS 업종을 얻은 기업을 빼고 다시 잰다.
R12 r12_tests.py의 sample/match/firm/boot 규칙을 복사(읽기만). 판정 밖 참고."""
import sys; sys.path.insert(0, __import__("os").path.dirname(__file__)); sys.stdout.reconfigure(encoding="utf-8")
from lib5 import *
R12 = os.path.join(MC, r"_shared_report\label_origin_R10\confirm_R12")
LO = os.path.join(MC, r"_shared_report\label_origin_R10\cache")
KSIC50 = ['15220', '20202', '20411', '20423', '20491', '21220', '23992', '24191', '24219', '25914', '25922', '25924',
          '25932', '26111', '26112', '26121', '26129', '26211', '26212', '26223', '26224', '26291', '26322', '26323',
          '26422', '26429', '26519', '26529', '27199', '27211', '27212', '27301', '28111', '28122', '28511', '28902',
          '29131', '29132', '29133', '29150', '29162', '29210', '29229', '29241', '29280', '30393', '31202', '31312',
          '33301', '33309']
KS5 = np.array(KSIC50 + [""], dtype=object)
LB = pd.read_parquet(os.path.join(R12, "cache", "labels.parquet"), columns=["appl_no", "split", "PROV", "H-sc-1"]); LB["appl_no"] = LB.appl_no.astype(str)
pf = pd.read_parquet(os.path.join(LO, "pat_firm.parquet")); pf["appl_no"] = pf.appl_no.astype(str)
reg = pd.read_parquet(os.path.join(R12, "cache", "reg12.parquet"))
X = pf.merge(LB, on="appl_no")
Rg = reg[["applicantCd", "nps"]].dropna(subset=["nps"]).rename(columns={"nps": "r"}); Rg = Rg[Rg.r.isin(KSIC50)]
Z = X[X.split == "eval"].merge(Rg, on="applicantCd").reset_index(drop=True)
for c in ["PROV", "H-sc-1"]:
    Z["m_" + c] = (KS5[Z[c].values.astype(int)] == Z.r.values).astype(float)
T = Z.groupby("applicantCd")[["m_PROV", "m_H-sc-1"]].mean()
A = pd.read_parquet(os.path.join(CACHE, "alias_loose_strict.parquet"))  # s5 결과(N4·N7 규칙). T1/R12 규칙은 아래에서 재구성
lk = pd.read_parquet(os.path.join(EXT, "firm_nps_applicant_link.parquet")); lk = lk[lk.nps_hit]
A2K = dict(zip(lk.applicantCd.astype(str), lk.biz6 + "|" + lk.nname))
S = json.load(open(os.path.join(CACHE, "s5_alias.json"), encoding="utf-8"))
bad = set(pd.read_parquet(os.path.join(CACHE, "r12_alias_2019.parquet")).query("strict == False").K)
allal = set(pd.read_parquet(os.path.join(CACHE, "r12_alias_2019.parquet")).K)
T["K"] = T.index.astype(str).map(A2K)
T["via_alias"] = T.K.isin(allal); T["via_bad"] = T.K.isin(bad)
rng = np.random.default_rng(20261012)
def est(TT, nb=2000):
    d = (TT["m_PROV"] - TT["m_H-sc-1"]).values
    idx = rng.integers(0, len(d), (nb, len(d)))
    b = d[idx].mean(1)
    return dict(firms=int(len(d)), prov=round(float(TT.m_PROV.mean()), 4), best=round(float(TT["m_H-sc-1"].mean()), 4),
                diff=round(float(d.mean()), 4), lo=round(float(np.percentile(b, 2.5)), 4), hi=round(float(np.percentile(b, 97.5)), 4))
out = {"재현(원고 1,633사 −0.0062 [−0.0204, +0.0104])": est(T),
       "별칭으로 업종 얻은 기업 수": int(T.via_alias.sum()), "엄격 탈락 별칭 기업 수": int(T.via_bad.sum()),
       "엄격 탈락 별칭 기업 제외": est(T[~T.via_bad]), "별칭 기업 전부 제외": est(T[~T.via_alias])}
print(json.dumps(out, ensure_ascii=False, indent=1))
json.dump(out, open(os.path.join(CACHE, "s5b_r12.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
