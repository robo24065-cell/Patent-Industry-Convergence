# -*- coding: utf-8 -*-
# 반박 시도 2단계: 공동출원 선택에 '일정한 규칙'이 있는지 추가 후보(빈도·소유자·출원 당시 seq 등) 검정. 읽기 전용.
import sys, os, json, sqlite3, re, unicodedata, time
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C08\C08-T1\verify_0")
def V(s):
    if not isinstance(s, str): return None
    s = unicodedata.normalize("NFKC", s).lower()
    for t in ["주식회사", "(주)", "㈜", "유한회사", "재단법인", "사단법인"]:
        s = s.replace(t, "")
    s = re.sub(r"[^\w]|_", "", s)
    return s or None
J = pd.read_parquet(os.path.join(OUT, "v2_joint_tmp.parquet"))
db = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + "?mode=ro", uri=True)
allap = pd.read_sql("select applicantCd cd, count(*) n from applicant group by 1", db).set_index("cd").n
rg = list(set(J.rgst_no))
ow = []
for i in range(0, len(rg), 900):
    ch = rg[i:i + 900]
    ow.append(pd.read_sql(f"select rgst_no, seq, ownerName, finalOwnerYn from owner where rgst_no in ({','.join('?'*len(ch))})", db, params=ch))
ow = pd.concat(ow); db.close()
r = {}
J["freq"] = J.cd.map(allap).fillna(0)
f = J.sort_values(["applNo", "freq", "seq"], ascending=[True, False, True]).groupby("applNo").head(1).set_index("applNo")
r["match_most_frequent_applicant"] = round(float(f.hit.mean()), 4)
# 동률 없는 2인 문서에서만
n_ap = J.groupby("applNo").size(); anyhit = J.groupby("applNo").hit.any()
two = J[J.applNo.isin(n_ap.index[n_ap == 2]) & J.applNo.map(anyhit)]
fr = two.groupby("applNo").freq.agg(["max", "min"]); untied = fr.index[fr["max"] != fr["min"]]
f2 = two[two.applNo.isin(untied)].sort_values(["applNo", "freq"], ascending=[True, False]).groupby("applNo").head(1)
r["cond_two_untied_most_frequent"] = round(float(f2.hit.mean()), 4); r["cond_two_untied_n"] = int(len(untied))
# 소유자(등록원부 owner) 대비
ow["v"] = ow.ownerName.map(V)
pv = J.drop_duplicates("applNo").set_index("rgst_no").vp
ow["vp"] = ow.rgst_no.map(pv); ow["hit"] = ow.v == ow.vp
og = ow.sort_values(["rgst_no", "seq"]).groupby("rgst_no")
r["owner_any"] = round(float(og.hit.any().mean()), 4)
r["owner_seq_first"] = round(float(og.head(1).hit.mean()), 4)
r["owner_final_Y_any"] = round(float(ow[ow.finalOwnerYn == "Y"].groupby("rgst_no").hit.any().mean()), 4)
r["owner_count_per_doc_mean"] = round(float(og.size().mean()), 3)
# 출원번호 끝자리 등 무작위성 검사: first/last 일치가 출원번호 홀짝과 관련?
d1 = J.sort_values(["applNo", "seq"]).groupby("applNo").head(1).copy()
d1["odd"] = d1.applNo.str[-1].astype(int) % 2
r["first_match_by_applno_parity"] = d1.groupby("odd").hit.mean().round(4).to_dict()
# 같은 출원인 쌍(코드 집합)이 여러 문서에 나올 때 같은 쪽이 선택되는지 (쌍 안 일관성)
tw = two.copy()
pair = tw.groupby("applNo").cd.agg(lambda x: "|".join(sorted(x)))
chosen = tw[tw.hit].groupby("applNo").cd.first()
pc = pd.DataFrame({"pair": pair, "chosen": chosen}).dropna()
cons = pc.groupby("pair").chosen.agg(lambda x: x.value_counts().iloc[0] / len(x))
sz = pc.groupby("pair").size()
big = sz.index[sz >= 5]
r["pairs_with_ge5_docs"] = int(len(big)); r["docs_in_those"] = int(sz[big].sum())
r["within_pair_consistency_weighted"] = round(float((cons[big] * sz[big]).sum() / sz[big].sum()), 4)
r["within_pair_fully_consistent_share"] = round(float((cons[big] == 1).mean()), 4)
# 쌍 안 일관성이 높다면, 쌍 안에서 어떤 쪽이 고정적으로 선택되는지(예: 대학 산학협력단 vs 기업)
top_pairs = sz[big].sort_values(ascending=False).head(12)
ex = []
for p in top_pairs.index:
    names = tw[tw.applNo.isin(pc.index[pc.pair == p])].groupby("cd").nm.first().to_dict()
    vc = pc[pc.pair == p].chosen.value_counts().to_dict()
    ex.append(dict(pair={k: names.get(k) for k in p.split("|")}, chosen_counts={names.get(k, k): int(v) for k, v in vc.items()}))
r["top_pairs"] = ex
print(json.dumps(r, ensure_ascii=False, indent=1))
json.dump(r, open(os.path.join(OUT, "v2_rules_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
