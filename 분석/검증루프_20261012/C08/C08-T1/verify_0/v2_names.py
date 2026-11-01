# -*- coding: utf-8 -*-
# 반박 시도 3단계: (1) 쌍 안 선택의 전역 순서(비순환성) (2) 개명·시점 주장 재계산 (3) 불일치율 (4) 코드1위 vs 이름1위. 읽기 전용.
import sys, os, json, sqlite3, re, unicodedata, time
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
import pandas as pd, duckdb
t0 = time.time()
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
OUT = os.path.join(ROOT, r"분석\검증루프_20261012\C08\C08-T1\verify_0")
def V(s):
    if not isinstance(s, str): return None
    s = unicodedata.normalize("NFKC", s).lower()
    for t in ["주식회사", "(주)", "㈜", "유한회사", "재단법인", "사단법인"]:
        s = s.replace(t, "")
    s = re.sub(r"[^\w]|_", "", s)
    return s or None
r = {}
# (1) 쌍 안 일관성 (2건 이상 쌍) + 전역 순서 비순환성
J = pd.read_parquet(os.path.join(OUT, "v2_joint_tmp.parquet"))
n_ap = J.groupby("applNo").size(); anyhit = J.groupby("applNo").hit.any()
two = J[J.applNo.isin(n_ap.index[n_ap == 2]) & J.applNo.map(anyhit)]
# 둘 다 hit(같은 이름) 제외
both = two.groupby("applNo").hit.sum(); two = two[two.applNo.map(both) == 1]
pair = two.groupby("applNo").cd.agg(lambda x: "|".join(sorted(x)))
chosen = two[two.hit].groupby("applNo").cd.first()
pc = pd.DataFrame({"pair": pair, "chosen": chosen})
sz = pc.groupby("pair").size(); cons = pc.groupby("pair").chosen.nunique()
m2 = sz.index[sz >= 2]
r["pairs_ge2"] = int(len(m2)); r["pairs_ge2_consistent_share"] = round(float((cons[m2] == 1).mean()), 4)
r["docs_in_pairs_ge2"] = int(sz[m2].sum())
incons = cons[m2][cons[m2] > 1].index[:10]
r["inconsistent_pair_examples"] = []
for p in incons:
    nm = two[two.applNo.isin(pc.index[pc.pair == p])].groupby("cd").nm.first().to_dict()
    vc = pc[pc.pair == p].chosen.value_counts().to_dict()
    r["inconsistent_pair_examples"].append({nm.get(k, k): int(v) for k, v in vc.items()})
# 전역 순서: 간선 chosen -> other ; 순환 검사 (Kahn)
edges = set()
for a, row in pc.iterrows():
    x, y = row.pair.split("|"); w = row.chosen; o = y if w == x else x
    edges.add((w, o))
nodes = set([e[0] for e in edges]) | set([e[1] for e in edges])
indeg = {n: 0 for n in nodes}; adj = {n: [] for n in nodes}
for w, o in edges:
    adj[w].append(o); indeg[o] += 1
q = [n for n in nodes if indeg[n] == 0]; seen = 0
while q:
    n = q.pop(); seen += 1
    for m in adj[n]:
        indeg[m] -= 1
        if indeg[m] == 0: q.append(m)
r["order_graph_nodes"] = len(nodes); r["order_graph_edges"] = len(edges); r["nodes_in_cycles_or_after"] = len(nodes) - seen
r["two_way_edges"] = int(sum(1 for (w, o) in edges if (o, w) in edges) // 2)
# (2) 개명 사례 연도별
con = duckdb.connect(); con.sql("set memory_limit='3GB'; set threads=2; set enable_progress_bar=false")
dd = con.sql(f"select appl_year yr, trim(applicant) raw, count(distinct appl_no) n from read_parquet('{META}') where cc='KR' group by 1,2").df()
r["kr_year_range"] = [int(dd.yr.min()), int(dd.yr.max())]
r["kr_docs_by_year_tail"] = dd.groupby("yr").n.sum().tail(6).astype(int).to_dict()
chk = {"기아자동차": "기아자동차", "기아(주)": r"^기아 ?주식회사$|^기아주식회사$", "하이닉스반도체": "하이닉스반도체", "에스케이하이닉스": "에스케이하이닉스|에스케이 하이닉스",
       "삼성테크윈": "삼성테크윈", "한화테크윈": "한화테크윈", "한화에어로스페이스": "한화에어로스페이스", "삼성모바일디스플레이": "삼성모바일디스플레이",
       "포스코홀딩스": "포스코홀딩스", "포스코(정확)": r"^주식회사 ?포스코$|^포스코$|^\(주\)포스코$", "포스코퓨처엠": "포스코퓨처엠", "포스코케미칼": "포스코케미칼",
       "두산중공업": "두산중공업", "두산에너빌리티": "두산에너빌리티", "현대두산인프라코어": "현대두산인프라코어", "에이치디현대인프라코어": "에이치디현대인프라코어|HD현대인프라코어",
       "두산인프라코어": "^두산인프라코어|^주식회사 ?두산인프라코어", "에스케이온": "에스케이온", "엘지에너지솔루션": "엘지에너지솔루션", "엘지화학": "엘지화학",
       "현대중공업": "^현대중공업", "한국조선해양": "한국조선해양", "에이치디한국조선해양": "에이치디한국조선해양", "에이치디현대중공업": "에이치디현대중공업"}
out = {}
for k, rg in chk.items():
    s = dd[dd.raw.fillna("").str.contains(rg, regex=True)]
    yy = s.groupby("yr").n.sum()
    out[k] = dict(docs=int(yy.sum()), yr=[int(yy.index.min()), int(yy.index.max())] if len(yy) else None,
                  le2010=int(yy[yy.index <= 2010].sum()), y2022p=int(yy[yy.index >= 2022].sum()),
                  variants=s.groupby("raw").n.sum().sort_values(ascending=False).head(3).astype(int).to_dict())
r["rename"] = out
# (3) 제공 이름 대 등록원부 seq0 / 등록원부 '어느 출원인과도' 불일치율 (등록 KR 발명특허, 출원 연도 구간별)
kr = con.sql(f"select distinct appl_no, appl_year yr, trim(applicant) raw from read_parquet('{META}') where cc='KR' and substr(appl_no,1,2)='10'").df().drop_duplicates("appl_no")
db = sqlite3.connect("file:" + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + "?mode=ro", uri=True)
ap = pd.read_sql("select a.rgst_no, a.seq, a.applicantCd cd, a.applicantName nm, p.applNo from applicant a join patent p on a.rgst_no=p.rgst_no", db); db.close()
ap["applNo"] = ap.applNo.astype(str).str.replace("-", "")
ap = ap[ap.applNo.isin(set(kr.appl_no))]
ap["v"] = ap.nm.map(V)
kr["vp"] = kr.raw.map(V)
ap = ap.merge(kr[["appl_no", "yr", "vp"]], left_on="applNo", right_on="appl_no")
ap["hit"] = ap.v == ap.vp
g = ap.sort_values(["applNo", "seq"]).groupby("applNo")
f0 = g.head(1).set_index("applNo"); any_ = g.hit.any()
yr = f0.yr
per = {"2005-07": (2005, 2007), "2008-10": (2008, 2010), "2015-17": (2015, 2017), "2017-19": (2017, 2019), "2020-22": (2020, 2022)}
r["mismatch_vs_seq0"] = {p: round(float(1 - f0.hit[yr.between(a, b)].mean()), 4) for p, (a, b) in per.items()}
r["mismatch_vs_any_applicant"] = {p: round(float(1 - any_[yr.between(a, b)].mean()), 4) for p, (a, b) in per.items()}
r["n_registered_kr_inv_linked"] = int(any_.size); r["n_kr_inv_registered_in_meta"] = None
# 포스코 대 포스코홀딩스 쌍 수
pr = f0[~f0.hit]
pr = pr.assign(pv=pr.index.map(kr.set_index("appl_no").raw))
r["posco_pair"] = int(((pr.pv.fillna("").str.contains("포스코")) & (~pr.pv.fillna("").str.contains("홀딩스")) & (pr.nm.str.contains("포스코홀딩스"))).sum())
r["top_mismatch_pairs"] = [(a, b, int(n)) for (a, b), n in pr.groupby(["pv", "nm"]).size().sort_values(ascending=False).head(12).items()]
# (4) 코드 1위 vs 이름 1위 (KR 등록 발명특허, 출원 2017–19, KSIC별; 코드=seq0)
ks = con.sql(f"select distinct ksic, appl_no from read_parquet('{META}') where cc='KR' and substr(appl_no,1,2)='10' and appl_year between 2017 and 2019").df()
ks = ks.merge(f0[["cd", "vp"]], left_on="appl_no", right_index=True)
tc = ks.groupby(["ksic", "cd"]).size().rename("n").reset_index().sort_values(["ksic", "n"], ascending=[True, False]).drop_duplicates("ksic").set_index("ksic")
tn = ks.groupby(["ksic", "vp"]).size().rename("n").reset_index().sort_values(["ksic", "n"], ascending=[True, False]).drop_duplicates("ksic").set_index("ksic")
# 이름 1위의 대표 코드
ks2 = ks[ks.vp.values == tn.vp.reindex(ks.ksic).values]
code_of_top = ks2.groupby("ksic").cd.agg(lambda x: x.value_counts().index[0])
same = code_of_top.reindex(tc.index) == tc.cd
tot = ks.groupby("ksic").size()
r["top1_code_vs_name"] = dict(n=int(len(tc)), same=int(same.sum()), differ=sorted(tc.index[~same].tolist()),
                              share_diff_max=round(float(((tc.n - tn.n.reindex(tc.index)).abs() / tot.reindex(tc.index)).max()), 4))
print(json.dumps(r, ensure_ascii=False, indent=1))
json.dump(r, open(os.path.join(OUT, "v2_names_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print("done", round(time.time() - t0))
