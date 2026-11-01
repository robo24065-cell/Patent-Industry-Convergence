# C14-T2 s9: 그림 원천 CSV의 대표 숫자가 v15 본문(부록 앞)과 같은지 + 그림 PNG가 figs 원본과 같은지(이미 s2) — v15에서 바뀐 줄과 겹치는 그림 숫자 중심
import sys, re, json
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
R = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
W = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C14\C14-T2"
md = open(R + r"\manuscript_v3.md", encoding="utf-8").read()
body = md[:md.find("# [부록")]
norm = lambda s: s.replace("−", "-")
B = norm(body)
checks = []
def chk(fig, label, token):
    checks.append(dict(fig=fig, item=label, token=token, in_body=norm(token) in B))
f1 = pd.read_csv(R + r"\figs\fig01_framework.csv", encoding="utf-8-sig")
for it in f1["항목"]:
    for m in re.findall(r"\d[\d.]*(?:–\d[\d.]*)?%|\d+/\d+", str(it)): chk("fig01", it, m)
f2 = pd.read_csv(R + r"\figs\fig02_scorecard_9tracks.csv", encoding="utf-8-sig")
for _, r in f2.iterrows():
    for c in ["AUC_모형", "AUC_기준선", "skill"]:
        v = r[c]; s = f"{v:.3f}"
        chk("fig02", f"{r['트랙']} {c}", s)
    chk("fig02", f"{r['트랙']} CI", f"[{r['skill_CI_lo']:.3f}, {r['skill_CI_hi']:.3f}]")
f4 = pd.read_csv(R + r"\figs\fig04_within_code_divergence.csv", encoding="utf-8-sig")
for _, r in f4[f4["ksic"].astype(str).isin(["26211", "26212", "26111"]) & f4["KR출원_연성장%(08-09→16-17)"].notna()].iterrows():
    v = r["KR출원_연성장%(08-09→16-17)"]; chk("fig04", f"{r['ksic']} KR출원", f"{abs(v):.1f}%")
f8 = pd.read_csv(R + r"\figs\fig08_t1_waiting_room_index.csv", encoding="utf-8-sig")
for _, r in f8.head(5).iterrows(): chk("fig08", r["name"], f"{r['index_share']*100:.1f}%")
f9 = pd.read_csv(R + r"\figs\fig09_n9_topk_vs_baseline.csv", encoding="utf-8-sig")
r = f9.iloc[0]; chk("fig09", "상위1% 챔피언", f"{r['챔피언']*100:.1f}%"); chk("fig09", "상위1% 기준선", f"{r['M0_L2(같은 정보 시점 기준선)']*100:.1f}%")
f5 = pd.read_csv(R + r"\figs\fig05_rta_map_5x50.csv", encoding="utf-8-sig")
chk("fig05", "한국 1위 수", f"{int((f5.leader == 'KR').sum())}개")
chk("fig05", "한국 RTA>1 수", f"{int((f5.KR > 1).sum())}개")
for _, r in f5[f5.leader == "KR"].head(3).iterrows(): chk("fig05", r["name"], f"{r['KR']:.2f}")
T = pd.DataFrame(checks); T.to_csv(W + r"\cache\s9_fig_numbers.csv", index=False, encoding="utf-8-sig")
print(T.groupby("fig").in_body.agg(["sum", "count"]).to_string()); print(T[~T.in_body].to_string())
