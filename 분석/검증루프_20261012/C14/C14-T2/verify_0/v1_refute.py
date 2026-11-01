# C14-T2 verify_0 v1: '정확' 판정 무작위 10개 독립 재계산(첫 계산과 다른 파일·다른 코드 경로). numpy seed 20261017.
import sys, json, re, hashlib, zipfile, glob
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
B = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
L = B + r"\검증루프_20261012"; RP = B + r"\보고서_20261012"; W = L + r"\C14\C14-T2"
ms = open(RP + r"\manuscript_v3.md", encoding="utf-8").read().split("\n")
def rk(x):  # 평균 순위(동률 평균) — scipy 미사용
    return pd.Series(x).rank(method="average").values
def spear(a, b):
    ra, rb = rk(a), rk(b); return float(np.corrcoef(ra, rb)[0, 1])
items = {}
def I(name):
    def deco(fn): items[name] = fn; return fn
    return deco
@I("L13 AUC 0.002~0.005: 다섯 쌍 ΔAUC 범위(원점별 per_origin 평균으로 재계산)")
def _():
    bs = pd.read_csv(L + r"\C10\C10-T2\res\boot_summary.csv", encoding="utf-8-sig")
    v = [np.mean([float(x) for x in s.split()]) for s in bs.set_index("pair").loc[["P50-FREE50", "P50-TXT50", "P50-FREEIPC50", "P50-HSC1_50", "P50-HMG3_50"], "per_origin"]]
    return dict(min=round(min(v), 4), max=round(max(v), 4), ok=round(min(v), 3) == 0.002 and round(max(v), 3) in (0.004, 0.005) and max(v) >= 0.0045)
@I("L13 해설서 단독 대비 이원 군집 CI 0 포함(C10-T2 summary.txt 30행 텍스트에서 읽기)")
def _():
    t = open(L + r"\C10\C10-T2\summary.txt", encoding="utf-8").read().split("\n")[29]
    ci = re.findall(r"\[([−\-+.\d]+),([−\-+.\d]+)\]", t.replace(" ", ""))[4]
    lo = float(ci[0].replace("−", "-")); hi = float(ci[1].replace("−", "-"))
    return dict(line=t[:40], ci5=(lo, hi), ok=lo < 0 < hi)
@I("L168 동률 전면 반전 최대 0.48(v5.log 텍스트)")
def _():
    t = open(L + r"\C11\C11-T2\verify_5\v5.log", encoding="utf-8", errors="replace").read()
    m = re.findall(r"0\.475\d*", t)
    return dict(found=m[:3], ok=bool(m))
@I("L219 최종 외부 전체 제거 5/5 seed 양수(seed별 JSON 직접)")
def _():
    d = [json.load(open(f, encoding="utf-8")) for f in sorted(glob.glob(L + r"\C12\C12-T3\verify_1\v1_final_s*.json"))]
    dd = [x["no_ext"]["skill"] - x["full"]["skill"] for x in d]
    return dict(n=len(dd), d=[round(x, 4) for x in dd], mean=round(float(np.mean(dd)), 4), ok=len(dd) == 5 and min(dd) >= 0)
@I("L286 39.3%·33.3%: 적중 건수 33/84·28/84에서")
def _():
    rs = pd.read_csv(L + r"\C01\C01-T3\rescoring_sensitivity.csv", encoding="utf-8-sig")
    li = rs[rs.label == "L_inv"].set_index("metric")["point"]
    c, s = li["hits1pct_champ"] / 84, li["hits1pct_SB"] / 84
    return dict(champ=round(c * 100, 1), sb=round(s * 100, 1), ok=round(c * 100, 1) == 39.3 and round(s * 100, 1) == 33.3)
@I("L387 전향 새 분쟁 4/65·M0 7/65(subgroup_newdispute.csv, v02와 다른 파일)")
def _():
    d = pd.read_csv(L + r"\C07\C07-T3\subgroup_newdispute.csv", encoding="utf-8-sig")
    r = d[d.section.astype(str).str.startswith("새 분쟁")].iloc[0]
    return dict(champ=r.champ_recut_top1pct_hits, M0=r.M0_recut_top1pct_hits, pos=r.pos, ok=(r.champ_recut_top1pct_hits, r.M0_recut_top1pct_hits, r.pos) == (4.0, 7.0, 65.0))
@I("L395 ρ 0.513/0.643·0.496/0.650(numpy 순위상관)")
def _():
    a = pd.read_csv(L + r"\C07\C07-T3\ksic_rank_check.csv", encoding="utf-8-sig"); b = pd.read_csv(L + r"\C08\C08-T3\industry_rank_check.csv", encoding="utf-8-sig")
    v = [spear(a.mean_p, a.realized_rate_permil), spear(a.persist_rate_permil_2023_24, a.realized_rate_permil), spear(-b.index_pred_risk_3y, -b.realized_rate_window), spear(b.persist_rate_prev_window, b.realized_rate_window)]
    v = [round(x, 3) for x in v]
    return dict(v=v, ok=v == [0.513, 0.643, 0.496, 0.65])
@I("L534 CI 하한>0 56/56·T4 강화 하한 범위(seed_bundles 원 열, variance_decomp 미사용)")
def _():
    sb = pd.read_csv(L + r"\C13\C13-T2\seed_bundles.csv", encoding="utf-8-sig")
    t4 = sb[sb.track == "T4"].strong_ci_lo
    return dict(n=len(sb), pos=int((sb.ci_lo > 0).sum()), t4=(round(t4.min(), 3), round(t4.max(), 3)), ok=len(sb) == 56 and (sb.ci_lo > 0).all() and (round(t4.min(), 3), round(t4.max(), 3)) == (-0.016, -0.003))
@I("L534 시드 sd/부트스트랩 SE 0.06–0.19(seed_bundles에서 sd 직접, SE는 variance_decomp의 공식 boot_se — sd만 독립 계산)")
def _():
    sb = pd.read_csv(L + r"\C13\C13-T2\seed_bundles.csv", encoding="utf-8-sig"); vd = pd.read_csv(L + r"\C13\C13-T2\variance_decomp.csv", encoding="utf-8-sig").set_index("track")
    r = {}
    for t, g in sb.groupby("track"):
        sd = g.skill.std(ddof=1); r[t] = round(sd / vd.loc[t, "boot_se_official"], 3)
    return dict(ratio=r, ok=round(min(r.values()), 2) == 0.06 and round(max(r.values()), 2) == 0.19)
@I("제출본 PNG 8/8 sha1 일치(zip 멤버 직접 해시, 순서 무관 집합 비교)")
def _():
    z = zipfile.ZipFile(RP + r"\보고서_초안_v15.docx")
    hz = {hashlib.sha1(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("word/media/") and n.endswith(".png")}
    hf = {hashlib.sha1(open(f, "rb").read()).hexdigest() for f in glob.glob(RP + r"\figs\fig*.png")}
    return dict(n_docx=len(hz), all_in_figs=hz <= hf, ok=len(hz) == 8 and hz <= hf)
@I("제출본 20쪽(pypdf로 쪽 수)")
def _():
    from pypdf import PdfReader
    n = len(PdfReader(RP + r"\보고서_초안_v15.pdf").pages); return dict(pages=n, ok=n == 20)
@I("PKR R8 원값 재현 68.5/71.1/68.0/68.8(c1_label_agreement.csv 원 파일과 s5 재구축 일치)")
def _():
    c1 = pd.read_csv(B + r"\모델경쟁_20260927\_shared_report\fusion_advantage\c5_R10\c1_label_agreement.csv", encoding="utf-8-sig")
    c1 = c1[c1.window == "출원2015-19_발명"].set_index("cc")["pkr5"]
    s5 = pd.read_csv(W + r"\cache\s5_pkr_agree_full.csv", encoding="utf-8-sig"); s5 = s5[s5.window == "출원2015-19_발명"].set_index("cc")["pkr5_r8"]
    d = {cc: round(abs(c1[cc] - s5[cc]) * 100, 4) for cc in ["US", "JP", "CN", "EP"]}
    return dict(absdiff_pp=d, ok=max(d.values()) < 0.1)
@I("L374 OLED 20% 상한 −0.4·액정 −6.6(C08-T1 summary.txt 문장)")
def _():
    t = open(L + r"\C08\C08-T1\summary.txt", encoding="utf-8").read()
    ok = ("−0.4" in t or "-0.4" in t) and "−6.6(S2)" in t
    return dict(ok=ok)
@I("L336 표 10 T3 메인그룹 '같다(경계)' 값 +0.0017 [−0.0004, +0.0038] 6/9(C11-T2 c11t2_table.csv)")
def _():
    t = open(L + r"\C11\C11-T2\res\c11t2_table.csv", encoding="utf-8-sig").read()
    return dict(ok=("0.0017" in t and "0.0038" in t), hit=[l[:160] for l in t.split("\n") if "0.0038" in l][:2])
@I("L143 T1 물가·산업연관표 제거 −0.00016·−0.00221(C10-T1 summary 텍스트)")
def _():
    t = open(L + r"\C10\C10-T1\summary.txt", encoding="utf-8").read()
    return dict(ok=("0.0022" in t), hits=[l[:150] for l in t.split("\n") if "0.0022" in l][:2])
names = list(items)
rng = np.random.default_rng(20261017)
pick = sorted(rng.choice(len(names), size=10, replace=False).tolist())
out = {}
for i in pick:
    try: out[names[i]] = items[names[i]]()
    except Exception as e: out[names[i]] = dict(ok=False, error=repr(e)[:300])
json.dump(dict(pool=names, picked=[names[i] for i in pick], results=out), open(W + r"\verify_0\v1_refute.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
for k, v in out.items(): print(v.get("ok"), "|", k, "|", {kk: vv for kk, vv in v.items() if kk != "ok"})
print("뒤집힘", sum(1 for v in out.values() if not v.get("ok")), "/ 10")
