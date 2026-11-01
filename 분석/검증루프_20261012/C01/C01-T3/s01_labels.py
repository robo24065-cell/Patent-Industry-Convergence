# -*- coding: utf-8 -*-
"""C01-T3 s01: N8 시험 레이블을 n8lib 없이 등록원부(sqlite mode=ro)에서 다시 뽑고, 구성(적극/소극·군집)을 낸다.
입력: registry_full.sqlite(읽기 전용), final_preds_r6.parquet(y 대조·KSIC)
출력(이 폴더): labels_rebuilt.csv(외부 반출 금지), label_composition.csv, s01_out.json, s01_log.txt"""
import os, re, json, sqlite3, sys
import numpy as np, pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
H = os.path.join(BASE, r"분석\모델경쟁_20260927\N8_특허분쟁표적레이더")
REG = os.path.join(BASE, r"수집파이프라인\registry_full.sqlite")
c = sqlite3.connect(f"file:{REG}?mode=ro", uri=True)
E = 20221231; LO, HI = 20230101, 20241231
LOG = []
def log(*a):
    s = " ".join(str(x) for x in a); LOG.append(s); print(s, flush=True)

INV, SCO = "무효(심판)예고등록", "권리범위(심판)예고등록"
# ---------------------------------------------------------------- 1) 모집단·사건 직접 추출
pt = pd.read_sql("select rgst_no, ksic, rgstDate, cndrtExptnDate from patent", c)
rt = pd.read_sql("select rgst_no, seq, rgstCsDate, rgstCsName, rgstCsReason, rgstCsPupos from right", c)
rt["d"] = pd.to_numeric(rt.rgstCsDate, errors="coerce")
ext = set(rt[rt.rgstCsName.isin(["소멸등록", "권리의 말소등록", "말소등록"]) & (rt.d <= E)].rgst_no)
rg = pd.to_numeric(pt.rgstDate, errors="coerce"); ex = pd.to_numeric(pt.cndrtExptnDate, errors="coerce")
pop = pt[(rg <= E) & (ex > E) & ~pt.rgst_no.isin(ext)].copy()
log("pop", len(pop))
ev = rt[(rt.d >= LO) & (rt.d <= HI) & rt.rgstCsName.isin([INV, SCO])].copy()
log("events", len(ev), ev.rgstCsName.value_counts().to_dict())
# 이름 변형 점검
var = rt[(rt.d >= LO) & (rt.d <= HI) & (rt.rgstCsName.str.contains("무효|권리범위", na=False))].rgstCsName.value_counts().to_dict()
log("name variants in window", var)
dup = ev.duplicated(["rgst_no", "d", "rgstCsName"]).sum(); log("dup same patent-date-type", int(dup))
evp = ev[ev.rgst_no.isin(set(pop.rgst_no))]
log("events in pop", len(evp), evp.rgstCsName.value_counts().to_dict(), "patents", evp.rgst_no.nunique(),
    "event patents total", ev.rgst_no.nunique())

P = pd.read_parquet(os.path.join(H, "final_preds_r6.parquet"), columns=["rgst_no", "ksic", "y"])
assert set(P.rgst_no) == set(pop.rgst_no), "모집단 불일치"
ypos = set(evp.rgst_no)
y_re = P.rgst_no.isin(ypos).astype(int)
log("pos rebuilt", int(y_re.sum()), "final y", int(P.y.sum()), "identical", bool((y_re.values == P.y.values).all()))

# ---------------------------------------------------------------- 2) 특허별 사건 요약
pos = sorted(ypos)
R = rt[rt.rgst_no.isin(pos)].copy()
rows = []
for r in pos:
    g = R[R.rgst_no == r].sort_values(["d", "seq"])
    w = g[(g.d >= LO) & (g.d <= HI)]
    inv_d = sorted(w[w.rgstCsName == INV].d.astype(int)); sco_d = sorted(w[w.rgstCsName == SCO].d.astype(int))
    first = min(inv_d + sco_d)
    pre_sco = g[(g.rgstCsName == SCO) & (g.d < LO)]
    pre_inv = g[(g.rgstCsName == INV) & (g.d < LO)]
    allinv = sorted(g[g.rgstCsName == INV].d.dropna().astype(int))
    # 권리범위 방향 판정(R1): 시험창 첫 권리범위 예고 이후 확정 사유
    act = pas = 0; amb = 0
    if sco_d:
        dec = g[g.rgstCsName.str.contains("확정", na=False) & ~g.rgstCsName.str.contains("기술평가|이의신청", na=False) & (g.d >= sco_d[0])]
        rs = dec.rgstCsReason.fillna("").str.replace(" ", "")
        act = int(rs.str.contains("속함").any()); pas = int(rs.str.contains("무속").any())
        # 시험창 전 권리범위 사건이 미확정 상태로 넘어왔으면 모호
        if len(pre_sco):
            last_pre = pre_sco.d.max()
            dec_pre = g[g.rgstCsName.str.contains("확정", na=False) & (g.d > last_pre) & (g.d < sco_d[0])]
            amb = int(len(dec_pre) == 0)
    # 대리 지표
    def within(ds, lo, hi):
        return int(any(lo <= x <= hi for x in ds))
    s0 = pd.Timestamp(str(sco_d[0])) if sco_d else None
    def dd(x, k):
        return int((pd.Timestamp(str(x)) + pd.Timedelta(days=k)).strftime("%Y%m%d"))
    counter = within(allinv, dd(sco_d[0], 1), dd(sco_d[0], 365)) if sco_d else 0
    prior = within(allinv, dd(sco_d[0], -365), sco_d[0]) if sco_d else 0
    corr = g[g.rgstCsName == "정정(심판)예고등록"].d.dropna().astype(int).tolist()
    corr_near = within(corr, dd(first, -30), dd(first, 30))
    corr_after = within(corr, first, dd(first, 365))
    wd = g[g.rgstCsName.str.startswith("취하등록", na=False) & (g.d >= first)]
    rows.append(dict(rgst_no=r, first_event=first, n_inv_win=len(inv_d), n_sco_win=len(sco_d),
                     first_inv=inv_d[0] if inv_d else None, first_sco=sco_d[0] if sco_d else None,
                     prior_disp_pre2023=int(len(pre_sco) + len(pre_inv) > 0), sco_active_R1=act, sco_passive_R1=pas,
                     sco_amb_pre=amb, inv_after_sco_365=counter, inv_before_sco_365=prior,
                     corr_pm30=corr_near, corr_after_365=corr_after,
                     withdrawn_after=int(len(wd) > 0),
                     withdrawn_pupos="|".join(wd.rgstCsPupos.fillna("").str[:40].tolist())))
L = pd.DataFrame(rows)
L["ksic"] = L.rgst_no.map(dict(zip(P.rgst_no, P.ksic)))

# ---------------------------------------------------------------- 3) 원점 당시 권리자(군집키, 공식 방식 재구성)
def norm_name(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = str(s)
    s = re.sub(r"\(주\)|㈜|주식회사|유한회사|\(유\)|유한책임회사|\(사\)|사단법인|재단법인|\(재\)|株式会社|co\.?,?\s*ltd\.?|inc\.?|corp(oration)?\.?|limited", "", s, flags=re.I)
    return re.sub(r"[\s\.,\-·()\[\]]", "", s).lower()
ow = pd.read_sql(f"""select rgst_no, seq, ownerName nm, ownerRgstCsDate d from owner where ownerRgstCsDate is null or ownerRgstCsDate='' or ownerRgstCsDate <= '{E}'
                    union all select rgst_no, seq, utilityModelOwnerName nm, utilityModelOwnerRgstCsDate d from utilityModelOwner
                    where utilityModelOwnerRgstCsDate is null or utilityModelOwnerRgstCsDate='' or utilityModelOwnerRgstCsDate <= '{E}'""", c)
ow = ow.sort_values(["rgst_no", "seq"]).drop_duplicates("rgst_no", keep="last")
cur = dict(zip(ow.rgst_no, ow.nm.map(norm_name)))
P["g_owner"] = [cur.get(r, "") or ("R" + r) for r in P.rgst_no]
log("owner clusters", P.g_owner.nunique())
ap = pd.read_sql("select rgst_no, seq, applicantCd from applicant", c).sort_values(["rgst_no", "seq"]).drop_duplicates("rgst_no")
atype = dict(zip(ap.rgst_no, ap.applicantCd.astype(str).str[0]))
L["owner"] = L.rgst_no.map(dict(zip(P.rgst_no, P.g_owner)))
L["indiv"] = (L.rgst_no.map(atype) == "4").astype(int)

# ---------------------------------------------------------------- 4) R1 타당성 점검(전 기간)
def trial_profile():
    T = rt[rt.rgstCsName.isin([INV, SCO])].groupby("rgst_no").rgstCsName.agg(lambda s: set(s))
    D = rt[rt.rgstCsName.str.contains("확정", na=False) & ~rt.rgstCsName.str.contains("기술평가|이의신청", na=False)].copy()
    D["rs"] = D.rgstCsReason.fillna("").str.replace(" ", "")
    Dg = D.groupby("rgst_no").rs.agg(lambda s: "|".join(s))
    inv_only = [r for r, s in T.items() if s == {INV}]; sco_only = [r for r, s in T.items() if s == {SCO}]
    io = Dg.reindex(inv_only).dropna(); so = Dg.reindex(sco_only).dropna()
    return dict(inv_only_with_decision=int(len(io)), inv_only_scope_reason=int(io.str.contains("속함|무속").sum()),
                sco_only_with_decision=int(len(so)), sco_only_invalid_reason=int(so.str.contains("무효").sum()),
                sco_only_active=int(so.str.contains("속함").sum()), sco_only_passive=int(so.str.contains("무속").sum()),
                sco_only_reason_counts=so.str.split("|").explode().value_counts().head(15).to_dict())
VAL = trial_profile()
VAL["inv_only_scope_reason_rate"] = VAL["inv_only_scope_reason"] / max(1, VAL["inv_only_with_decision"])
VAL["sco_only_invalid_reason_rate"] = VAL["sco_only_invalid_reason"] / max(1, VAL["sco_only_with_decision"])
VAL["R1_valid"] = bool(VAL["inv_only_scope_reason_rate"] < 0.02 and VAL["sco_only_invalid_reason_rate"] < 0.05)
log("R1 validation", VAL)

# ---------------------------------------------------------------- 5) R2 대리 지표 보정(2008–2022 권리범위 사건, R1 판정분)
sco_all = rt[rt.rgstCsName == SCO].dropna(subset=["d"])
one = sco_all.groupby("rgst_no").filter(lambda g: len(g) == 1)
one = one[(one.d >= 20080101) & (one.d <= 20221231)]
Dall = rt[rt.rgstCsName.str.contains("확정", na=False) & ~rt.rgstCsName.str.contains("기술평가|이의신청", na=False)].copy()
Dall["rs"] = Dall.rgstCsReason.fillna("").str.replace(" ", "")
invd = rt[rt.rgstCsName == INV].dropna(subset=["d"]).groupby("rgst_no").d.apply(lambda s: sorted(s.astype(int)))
cal = []
for r, d0 in zip(one.rgst_no, one.d.astype(int)):
    dec = Dall[(Dall.rgst_no == r) & (Dall.d >= d0)].rs
    a = dec.str.contains("속함").any(); p = dec.str.contains("무속").any()
    if a == p:
        continue
    t0 = pd.Timestamp(str(d0)); ds = invd.get(r, [])
    ts = [pd.Timestamp(str(x)) for x in ds]
    cal.append(dict(active=int(a), counter=int(any(t0 < x <= t0 + pd.Timedelta(days=365) for x in ts)),
                    prior=int(any(t0 - pd.Timedelta(days=365) <= x <= t0 for x in ts)), indiv=int(atype.get(r, "") == "4"),
                    yr=d0 // 10000))
C = pd.DataFrame(cal)
log("calibration n", len(C), "active share", round(C.active.mean(), 3))
cell = C.groupby(["counter", "prior", "indiv"]).active.agg(["mean", "size"]).reset_index()
log(cell.to_string())
overall = C.active.mean()
cmap = {(int(a), int(b), int(d)): (m if n >= 10 else overall) for a, b, d, m, n in cell.itertuples(index=False)}
# 단변량 비율
uni = {k: C.groupby(k).active.agg(["mean", "size"]).to_dict() for k in ["counter", "prior", "indiv"]}

# 시험 사건 방향
L["sco_dir"] = np.where(L.n_sco_win == 0, "none",
               np.where(L.sco_active_R1 == 1, np.where(L.sco_passive_R1 == 1, "mixed", "active"),
               np.where(L.sco_passive_R1 == 1, "passive", "undetermined")))
L["p_active"] = np.where(L.sco_dir.isin(["active", "mixed"]), 1.0, np.where(L.sco_dir == "passive", 0.0, np.nan))
und = L.sco_dir == "undetermined"
L.loc[und, "p_active"] = [cmap.get((a, b, d), overall) for a, b, d in zip(L[und].inv_after_sco_365, L[und].inv_before_sco_365, L[und].indiv)]
L.loc[L.n_sco_win == 0, "p_active"] = 0.0

# ---------------------------------------------------------------- 6) 군집: 같은 날·같은 KSIC 묶음
def bundles(L, days):
    par = {}
    def f(x):
        par.setdefault(x, x)
        while par[x] != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    def u(a, b):
        par[f(a)] = f(b)
    for r, o in zip(L.rgst_no, L.owner):
        u("P" + r, "O" + o)
    Ls = L.sort_values("first_event")
    ts = {r: pd.Timestamp(str(d)) for r, d in zip(Ls.rgst_no, Ls.first_event)}
    for k, g in Ls.groupby("ksic"):
        rs = list(g.rgst_no)
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                if abs((ts[rs[j]] - ts[rs[i]]).days) <= days:
                    u("P" + rs[i], "P" + rs[j])
    return par, f
for nm, days in (("bundle", 0), ("bundle30", 30)):
    par, f = bundles(L, days)
    L[nm] = [f("P" + r) for r in L.rgst_no]
    # 같은 owner를 공유하는 음성도 같은 묶음 루트로
    omap = {o: f("O" + o) for o in L.owner.unique()}
    P["g_" + nm] = [omap.get(o, "O" + o) for o in P.g_owner]
    log(nm, "pos bundles", L[nm].nunique())
log("pos owner clusters", L.owner.nunique())

# ---------------------------------------------------------------- 7) 구성 표
comp = []
def add(sec, key, n, extra=""):
    comp.append(dict(section=sec, key=str(key), n=n, share_of_108=round(n / len(L), 4), note=extra))
add("type", "무효만", int(((L.n_inv_win > 0) & (L.n_sco_win == 0)).sum()))
add("type", "권리범위만", int(((L.n_inv_win == 0) & (L.n_sco_win > 0)).sum()))
add("type", "무효+권리범위", int(((L.n_inv_win > 0) & (L.n_sco_win > 0)).sum()))
for k, v in L.sco_dir.value_counts().items():
    add("scope_dir_R1", k, int(v))
add("scope_active", "R1 적극(혼합 포함) 하한", int(L.sco_dir.isin(["active", "mixed"]).sum()))
add("scope_active", "점추정(R1+R2 확률합)", float(round(L.p_active.sum(), 2)))
add("scope_active", "상한(R1 적극+미판정 전부)", int(L.sco_dir.isin(["active", "mixed", "undetermined"]).sum()))
only_sco = L.n_inv_win == 0
add("scope_active_only", "적극만으로 양성(하한)", int((only_sco & L.sco_dir.isin(["active", "mixed"])).sum()))
add("scope_active_only", "적극만으로 양성(점추정)", float(round(L[only_sco].p_active.sum(), 2)))
add("scope_active_only", "적극만으로 양성(상한)", int((only_sco & L.sco_dir.isin(["active", "mixed", "undetermined"])).sum()))
add("proxy", "권리범위 후 365일 내 무효(반격형)", int(L.inv_after_sco_365.sum()), "권리범위 양성 중")
add("proxy", "권리범위 전 365일 내·같은 날 무효", int(L.inv_before_sco_365.sum()), "권리범위 양성 중")
add("proxy", "첫 사건 ±30일 정정심판 예고", int(L.corr_pm30.sum()))
add("proxy", "첫 사건 후 365일 내 정정심판 예고", int(L.corr_after_365.sum()))
add("proxy", "첫 사건 후 취하 등록", int(L.withdrawn_after.sum()))
add("proxy", "2023 전 분쟁 이력", int(L.prior_disp_pre2023.sum()))
add("proxy", "개인 출원(첫 출원인코드 4)", int(L.indiv.sum()))
oc = L.owner.value_counts()
add("owner", "양성 권리자 군집 수", int(len(oc)))
for i, v in enumerate(oc.head(5).values):
    add("owner", f"상위{i+1} 권리자 양성", int(v))
add("owner", "상위5 권리자 합", int(oc.head(5).sum()))
sd = L.groupby(["owner", "first_event"]).size()
for k, v in sd.value_counts().sort_index().items():
    add("same_day_owner_bundle_size", f"크기{k}", int(v), "묶음 수(같은 권리자·같은 첫 사건일)")
add("same_day_owner_bundle_size", "2건 이상 묶음에 든 양성", int(sd[sd >= 2].sum()))
sdk = L.groupby(["ksic", "first_event"]).size()
add("same_day_ksic", "같은 KSIC·같은 날 2건 이상 묶음에 든 양성", int(sdk[sdk >= 2].sum()), f"묶음 {int((sdk>=2).sum())}개")
for nm in ("bundle", "bundle30"):
    bc = L[nm].value_counts()
    add(nm, "양성 묶음 수(실효 표본)", int(len(bc)))
    add(nm, "최대 묶음 크기", int(bc.max()))
kc = L.ksic.value_counts()
add("ksic", "양성 KSIC 수", int(len(kc)))
for k, v in kc.head(5).items():
    add("ksic", f"KSIC {k}", int(v))
add("ksic", "상위3 KSIC 합", int(kc.head(3).sum()))
yr = (L.first_event // 10000).value_counts().sort_index()
for k, v in yr.items():
    add("year_first_event", int(k), int(v))
pd.DataFrame(comp).to_csv(os.path.join(OUT, "label_composition.csv"), index=False, encoding="utf-8-sig")

# 레이블 파일(외부 반출 금지)
with open(os.path.join(OUT, "labels_rebuilt.csv"), "w", encoding="utf-8-sig", newline="") as fh:
    fh.write("# 외부 반출 금지(INTERNAL ONLY): 개별 특허 레이블·권리자. 공모전 제출물·외부 서비스에 쓰지 않는다.\n")
    L.drop(columns=["withdrawn_pupos"]).to_csv(fh, index=False)
P[["rgst_no", "g_owner", "g_bundle", "g_bundle30"]].to_parquet(os.path.join(OUT, "_clusters_internal.parquet"), index=False)
out = dict(pop=len(pop), events=len(ev), events_in_pop=len(evp), pos=int(y_re.sum()), labels_identical=bool((y_re.values == P.y.values).all()),
           n_owner_clusters=int(P.g_owner.nunique()), R1_validation=VAL, calib_n=len(C), calib_active_share=float(overall),
           calib_cells=cell.to_dict("records"), calib_univariate=uni, withdrawn_pupos_examples=L.withdrawn_pupos[L.withdrawn_pupos != ""].head(10).tolist(),
           calib_by_year=C.groupby("yr").active.agg(["mean", "size"]).reset_index().to_dict("records"))
json.dump(out, open(os.path.join(OUT, "s01_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
open(os.path.join(OUT, "s01_log.txt"), "w", encoding="utf-8").write("\n".join(LOG))
print("done")
