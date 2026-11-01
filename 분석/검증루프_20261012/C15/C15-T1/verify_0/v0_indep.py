# -*- coding: utf-8 -*-
"""C15-T1 verify_0: 독립 재계산(작업 스크립트 s2/s3a/s4 코드·중간 산출 미사용; sqlite3 SQL + pandas.read_excel + 순위합 AUC).
 V1 제공 이탈 X^p_y(SQL 'exit 뒤 회복 없음' 의미론)·공식 X_y(연보 원표 직접)·r_F·r_cum
 V2 T4 최종 모집단 실용신안 수(sqlite rowid 순 pid 체계)
 V3 S3 레이블(납부기록, DATA_END vs 스냅샷 끝) SQL 재구성과 뒤집힘 수
 V4 S0·S2·S3·S4 점 추정(순위합 AUC, 가중은 직접 쌍 가중 공식)
 반박: R1 분모(연말 s·연초 s) R2 상태일 12-31 R3 내부 누락 SQL R4 S2 무작위 KSIC 귀무 R5 4.2 교차가 E 계산 산물인지(소멸등록일 기준 h=5·7·10)
사용: python -I v0_indep.py"""
import os, sys, json, sqlite3, time
import numpy as np, pandas as pd
t0 = time.time()
sys.stdout.reconfigure(encoding="utf-8")
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
DB = os.path.join(BASE, "수집파이프라인", "registry_full.sqlite")
U = os.path.join(BASE, r"분석\검증루프_20261012\C13\C13-T1\download\moip_unzip\f_437_1")
TR = os.path.join(BASE, r"분석\모델경쟁_20260927\T4_특허포기예측"); FD = os.path.join(TR, r"R2\runs_final\F22")
R = {}
# ---------------- 공식 원표(pandas.read_excel)
f1 = [os.path.join(U, x) for x in os.listdir(U) if "총괄" in x][0]
xl = pd.read_excel(f1, sheet_name=None, header=None)
def series(key):
    sh = [s for s in xl if key in s.replace(" ", "")][0]
    d = xl[sh]; d = d[pd.to_numeric(d[0], errors="coerce").between(1940, 2030) & pd.to_numeric(d[1], errors="coerce").notna()]
    return pd.Series(pd.to_numeric(d[1]).values, index=pd.to_numeric(d[0]).astype(int).values)
I = series("연도별존속권리"); Rg = series("1.5.연도별등록")
R["official_I_2021_2025"] = I.loc[2021:2025].tolist(); R["official_R_2022_2025"] = Rg.loc[2022:2025].tolist()
# ---------------- 제공 쪽 SQL
src = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
mem = sqlite3.connect(":memory:")
pt = pd.read_sql("select rowid rid, rgst_no, ksic, rgstDate, cndrtExptnDate, applDate from patent where substr(rgst_no,1,2)='10' and rgstDate is not null and length(rgstDate)>=8 order by rowid", src)
pt.to_sql("p", mem, index=False)
ev = pd.read_sql("""select rgst_no, substr(rgstCsDate,1,8) d,
  case when rgstCsName like '회복등록%' then 1 else 0 end isrec
  from right where substr(rgst_no,1,2)='10' and length(rgstCsDate)>=8 and (
   (rgstCsName='소멸등록') or rgstCsName in ('권리의 말소등록','말소등록','직권말소등록') or rgstCsName like '회복등록%'
   or (rgstCsName like '%확정%' and rgstCsName not like '%기술평가%' and rgstCsName not like '%이의신청%' and rgstCsReason like '%무효%' and rgstCsReason not like '%정정%'))""", src)
ev.to_sql("e", mem, index=False)
mem.execute("create index ie on e(rgst_no, d)"); mem.execute("create index ip on p(rgst_no)")
def alive(yend, st):
    q = f"""select count(*) from p where substr(rgstDate,1,8) <= '{yend}' and not exists (
      select 1 from e x where x.rgst_no=p.rgst_no and x.isrec=0 and x.d <= '{st}'
      and not exists (select 1 from e r where r.rgst_no=x.rgst_no and r.isrec=1 and r.d >= x.d and r.d <= '{st}'))"""
    return mem.execute(q).fetchone()[0]
def regs(y): return mem.execute(f"select count(*) from p where substr(rgstDate,1,4)='{y}'").fetchone()[0]
A = {y: alive(f"{y}1231", f"{y+1}0531") for y in range(2016, 2026)}
Ad = {y: alive(f"{y}1231", f"{y}1231") for y in range(2020, 2026)}
Rp = {y: regs(y) for y in range(2002, 2026)}
R["A_may"] = A; R["A_dec"] = Ad
def rF(Ad_, sfun):
    num = sum(Ad_[y - 1] + Rp[y] - Ad_[y] for y in range(2022, 2026))
    den = sum((I[y - 1] + Rg[y] - I[y]) * sfun(y) for y in range(2022, 2026))
    return num / den
s = {y: A[y] / I[y] for y in A}
s3 = {y: (A[y] - sum(Rp[t] for t in range(y - 2, y + 1))) / (I[y] - sum(Rg[t] for t in range(y - 2, y + 1))) for y in range(2018, 2026)}
R["r_F_s3"] = rF(A, lambda y: (s3[y - 1] + s3[y]) / 2)
R["R1_r_F_s_avg"] = rF(A, lambda y: (s[y - 1] + s[y]) / 2)
R["R1_r_F_s_prev"] = rF(A, lambda y: s[y - 1])
sd = {y: Ad[y] / I[y] for y in Ad}
R["R2_r_F_dec_status"] = rF(Ad, lambda y: (sd[y - 1] + sd[y]) / 2)
R["r_by_year_s3"] = {y: (A[y - 1] + Rp[y] - A[y]) / ((I[y - 1] + Rg[y] - I[y]) * (s3[y - 1] + s3[y]) / 2) for y in range(2019, 2026)}
# 코호트 누적 2002–2023
a23post = mem.execute("""select count(*) from p where substr(rgstDate,1,4) between '2002' and '2023' and not exists (
      select 1 from e x where x.rgst_no=p.rgst_no and x.isrec=0 and x.d <= '20240531'
      and not exists (select 1 from e r where r.rgst_no=x.rgst_no and r.isrec=1 and r.d >= x.d and r.d <= '20240531'))""").fetchone()[0]
sRp = sum(Rp[y] for y in range(2002, 2024)); sR = sum(Rg[y] for y in range(2002, 2024))
R["r_cum_2023"] = ((sRp - a23post) / sRp) / ((sR - I[2023]) / sR)
# R3 내부 누락: 마지막 납부 연차로 본 납부완료일 + 583일 < 납부 최대일, 만료 전, 불납 소멸·말소·무효·만료 없음
pay = pd.read_sql("select rgst_no, max(cast(lastAnnl as integer)) l from pay where substr(rgst_no,1,2)='10' group by rgst_no", src)
pmax = src.execute("select max(payDate) from pay").fetchone()[0]
pp = pt.merge(pay, on="rgst_no", how="left")
pp["rd"] = pd.to_datetime(pp.rgstDate.str[:8], format="%Y%m%d"); pp["ex"] = pd.to_datetime(pp.cndrtExptnDate.str[:8], format="%Y%m%d", errors="coerce")
pp["thru"] = [r + pd.DateOffset(years=int(l)) for r, l in zip(pp.rd, pp.l.fillna(0))]
anyexit = set(ev[ev.isrec == 0].rgst_no)
due = (pp.l > 0) & (pp.thru + pd.Timedelta(days=583) < pd.Timestamp(pmax[:8])) & (pp.thru < pp.ex)
R["R3_internal"] = dict(due=int(due.sum()), due_no_exit_event=int((due & ~pp.rgst_no.isin(anyexit)).sum()),
                        rate=float((due & ~pp.rgst_no.isin(anyexit)).sum() / due.sum()))
# ---------------- V2 실용신안 혼입
evp = pd.read_parquet(os.path.join(FD, "ev.parquet"), columns=["pid", "acd", "atype", "ksic", "y"])
rg = pt.rgst_no.values[evp.pid.values]
R["V2_final_pop_non10"] = int((pd.Series(rg).str[:2] != "10").sum()); R["V2_ksic_match"] = float((pt.ksic.values[evp.pid.values].astype(str) == evp.ksic.astype(str).values).mean())
# ---------------- V3 S3 레이블 재구성(SQL 납부완료 연차)
def thru_at(de):
    q = pd.read_sql(f"select rgst_no, max(cast(lastAnnl as integer)) l from pay where substr(rgst_no,1,2)='10' and substr(payDate,1,8) <= '{de}' group by rgst_no", src)
    m = pd.Series(q.l.values, index=q.rgst_no.values)
    l = m.reindex(rg).fillna(0).values.astype(int)
    rdv = pd.to_datetime(pt.rgstDate.str[:8].values[evp.pid.values], format="%Y%m%d")
    return pd.to_datetime([pd.Timestamp(r) + pd.DateOffset(years=int(k)) for r, k in zip(rdv, l)])
exv = pd.to_datetime(pt.cndrtExptnDate.str[:8].values[evp.pid.values], format="%Y%m%d", errors="coerce")
ma = pd.read_sql("select distinct rgst_no from right where rgstCsName in ('권리의 말소등록','말소등록','직권말소등록') and substr(rgstCsDate,1,8) > '20220101' and substr(rgstCsDate,1,8) <= '20250101'", src)
mz = pd.Series(rg).isin(set(ma.rgst_no)).values
def lab(de):
    th = thru_at(de); end = pd.Timestamp("2025-01-01")
    one = (th > pd.Timestamp("2022-01-01")) & (th <= end) & (th < exv) & (th + pd.Timedelta(days=213) <= pd.Timestamp(de)) & ~mz
    return np.asarray(one).astype(int)
yv0 = lab("20250822"); yv3 = lab(pmax[:8])
R["V3"] = dict(match_official=float((yv0 == evp.y.values).mean()), pos_de=int(yv0.sum()), pos_snapshot=int(yv3.sum()),
               flips_1to0=int(((yv0 == 1) & (yv3 == 0)).sum()), flips_0to1=int(((yv0 == 0) & (yv3 == 1)).sum()))
# ---------------- V4 점 추정(순위합; 가중은 쌍 가중)
FS = json.load(open(os.path.join(TR, r"R2\final_spec_r2.json"), encoding="utf-8")); ws = FS["spec"]["ensemble"]["weights"]
files = {"cls": "cls_99f270b3cd.npy", "haz": "haz_0568d4297b.npy", "logit": "logit_3008b8b195.npy", "hgb": "hgb_b4bd0fa46a.npy", "base": "base_46526c56d8.npy"}
P = {k: np.load(os.path.join(FD, "pred", f)) for k, f in files.items()}
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
pm = 1 / (1 + np.exp(-(sum(w * lg(P[k]) for k, w in ws.items() if w) / sum(w for w in ws.values() if w)))); pb = P["base"]
def wauc(yy, s, w=None):
    w = np.ones(len(yy)) if w is None else w
    d = pd.DataFrame({"s": s, "p": w * yy, "n": w * (1 - yy)}).groupby("s")[["p", "n"]].sum().sort_index()
    cn = d.n.cumsum() - d.n
    return float((d.p * (cn + 0.5 * d.n)).sum() / (d.p.sum() * d.n.sum()))
sk = lambda a, b: (a - b) / (1 - b)
y = evp.y.values
R["V4_S0"] = sk(wauc(y, pm), wauc(y, pb))
ks = pd.read_csv(os.path.join(W, "cache", "s3a_ksic.csv"), dtype={"ksic": str})   # 제외 목록만 대조용(정의 재계산은 아래)
# r_int_k 독립: 소멸 사건 기준 양성(E∈(L, L+3y], 전체 스냅샷)을 SQL로
lapq = pd.read_sql("""select r.rgst_no, substr(r.rgstCsDate,1,8) cd from right r where substr(r.rgst_no,1,2)='10' and r.rgstCsName='소멸등록' and r.rgstCsReason in ('등록료불납','연차료불납')""", src)
payall = pd.read_sql("select rgst_no, cast(lastAnnl as integer) l, substr(payDate,1,8) pd from pay where substr(rgst_no,1,2)='10'", src)
mm = lapq.merge(payall, on="rgst_no"); mm = mm[mm.pd < mm.cd]
lm = mm.groupby(["rgst_no", "cd"]).l.max().reset_index()
rdm = pd.Series(pd.to_datetime(pt.rgstDate.str[:8], format="%Y%m%d").values, index=pt.rgst_no.values)
lm["E"] = [pd.Timestamp(rdm[r]) + pd.DateOffset(years=int(l)) for r, l in zip(lm.rgst_no, lm.l)]
inw = set(lm[(lm.E > "2022-01-01") & (lm.E <= "2025-01-01")].rgst_no)
ylap = pd.Series(rg).isin(inw).values.astype(int)
kk = pd.DataFrame({"k": evp.ksic.astype(str).values, "y": y, "yl": ylap}).groupby("k").sum()
kk["r"] = kk.yl / kk.y
excl = sorted(kk[(kk.r < 0.97) | (kk.r > 1.03)].index)
R["V4_S2_excluded_n"] = len(excl); R["V4_S2_excluded_same_as_s3a"] = excl == sorted(ks[(ks.r_int < 0.97) | (ks.r_int > 1.03)].ksic.astype(str))
keep = ~evp.ksic.astype(str).isin(excl).values
R["V4_S2"] = sk(wauc(y[keep], pm[keep]), wauc(y[keep], pb[keep]))
R["V4_S3"] = sk(wauc(yv3, pm), wauc(yv3, pb))
fr = evp.atype.isin(["fcorp", "findiv"]).values; of = 270669 / 1153320
w4 = np.where(fr, of / fr.mean(), (1 - of) / (1 - fr.mean()))
R["V4_S4"] = sk(wauc(y, pm, w4), wauc(y, pb, w4))
# ---------------- R4 S2 무작위 KSIC 귀무(같은 KSIC 수, 200회, 점 추정; 보고용)
from sklearn.metrics import roc_auc_score
kl = np.array(sorted(evp.ksic.astype(str).unique())); nkeep = len(kl) - len(excl)
rng = np.random.default_rng(7); d0 = []
s0 = sk(roc_auc_score(y, pm), roc_auc_score(y, pb))
kv = evp.ksic.astype(str).values
for i in range(200):
    sel = rng.choice(kl, nkeep, replace=False); m = np.isin(kv, sel)
    d0.append(sk(roc_auc_score(y[m], pm[m]), roc_auc_score(y[m], pb[m])) - s0)
d0 = np.array(d0); dS2 = R["V4_S2"] - s0
R["R4_null"] = dict(n_keep_ksic=int(nkeep), delta_S2=float(dS2), null_mean=float(d0.mean()), null_sd=float(d0.std()), null_q025=float(np.quantile(d0, 0.025)),
                    null_q975=float(np.quantile(d0, 0.975)), frac_null_le_dS2=float((d0 <= dS2).mean()))
# ---------------- R5 4.2: 소멸등록일 기준 지평 h(5·7·10)년, 정부R&D 기원 대 비기원
fl = pd.read_parquet(os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext\ntis_patent_flag.parquet"), columns=["rgst_no", "gov_rnd"]).dropna(subset=["rgst_no"])
gset = set(fl[fl.gov_rnd == True].rgst_no.astype(str))
co = pd.read_sql("select rgst_no, substr(rgstDate,1,8) rd from patent where substr(rgst_no,1,2)='10' and substr(rgstDate,1,4) between '2007' and '2016'", src)
co["cd"] = co.rgst_no.map(lapq.groupby("rgst_no").cd.min())
co["rd"] = pd.to_datetime(co.rd, format="%Y%m%d"); co["cd"] = pd.to_datetime(co.cd, format="%Y%m%d", errors="coerce"); co["g"] = co.rgst_no.isin(gset)
r5 = {}
for h in (5, 7, 10):
    end = co.rd + pd.DateOffset(years=h); ob = end <= pd.Timestamp("2025-10-31")
    hit = co.cd.notna() & (co.cd <= end)
    r5[h] = dict(gov=float(hit[ob & co.g].mean()), nongov=float(hit[ob & ~co.g].mean()), n_gov=int((ob & co.g).sum()))
R["R5_regdate_horizon"] = r5
R["secs"] = round(time.time() - t0, 1)
json.dump(R, open(os.path.join(V, "v0_results.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
print(json.dumps(R, ensure_ascii=False, indent=1, default=str))
