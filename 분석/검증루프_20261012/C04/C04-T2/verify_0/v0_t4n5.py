# -*- coding: utf-8 -*-
"""C04-T2 반박 검증(verify_0): 본 작업과 다른 구현으로 T4·N5 양성 수와 AUC 재계산.
T4: sqlite3 순수 SQL(날짜 함수)로 위험집합·레이블 → 순위합(Mann–Whitney) AUC. N5: 기존 연결 파일 trade_etsi_sep_link.parquet × kr_patents를 duckdb SQL로 결합 → 순위합 AUC.
출력: verify_0/v0_out.json"""
import os, json, time, sqlite3
import numpy as np, pandas as pd, duckdb
from scipy.stats import rankdata
t0 = time.time()
V = os.path.dirname(os.path.abspath(__file__)); W = os.path.dirname(V)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")


def mw_auc(y, s):
    y = np.asarray(y).astype(bool); r = rankdata(s)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ---------------- T4 (SQL)
con = sqlite3.connect(f"file:{os.path.join(BASE, '수집파이프라인', 'registry_full.sqlite')}?mode=ro", uri=True)
ISO = "substr({c},1,4)||'-'||substr({c},5,2)||'-'||substr({c},7,2)"
ADDY = "(case when strftime('%m-%d',{d})='02-29' then date({d},'-1 day','+'||{n}||' years') else date({d},'+'||{n}||' years') end)"
q = f"""
with p as (
  select rowid rid, rgst_no, ksic, {ISO.format(c='rgstDate')} rd, {ISO.format(c='applDate')} ad,
         case when length(cndrtExptnDate)=8 and date({ISO.format(c='cndrtExptnDate')})={ISO.format(c='cndrtExptnDate')} then {ISO.format(c='cndrtExptnDate')} end ex0
  from patent where substr(rgst_no,1,2)='10' and length(rgstDate)=8 and date({ISO.format(c='rgstDate')})={ISO.format(c='rgstDate')}),
pp as (select *, coalesce(ex0, {ADDY.format(d='ad', n='20')}) ex, row_number() over (order by rid)-1 pid from p),
py as (select rgst_no, cast(lastAnnl as integer) l, {ISO.format(c='payDate')} pdt from pay
       where trim(lastAnnl) glob '[0-9]*' and length(payDate)=8 and date({ISO.format(c='payDate')})={ISO.format(c='payDate')}),
lL as (select rgst_no, max(l) l from py where pdt<='2022-01-01' group by rgst_no),
lE as (select rgst_no, max(l) l from py where pdt<='2025-08-22' group by rgst_no),
ml0 as (select rgst_no, {ISO.format(c='rgstCsDate')} cd from right where rgstCsName in ('권리의 말소등록','말소등록','직권말소등록')),
ml as (select rgst_no, max(cd<='2022-01-01') pre, max(cd>'2022-01-01' and cd<='2025-01-01') win from ml0 group by rgst_no),
b as (
  select pp.pid, pp.rgst_no, pp.rd, pp.ex,
    case when lL.l>0 then {ADDY.format(d='pp.rd', n='lL.l')} end pthL,
    case when lE.l>0 then {ADDY.format(d='pp.rd', n='lE.l')} end pthE,
    coalesce(ml.pre,0) mal_pre, coalesce(ml.win,0) mal_win
  from pp left join lL on lL.rgst_no=pp.rgst_no left join lE on lE.rgst_no=pp.rgst_no left join ml on ml.rgst_no=pp.rgst_no)
select pid, rgst_no,
  case when (pthE>'2025-01-01') or (ex<='2025-01-01' and pthE>=ex) or mal_win then 0
       when pthE>'2022-01-01' and pthE<='2025-01-01' and pthE<ex and date(pthE,'+213 days')<='2025-08-22' then 1 end y
from b where rd<'2022-01-01' and pthL>'2022-01-01' and ex>'2022-01-01' and not mal_pre
"""
T = pd.read_sql(q, con); con.close()
FD = os.path.join(MC, r"T4_특허포기예측\R2\runs_final\F22")
ev = pd.read_parquet(os.path.join(FD, "ev.parquet"), columns=["pid", "y"])
m = ev.merge(T, on="pid", how="outer", indicator=True)
ws = json.load(open(os.path.join(MC, r"T4_특허포기예측\R2\final_spec_r2.json"), encoding="utf-8"))["spec"]["ensemble"]["weights"]
P = {k: np.load(os.path.join(FD, "pred", f)) for k, f in [("cls", "cls_99f270b3cd.npy"), ("haz", "haz_0568d4297b.npy"), ("logit", "logit_3008b8b195.npy"), ("hgb", "hgb_b4bd0fa46a.npy"), ("base", "base_46526c56d8.npy")]}
lo = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6)) - np.log(1 - np.clip(p, 1e-6, 1 - 1e-6))
z = sum(w * lo(P[k]) for k, w in ws.items()) / sum(ws.values()); pm = 1 / (1 + np.exp(-z))
yv = ev.merge(T, on="pid", how="left").y_y.values
ps = np.load(os.path.join(MC, r"T4_특허포기예측\VERIFY_R5\cache\strong_full.npy"))
ok = ~np.isnan(yv)
a_m, a_b, a_s = mw_auc(yv[ok], pm[ok]), mw_auc(yv[ok], P["base"][ok]), mw_auc(yv[ok], ps[ok])
R = dict(T4=dict(n_sql=int(len(T)), pos_sql=int(T.y.sum()), undetermined_sql=int(T.y.isna().sum()), only_frozen=int((m._merge == "left_only").sum()), only_sql=int((m._merge == "right_only").sum()),
                 label_mismatch=int((m[m._merge == "both"].y_x != m[m._merge == "both"].y_y).sum()),
                 auc_model_mw=a_m, auc_base_mw=a_b, skill=(a_m - a_b) / (1 - a_b), auc_strong_mw=a_s, skill_vs_strong=(a_m - a_s) / (1 - a_s), secs=round(time.time() - t0, 1)))
print(R, flush=True)
# ---------------- N5 (duckdb, 기존 연결 파일 경로)
L = os.path.join(MC, r"_shared\ext\trade_etsi_sep_link.parquet").replace("\\", "/")
K = os.path.join(BASE, r"공모전데이터\작업데이터\kr_patents.parquet").replace("\\", "/")
d = duckdb.connect(); d.sql("SET threads=2; SET memory_limit='2GB'")
U = d.sql(f"""
with lk as (select cast(appl_no as varchar) a, min(cast(strftime(first_refl, '%Y%m%d') as integer)) refl from read_parquet('{L}') where cc='KR' group by 1),
k as (select cast("출원번호" as varchar) a, "출원연도" ay,
        least(try_cast(nullif(regexp_replace(cast("공개일자" as varchar), '[^0-9]', '', 'g'), '') as integer),
              try_cast(nullif(regexp_replace(cast("등록일자" as varchar), '[^0-9]', '', 'g'), '') as integer)) known
      from read_parquet('{K}') where cast("세세분류코드" as varchar)='26429')
select k.a appl_no, case when lk.refl>20221231 and lk.refl<=20241231 then 1 else 0 end y
from k left join lk on lk.a=k.a
where k.known<=20221231 and k.ay between 2017 and 2022 and not coalesce(lk.refl<=20221231, false)""").df()
n26429_link = d.sql(f"select count(distinct cast(appl_no as varchar)) from read_parquet('{L}') where cc='KR' and ksic='26429'").fetchone()[0]
F = pd.read_parquet(os.path.join(MC, r"N5_표준특허선언레이더\final_preds_r4.parquet")); F["appl_no"] = F.appl_no.astype(str)
mm = F.merge(U, on="appl_no", how="outer", indicator=True)
yy = F.merge(U, on="appl_no", how="left").y_y.values
S = pd.read_parquet(os.path.join(MC, r"N5_표준특허선언레이더\VERIFY_R5\cache\sens_preds.parquet"))
c, b, s = mw_auc(yy, F.p_champ.values), mw_auc(yy, F.p_M0.values), mw_auc(yy, S.p_M0plus.values)
R["N5"] = dict(n_sql=int(len(U)), pos_sql=int(U.y.sum()), only_frozen=int((mm._merge == "left_only").sum()), only_sql=int((mm._merge == "right_only").sum()),
               label_mismatch=int((mm[mm._merge == "both"].y_x != mm[mm._merge == "both"].y_y).sum()), link_file_kr26429_appl=int(n26429_link),
               auc_champ_mw=c, auc_M0_mw=b, skill=(c - b) / (1 - b), auc_strong_mw=s, skill_vs_strong=(c - s) / (1 - s))
# 본 작업 결과와 대조
t4 = json.load(open(os.path.join(W, "cache", "t4_s2_out.json"), encoding="utf-8")); n5 = json.load(open(os.path.join(W, "cache", "n5_s3_out.json"), encoding="utf-8"))
R["compare_main"] = dict(T4_pos_equal=R["T4"]["pos_sql"] == t4["pos"], T4_auc_model_diff=R["T4"]["auc_model_mw"] - t4["auc_model"], T4_auc_base_diff=R["T4"]["auc_base_mw"] - t4["auc_base"],
                         T4_auc_strong_diff=R["T4"]["auc_strong_mw"] - t4["strong_frozen_file"]["auc_strong"],
                         N5_pos_equal=R["N5"]["pos_sql"] == n5["pos"], N5_auc_champ_diff=R["N5"]["auc_champ_mw"] - n5["auc"]["champ"], N5_auc_M0_diff=R["N5"]["auc_M0_mw"] - n5["auc"]["M0"],
                         N5_auc_strong_diff=R["N5"]["auc_strong_mw"] - n5["strong_file"]["auc"])
R["compare_main"]["verdict"] = "반박 실패(확인)" if (R["compare_main"]["T4_pos_equal"] and R["compare_main"]["N5_pos_equal"] and
                                                 max(abs(v) for k, v in R["compare_main"].items() if k.endswith("_diff")) <= 1e-6) else "불일치 — 점검 필요"
R["secs"] = round(time.time() - t0, 1)
json.dump(R, open(os.path.join(V, "v0_out.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print(json.dumps(R, ensure_ascii=False, indent=1, default=float))
