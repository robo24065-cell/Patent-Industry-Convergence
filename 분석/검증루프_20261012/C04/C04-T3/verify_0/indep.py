# C04-T3 verify_0: 독립 구현(반박 검증). s1/s2 코드·캐시를 쓰지 않는다.
# 원천: _shared/meta/*.parquet 직접(공개연도 규칙 직접 구현) + 동결 예측 parquet. AUC는 Mann–Whitney U를 직접 계산.
import os, sys, json, hashlib, html, unicodedata
import numpy as np, pandas as pd, duckdb
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
V = os.path.join(ROOT, r"검증루프_20261012\C04\C04-T3\verify_0")
META = os.path.join(ROOT, r"모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
PRED = os.path.join(ROOT, r"모델경쟁_20260927\T3_기술셀부상예측\R3\preds\FINAL_R3_t0_2021.parquet")
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2")
# 공개연도 = min(pub_date, rgst_date) 의 연도(빈 문자열 무시), 메인그룹 = 'A01B 1/00' → 'A01B 1' (main_ipc 없으면 ipc_all 첫 항목)
q = f"""
with r as (
  select cc, ksic, appl_no, applicant,
     regexp_replace(trim(coalesce(nullif(main_ipc,''), split_part(ipc_all,'|',1))), '\\s+', '', 'g') ip,
     least(nullif(pub_date,''), nullif(rgst_date,'')) d1, nullif(pub_date,'') d2, nullif(rgst_date,'') d3
  from read_parquet('{META}') where appl_year >= 2000),
s as (select cc, ksic, appl_no, applicant, ip, try_cast(left(coalesce(d1,d2,d3),4) as int) yr from r),
t as (select cc, ksic, appl_no, applicant, yr,
     case when regexp_matches(ip, '^[A-H][0-9]{{2}}[A-Z][0-9]{{1,4}}/') then substr(ip,1,4)||' '||cast(cast(regexp_extract(ip,'^[A-H][0-9]{{2}}[A-Z]([0-9]{{1,4}})/',1) as int) as varchar)
          when regexp_matches(ip, '^[A-H][0-9]{{2}}[A-Z][0-9]{{3,6}}$') then substr(ip,1,4)||' '||cast(cast(substr(ip,5,length(ip)-6) as int) as varchar) end mg
  from s where yr between 2016 and 2024)
select * from t where mg is not null
"""
con.execute(f"create temp table T as {q}")
cellw = con.sql("""select ksic||'|'||mg cell,
   count(distinct case when yr between 2019 and 2021 then cc||appl_no end) tot_W,
   count(distinct case when cc='KR' and yr between 2019 and 2021 then appl_no end) kr_W,
   count(distinct case when cc='KR' and yr between 2022 and 2024 then appl_no end) kr_fut
   from T group by 1""").df()
el = cellw[cellw.tot_W >= 50].copy()
el['y'] = np.log1p(el.kr_fut) - np.log1p(el.kr_W); el['inc'] = el.kr_fut - el.kr_W
el['h'] = [int(hashlib.md5(c.encode()).hexdigest()[:8], 16) for c in el.cell]
el = el.sort_values(['y', 'inc', 'h'], ascending=[False, False, True]).reset_index(drop=True)
el['yb'] = (np.arange(len(el)) < round(0.1 * len(el))).astype(int)
P = pd.read_parquet(PRED)[['cell', 'p', 'p_base', 'yb']].rename(columns={'yb': 'yb_frozen'})
M = el.merge(P, on='cell', how='inner')
res = dict(n_elig=int(len(el)), n_merged=int(len(M)), yb_mismatch=int((M.yb != M.yb_frozen).sum()))

def mwu_auc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    pos, neg = s[y], s[~y]
    # 순위 평균(동점 0.5) — 정렬 기반
    allv = np.concatenate([pos, neg]); order = allv.argsort(kind='mergesort'); ranks = np.empty(len(allv)); sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]: j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    U = ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return U / (len(pos) * len(neg))
def sk(d):
    a, b = mwu_auc(d.yb, d.p), mwu_auc(d.yb, d.p_base); return dict(n=int(len(d)), pos=int(d.yb.sum()), auc=a, auc_base=b, skill=(a - b) / (1 - b))
res['ALL'] = sk(M)
for nm, f in [('S0', M.kr_W == 0), ('S1', M.kr_W == 1), ('S2', M.kr_W >= 2), ('>=5', M.kr_W >= 5)]: res[nm] = sk(M[f])

# 단일 출원인 부상(독립 정규화: NFKC → 엔티티 해제 → 한글·영숫자만 남김 → '주식회사' 제거)
def norm(s):
    s = unicodedata.normalize('NFKC', html.unescape(s or '')).lower().replace('주식회사', '').replace('(주)', '').replace('㈜', '')
    return ''.join(ch for ch in s if ch.isalnum())
kr = con.sql("""select distinct ksic||'|'||mg cell, appl_no, applicant, yr from T where cc='KR' and yr between 2019 and 2024""").df()
kr = kr[kr.cell.isin(set(M.cell))]
kr['a'] = kr.applicant.map(norm); kr['sgn'] = np.where(kr.yr <= 2021, -1, 1)
da = kr.groupby(['cell', 'a']).sgn.sum().groupby('cell').max()
M['maxd'] = M.cell.map(da).fillna(0)
M['sab'] = (M.yb == 1) & (M.inc > 0) & (M.maxd >= 0.5 * M.inc)
res['sab_n'] = int(M.sab.sum()); res['sab_rate'] = float(M.sab.sum() / M.yb.sum())
res['sab_rate_by_stratum'] = {nm: float(M.sab[f & (M.yb == 1)].sum() / (f & (M.yb == 1)).sum()) for nm, f in [('S0', M.kr_W == 0), ('S1', M.kr_W == 1), ('S2', M.kr_W >= 2)]}
res['altA'] = sk(M[~M.sab]); res['altA']['d_skill'] = res['altA']['skill'] - res['ALL']['skill']
o = M.p.rank(ascending=False, method='first') <= 504
res['top504_hits'] = int((o & (M.yb == 1)).sum()); res['top504_hits_krW_le1'] = int((o & (M.yb == 1) & (M.kr_W <= 1)).sum())
res['top504_hits_inc_le3'] = int((o & (M.yb == 1) & (M.inc <= 3)).sum())

# 레이블 정의 점검(독립): 절대 증가(kr_fut−kr_W) 상위 10%(동률 y → md5) 레이블에서 동결 예측 AUC
Mb = M.sort_values(['inc', 'y', 'h'], ascending=[False, False, True]).reset_index(drop=True)
Mb['yb'] = (np.arange(len(Mb)) < round(0.1 * len(Mb))).astype(int)
res['abs_label'] = sk(Mb); res['abs_label']['overlap_with_official'] = int(((Mb.yb == 1) & (Mb.yb_frozen == 1)).sum())
res['abs_label']['krW_le1_in_label'] = int(((Mb.yb == 1) & (Mb.kr_W <= 1)).sum())
# 비교
S2R = pd.read_csv(os.path.join(V, '..', 'strata_scores.csv')).set_index('stratum')
cmp = {}
for nm in ['ALL', 'S0', 'S1', 'S2', '>=5']:
    cmp[nm] = dict(d_auc=abs(res[nm]['auc'] - S2R.loc[nm, 'auc']), d_auc_base=abs(res[nm]['auc_base'] - S2R.loc[nm, 'auc_base']), d_n=int(res[nm]['n'] - S2R.loc[nm, 'n']))
s2 = json.load(open(os.path.join(V, '..', 's2_results.json'), encoding='utf-8'))
alt = pd.read_csv(os.path.join(V, '..', 'rescore_alt_labels.csv'))
a1 = alt[(alt.set == 'final') & (alt.key == 'k1') & (alt.alt == 'A_exclude_SAB')].iloc[0]
cmp['sab_n_diff'] = res['sab_n'] - s2['final']['sab_n']['k1']; cmp['altA_dskill_diff'] = abs(res['altA']['d_skill'] - a1.d_skill)
res['compare_to_main'] = cmp
res['all_ok'] = bool(res['yb_mismatch'] == 0 and all(cmp[k]['d_auc'] < 1e-9 and cmp[k]['d_auc_base'] < 1e-6 and cmp[k]['d_n'] == 0 for k in ['ALL', 'S0', 'S1', 'S2', '>=5'])
                     and cmp['sab_n_diff'] == 0 and cmp['altA_dskill_diff'] < 1e-9)
json.dump(res, open(os.path.join(V, 'indep_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps(res, ensure_ascii=False, indent=1, default=float))
