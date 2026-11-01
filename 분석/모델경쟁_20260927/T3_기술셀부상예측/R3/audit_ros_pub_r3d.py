"""T3 3라운드 최종시험 전 감사(D단계) — RoS 셀 배정 시점 점검.
동결 모형의 RoS 2피처(ros_sci, ros_ppp)는 'US 발행 ≤t0' 그리고 '셀 배정 근거 레코드의 최초 출원연도 ≤t0-2'(18개월 공개 지연 보수 규칙)로 자른다.
여기서는 배정 근거 레코드가 실제로 t0년 말까지 공개됐는지(공개연도 = least(pub_date, rgst_date) ≤ t0)를 제공 메타에서 직접 확인한다.
  (1) 현재 규칙으로 쓰이는 (셀, US 특허) 연결 중 배정 레코드가 모두 t0 이후 공개인 비율
  (2) 엄격 규칙(배정 레코드 공개연도 ≤t0 & 발행 ≤t0)으로 다시 만든 ros_sci·ros_ppp 와 동결 피처의 차이(적격 셀)
출력: audit_ros_pub_r3d.json (원점 2010–2018 개발 구간 + 2021 최종 원점의 피처만. 레이블은 쓰지 않는다)
"""
import os, sys, json, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import numpy as np, pandas as pd, duckdb
X = os.path.join(HERE, '..', '..', '_shared', 'ext').replace(os.sep, '/')
META = os.path.join(HERE, '..', '..', '_shared', 'meta').replace(os.sep, '/') + '/*.parquet'
t = time.time()
con = duckdb.connect(); con.execute("set memory_limit='3GB'; set threads=2; set preserve_insertion_order=false; set enable_progress_bar=false")
con.execute(f"""create table pubrec as select cc, appl_no, ksic,
   min(left(coalesce(least(nullif(pub_date,''), nullif(rgst_date,'')), nullif(pub_date,''), nullif(rgst_date,'')),4))::int pub_year
   from read_parquet('{META}') group by 1,2,3""")
con.execute(f"""create table cp as select r.ksic, r.mg, f.us_pn, min(r.appl_year) first_rec_year, min(p.pub_year) first_pub_year
   from read_parquet('{X}/r3ros_work/fam_us.parquet') f join read_parquet('{X}/r3ros_work/rec.parquet') r using(cc, appl_no, ksic)
   left join pubrec p using(cc, appl_no, ksic) where r.mg is not null group by 1,2,3""")
con.execute(f"""create table ci as select c.ksic, c.mg, s.issue_year, c.first_rec_year, coalesce(c.first_pub_year, 9999) first_pub_year,
   count(*) n_us_pat, sum((s.n_cit>0)::int)::double n_us_pat_sci, sum(s.n_cit)::double n_paper_cit
   from cp c join read_parquet('{X}/r3ros_patent_sci.parquet') s using(us_pn) where s.issue_year is not null group by all""")
print('built', round(time.time() - t), flush=True)
# 원 규칙 재현 점검: first_rec_year 기준 집계가 공유 파일과 같은지
chk = con.sql(f"""select sum(n_us_pat) from ci""").fetchone()[0], con.sql(f"select sum(n_us_pat) from read_parquet('{X}/r3ros_cell_issueyear.parquet')").fetchone()[0]
ci = con.sql("select * from ci").df(); ci['cell'] = ci.ksic + '|' + ci.mg
out = dict(recon_total_links=dict(recomputed=int(chk[0]), shared_file=int(chk[1])))
T0S = list(range(2010, 2019)) + [2021]
viol = {}
for t0 in T0S:
    cur = ci[(ci.first_rec_year <= t0 - 2) & (ci.issue_year <= t0) & (ci.issue_year >= t0 - 9)]
    viol[t0] = dict(links=int(cur.n_us_pat.sum()), links_mapping_not_public_by_t0=int(cur[cur.first_pub_year > t0].n_us_pat.sum()),
                    share=float(cur[cur.first_pub_year > t0].n_us_pat.sum() / max(cur.n_us_pat.sum(), 1)))
    print(t0, viol[t0], flush=True)
out['current_rule_links'] = viol

def ros(ci, t0s, cells, ksic_c, rule, k=5.0):
    rows = []
    for t0 in t0s:
        base = ci[(ci.first_rec_year <= t0 - 2) & (ci.issue_year <= t0)] if rule == 'rec' else ci[(ci.first_pub_year <= t0) & (ci.issue_year <= t0)]
        w10 = base[base.issue_year >= t0 - 9]
        A = w10.groupby('cell')[['n_us_pat', 'n_us_pat_sci', 'n_paper_cit']].sum().reindex(cells).fillna(0.0)
        ks = w10.groupby('ksic')[['n_us_pat', 'n_us_pat_sci', 'n_paper_cit']].sum()
        ks_sh = (ks.n_us_pat_sci / ks.n_us_pat.clip(lower=1)).reindex(ksic_c).fillna(0.2).values
        ks_pp = (ks.n_paper_cit / ks.n_us_pat.clip(lower=1)).reindex(ksic_c).fillna(1.0).values
        n = A.n_us_pat.values
        rows.append(pd.DataFrame(dict(cell=cells, t0=t0, ros_sci=(A.n_us_pat_sci.values + k * ks_sh) / (n + k),
                                      ros_ppp=np.log1p((A.n_paper_cit.values + k * ks_pp) / (n + k)))))
    return pd.concat(rows, ignore_index=True)

# 적격 셀(5개국 공개연도 t0-2..t0 ≥50): 개발 캐시 feat 파일(2010–2018) + 동결 피처 파일(최종 _final 은 t0=2021 피처만 사용)
fz = pd.read_parquet(os.path.join(HERE, 'extB_cell_pub.parquet'))[['cell', 't0', 'ros_sci', 'ros_ppp']]
ff = os.path.join(HERE, 'extB_cell_pub_final.parquet')
if os.path.exists(ff): fz = pd.concat([fz, pd.read_parquet(ff, columns=['cell', 't0', 'ros_sci', 'ros_ppp']).query('t0 == 2021')])
el = []
for t0 in range(2010, 2019):
    f = pd.read_parquet(os.path.join(HERE, 'cache_pub', f'feat_{t0}.parquet'), columns=['cell', 'ksic', 't0', 'tot_W']); el.append(f[f.tot_W >= 50])
el = pd.concat(el)
cells = np.array(sorted(set(fz.cell))); ksic_c = np.array([c.split('|')[0] for c in cells])
res = {}
for rule in ['rec', 'pub']:
    res[rule] = ros(ci, T0S, cells, ksic_c, rule)
m = fz.merge(res['rec'], on=['cell', 't0'], suffixes=('', '_rec')).merge(res['pub'], on=['cell', 't0'], suffixes=('', '_pub'))
out['recon_vs_frozen_maxdiff'] = float(max(np.abs(m.ros_sci - m.ros_sci_rec).max(), np.abs(m.ros_ppp - m.ros_ppp_rec).max()))
cmp_ = {}
for t0 in T0S:
    g = m[m.t0 == t0]
    if t0 <= 2018: g = g.merge(el[el.t0 == t0][['cell']], on='cell')
    cmp_[t0] = dict(n=int(len(g)),
                    sci_mad=float(np.abs(g.ros_sci - g.ros_sci_pub).mean()), sci_maxdiff=float(np.abs(g.ros_sci - g.ros_sci_pub).max()),
                    sci_spearman=float(g.ros_sci.rank().corr(g.ros_sci_pub.rank())),
                    ppp_mad=float(np.abs(g.ros_ppp - g.ros_ppp_pub).mean()), ppp_spearman=float(g.ros_ppp.rank().corr(g.ros_ppp_pub.rank())))
    print(t0, cmp_[t0], flush=True)
out['frozen_vs_strict_public_rule'] = cmp_
out['note'] = 't0=2021 행은 extB_cell_pub_final.parquet(있을 때)의 피처만 비교. 2021 적격 필터는 적용하지 않음(최종 캐시 레이블 미접촉)'
json.dump(out, open(os.path.join(HERE, 'audit_ros_pub_r3d.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('secs', round(time.time() - t))
