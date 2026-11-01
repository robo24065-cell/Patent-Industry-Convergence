# C02-T2 s6: 공식 통계(KOSIS/KIPO, WIPO, CNIPA) vs 제공 KR·CN 연도별 건수 → official_vs_provided.csv
# 제공 건수는 meta에서 직접 다시 센다(s1 캐시를 그대로 믿지 않음). 읽기 전용.
import os, sys, json, re, html
import duckdb, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
W = os.path.join(ROOT, r"분석\검증루프_20261012\C02\C02-T2")
META = os.path.join(ROOT, r"분석\모델경쟁_20260927\_shared\meta\*.parquet").replace("\\", "/")
BR = os.path.join(ROOT, r"수집파이프라인\business_rights.sqlite").replace("\\", "/")
DL = os.path.join(W, 'download'); OUT = W; C = os.path.join(W, 'cache')

# ---- 공식: KOSIS(KIPO)
k = pd.read_csv(os.path.join(C, 's5_kosis_long.csv'), dtype=str); k['DT'] = pd.to_numeric(k.DT, errors='coerce')
def piv(src, item):
    s = k[(k.src == src) & (k.C1_NM == item)]; return s.set_index(s.PRD_DE.astype(int)).DT
kipo_pat_reg = piv('data_TX_13801_A006', '특허'); kipo_um_reg = piv('data_TX_13801_A006', '실용신안')
kipo_pat_app = piv('data_TX_13801_A004', '특허')
a46 = k[(k.src == 'data_TX_13801_A046') & (k.C2_NM == '특허')]
kipo_dom = a46[a46.C1_NM == '내국인'].set_index(a46[a46.C1_NM == '내국인'].PRD_DE.astype(int)).DT
kipo_for = a46[a46.C1_NM == '외국인'].set_index(a46[a46.C1_NM == '외국인'].PRD_DE.astype(int)).DT
# ---- 공식: WIPO 지표(관청 기준 합계, 2000–2022)
wg = pd.read_csv(os.path.join(DL, r'wipo\pi\dc_indicator_patent_5_grant_by_technology.csv'), keep_default_na=False)
wp = pd.read_csv(os.path.join(DL, r'wipo\pi\dc_indicator_patent_4_publication_by_technology.csv'), keep_default_na=False)
wipo_grant = {o: wg[wg.office == o].groupby('year')['count'].sum() for o in ['KR', 'CN']}
wipo_pub = {o: wp[wp.office == o].groupby('year')['count'].sum() for o in ['KR', 'CN']}
# WIPO 국가 프로필(2024년 자료): KR 등록 127,806 ↓5.1%, CN 발명 등록 1,044,777 ↑13.5% (profile_kr.pdf, profile_cn.pdf 본문)
wipo_profile = {'KR': {2024: 127806}, 'CN': {2024: 1044777}}
# ---- 공식: CNIPA 연보 2-1 등록(授权), 발명·실용신안 (2023·2024·2025 연보에서 같은 해 값이 같은지 확인)
def cnipa(fn):
    b = open(fn, 'rb').read()
    for enc in ['utf-8', 'gb18030']:
        try: t = b.decode(enc); break
        except UnicodeDecodeError: pass
    t = re.sub(r'<style.*?</style>', '', t, flags=re.S); rows = []
    for r in re.findall(r'<tr.*?</tr>', t, flags=re.S):
        c = [html.unescape(re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', '', x))).strip() for x in re.findall(r'<td.*?</td>', r, flags=re.S)]
        c = [x for x in c if x]; rows.append(c)
    out = {}; sec = None
    for c in rows:
        if c and c[0].startswith('合计'): sec = 'total'; c = c[1:]
        elif c and c[0].startswith('国内'): sec = 'dom'; c = c[1:]
        elif c and c[0].startswith('国外'): sec = 'for'; c = c[1:]
        if sec and len(c) >= 3 and re.fullmatch(r'\d{4}', c[0]):
            out[(sec, int(c[0]))] = (int(c[1]), int(c[2]))
    return out
cn = {}
vint = {}
for fn in [r'cnipa\year2023_b_b1.html', r'cnipa_r2\year2024_b_b1.html', r'cnipa\year2025_b_b1.html']:
    d = cnipa(os.path.join(DL, fn)); vint[fn] = d
    for kk, v in d.items():
        if kk in cn and cn[kk] != v: print('!! CNIPA 연보 간 수정', kk, cn[kk], v)
        cn[kk] = v
cn_inv = pd.Series({y: v[0] for (s, y), v in cn.items() if s == 'total'}).sort_index()
cn_um = pd.Series({y: v[1] for (s, y), v in cn.items() if s == 'total'}).sort_index()

# ---- 제공: meta 직접 집계
con = duckdb.connect(); con.sql("set threads=2"); con.sql("set memory_limit='3GB'"); con.sql("install sqlite; load sqlite")
kr = con.sql(f"""select left(rgst_date,4)::int y, substr(rgst_no,1,2) p, count(distinct rgst_no) n from read_parquet('{META}')
                 where cc='KR' and rgst_no is not null and rgst_no<>'' group by 1,2""").df()
krp = kr[kr.p == '10'].set_index('y').n; kru = kr[kr.p == '20'].set_index('y').n
# 제공 KR 출원 단위로 rgst 공란인데 법인 포트폴리오(2026 스냅샷)에서 등록 확인된 건 (등록연도별) + br 커버율
con.sql(f"""create temp table kr as select appl_no, any_value(rgst_no) rgst_no, any_value(rgst_date) rgst_date from read_parquet('{META}') where cc='KR' group by appl_no""")
con.sql(f"""create temp table br as select appl_no, min(rgst_date) rgst_date from sqlite_scan('{BR}', 'right_list') where substr(rgst_no,1,2)='10' group by appl_no""")
miss = con.sql("""select left(b.rgst_date,4)::int y, count(*) n from kr k join br b using(appl_no)
                  where (k.rgst_no is null or k.rgst_no='') and substr(k.appl_no,1,2)='10' and b.rgst_date<='20250822' group by 1""").df().set_index('y').n
cover = con.sql("""select left(k.rgst_date,4)::int y, avg((b.appl_no is not null)::int) c from kr k left join br b using(appl_no)
                   where substr(k.rgst_no,1,2)='10' group by 1""").df().set_index('y').c
# CN: 발명(5번째 자리 1·8)·실용신안(2·9), 등록연도·공개연도·T3 공개연도(least) 별 고유 출원
cnq = con.sql(f"""select appl_no, any_value(case when length(appl_no) in (12,14) and substr(appl_no,5,1) in ('1','8') then 'inv'
                         when length(appl_no) in (12,14) and substr(appl_no,5,1) in ('2','9') then 'um' else 'other' end) dt,
                   any_value(left(nullif(rgst_date,''),4)) ry, any_value(left(nullif(pub_date,''),4)) py,
                   any_value(left(coalesce(least(nullif(pub_date,''),nullif(rgst_date,'')),nullif(pub_date,''),nullif(rgst_date,'')),4)) ty,
                   bool_and(coalesce(nullif(main_ipc,''), nullif(split_part(ipc_all,'|',1),'')) is null) miss_all
                  from read_parquet('{META}') where cc='CN' group by appl_no""").df()
con.close()
rows = []
for y in range(2012, 2026):
    r = dict(country='KR', year=y, prov_pat_reg=int(krp.get(y, 0)), prov_um_reg=int(kru.get(y, 0)),
             kipo_pat_reg=kipo_pat_reg.get(y, np.nan), kipo_um_reg=kipo_um_reg.get(y, np.nan),
             wipo_pat_grant=wipo_grant['KR'].get(y, wipo_profile['KR'].get(y, np.nan)),
             kipo_dom_reg=kipo_dom.get(y, np.nan), kipo_for_reg=kipo_for.get(y, np.nan),
             prov_missing_reg_br=int(miss.get(y, 0)), br_cover=cover.get(y, np.nan))
    rows.append(r)
K = pd.DataFrame(rows)
K['kipo_vs_wipo_diff'] = K.wipo_pat_grant / K.kipo_pat_reg - 1
K['ratio_pat'] = K.prov_pat_reg / K.kipo_pat_reg
K['ratio_um'] = K.prov_um_reg / K.kipo_um_reg
K['prov_pat_reg_corrected'] = K.prov_pat_reg + K.prov_missing_reg_br / K.br_cover
K['ratio_pat_corrected'] = K.prov_pat_reg_corrected / K.kipo_pat_reg
rbar = K[K.year.between(2019, 2023)].ratio_pat.mean()
K['d_vs_2019_23'] = K.ratio_pat / rbar - 1
K['d_corrected_vs_2019_23'] = K.ratio_pat_corrected / K[K.year.between(2019, 2023)].ratio_pat_corrected.mean() - 1
K['d_um_vs_2019_23'] = K.ratio_um / K[K.year.between(2019, 2023)].ratio_um.mean() - 1
# 2019–23 선형추세 외삽 대비(보조)
b = np.polyfit(K[K.year.between(2019, 2023)].year, K[K.year.between(2019, 2023)].ratio_pat, 1)
K['d_vs_trend'] = K.ratio_pat / np.polyval(b, K.year) - 1

g = cnq.groupby
def cnt(dt, col):
    s = cnq[cnq.dt == dt] if dt else cnq
    return s.groupby(col).size(), s[s.miss_all].groupby(col).size()
inv_r, inv_r_m = cnt('inv', 'ry'); um_r, um_r_m = cnt('um', 'ry'); inv_p, inv_p_m = cnt('inv', 'py'); all_t, all_t_m = cnt(None, 'ty')
crow = []
for y in range(2012, 2026):
    s = str(y)
    crow.append(dict(country='CN', year=y, prov_inv_by_rgst=int(inv_r.get(s, 0)), prov_um_by_rgst=int(um_r.get(s, 0)),
                     prov_inv_by_pub=int(inv_p.get(s, 0)), prov_all_by_t3year=int(all_t.get(s, 0)),
                     miss_inv_by_rgst=int(inv_r_m.get(s, 0)), miss_um_by_rgst=int(um_r_m.get(s, 0)), miss_inv_by_pub=int(inv_p_m.get(s, 0)),
                     miss_all_by_t3year=int(all_t_m.get(s, 0)),
                     cnipa_inv_grant=cn_inv.get(y, np.nan), cnipa_um_grant=cn_um.get(y, np.nan),
                     wipo_cn_grant=wipo_grant['CN'].get(y, wipo_profile['CN'].get(y, np.nan)), wipo_cn_pub=wipo_pub['CN'].get(y, np.nan)))
Cn = pd.DataFrame(crow)
Cn['cnipa_vs_wipo_grant_diff'] = Cn.wipo_cn_grant / Cn.cnipa_inv_grant - 1
Cn['ratio_inv_rgst'] = Cn.prov_inv_by_rgst / Cn.cnipa_inv_grant
Cn['ratio_um_rgst'] = Cn.prov_um_by_rgst / Cn.cnipa_um_grant
Cn['ratio_inv_pub'] = Cn.prov_inv_by_pub / Cn.wipo_cn_pub
Cn['official_t3year_proxy'] = Cn.wipo_cn_pub + Cn.cnipa_um_grant   # 발명 공개 + 실용신안 등록(=첫 공개)
Cn['ratio_t3year'] = Cn.prov_all_by_t3year / Cn.official_t3year_proxy
Cn['miss_share_t3year'] = Cn.miss_all_by_t3year / Cn.prov_all_by_t3year
ref = Cn[Cn.year.isin([2018, 2019, 2020, 2022])]
for c in ['ratio_inv_rgst', 'ratio_um_rgst', 'ratio_inv_pub', 'ratio_t3year']:
    Cn['d_' + c + '_2021ref'] = Cn[c] / ref[c].mean() - 1
out = pd.concat([K, Cn], ignore_index=True)
out.to_csv(os.path.join(OUT, 'official_vs_provided.csv'), index=False, encoding='utf-8-sig')
res = dict(rbar_2019_23=rbar, d_2024=float(K.loc[K.year == 2024, 'd_vs_2019_23'].iloc[0]),
           d_2024_corrected=float(K.loc[K.year == 2024, 'd_corrected_vs_2019_23'].iloc[0]),
           d_2024_trend=float(K.loc[K.year == 2024, 'd_vs_trend'].iloc[0]),
           d_um_2024=float(K.loc[K.year == 2024, 'd_um_vs_2019_23'].iloc[0]),
           kipo_vs_wipo_maxabs_2015_2024=float(K[K.year.between(2015, 2024)].kipo_vs_wipo_diff.abs().max()),
           cnipa_vs_wipo_maxabs_2015_2022=float(Cn[Cn.year.between(2015, 2022)].cnipa_vs_wipo_grant_diff.abs().max()),
           cn_2021=Cn[Cn.year == 2021].iloc[0].to_dict(),
           cnipa_vintages={k: len(v) for k, v in vint.items()})
json.dump(res, open(os.path.join(C, 's6_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(K[['year', 'prov_pat_reg', 'kipo_pat_reg', 'wipo_pat_grant', 'kipo_vs_wipo_diff', 'ratio_pat', 'd_vs_2019_23', 'd_vs_trend', 'prov_missing_reg_br', 'br_cover', 'ratio_pat_corrected', 'd_corrected_vs_2019_23', 'prov_um_reg', 'kipo_um_reg', 'ratio_um']].round(4).to_string())
print(Cn[['year', 'prov_inv_by_rgst', 'cnipa_inv_grant', 'wipo_cn_grant', 'ratio_inv_rgst', 'prov_um_by_rgst', 'cnipa_um_grant', 'ratio_um_rgst', 'prov_inv_by_pub', 'wipo_cn_pub', 'ratio_inv_pub', 'prov_all_by_t3year', 'miss_all_by_t3year', 'ratio_t3year']].round(4).to_string())
print(json.dumps({k: v for k, v in res.items() if k != 'cn_2021'}, ensure_ascii=False, default=float))
