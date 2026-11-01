# C16-T3 s3: 제공 CN 조직 비중 연도 계열(공개연도·출원연도, 전체·국내, KSIC 2015 표준화) + 공식 대조 → cn_org_share_vs_official.csv
# 사용: python -I s3_series.py <cache_dir> <out_csv> <c13_s6_checks_json>
import sys, os, json
import pandas as pd, numpy as np
C, OUT, S6 = sys.argv[1], sys.argv[2], sys.argv[3]
sys.stdout.reconfigure(encoding='utf-8')
d = pd.read_parquet(os.path.join(C, 'cn_docs.parquet'))
off = pd.read_csv(os.path.join(C, 'cnipa_official.csv'), encoding='utf-8-sig').set_index('year')
d['anyorg'] = d.n_orgkey >= 1
d['multi'] = d.n_orgkey >= 2
d['nm_miss'] = d.n_names == 0
dom = ~d.foreign


def ksic_std(g, flag, base_w):
    s = g.groupby('ksic')[flag].mean()
    w = base_w.reindex(s.index).fillna(0)
    return float((s * w).sum() / w.sum())


rows = []
res = {}
for basis, ycol, yrs in [('pub', 'py', range(2015, 2025)), ('appl', 'ay', range(2015, 2024))]:
    sub = d[d[ycol].notna() & d.pub.notna() & (d.pub < 20250801)].copy()   # 공개된 문서만(2025-07까지)
    sub[ycol] = sub[ycol].astype(int)
    b = sub[(sub[ycol] == 2015) & ~sub.foreign & ~sub.app1_miss]
    base_w = b.ksic.value_counts(normalize=True)
    for y in yrs:
        g = sub[sub[ycol] == y]
        gd = g[~g.foreign]; gdn = gd[~gd.app1_miss]
        r = dict(basis=basis, year=y, n_docs=len(g), n_dom=len(gd), foreign_share=float(g.foreign.mean()),
                 typ8_share=float((g.typ == '8').mean()), prio_for_share=float(g.prio_for.mean()),
                 app1_miss=float(g.app1_miss.mean()),
                 c13_anyorg_all=float(g.anyorg.mean()),            # C13-T4 정의(결측=비조직, 외국 포함)
                 anyorg_all=float(g[~g.nm_miss].anyorg.mean()),
                 app1org_all=float(g[~g.app1_miss].app1_org.mean()),
                 multi_all=float(g.multi.mean()),
                 app1org_for=float(g[g.foreign & ~g.app1_miss].app1_org.mean()),
                 anyorg_dom=float(gd[~gd.nm_miss].anyorg.mean()),
                 app1org_dom=float(gdn.app1_org.mean()),
                 multi_dom=float(gd.multi.mean()),
                 app1org_dom_ksicstd=ksic_std(gdn, 'app1_org', base_w),
                 anyorg_dom_ksicstd=ksic_std(gd[~gd.nm_miss], 'anyorg', base_w),
                 multi_dom_ksicstd=ksic_std(gd, 'multi', base_w))
        rows.append(r)
S = pd.DataFrame(rows)
# 공식 계열: appl → 같은 연도, pub → 공식 y-1 (보조)
S['off_year'] = np.where(S.basis == 'appl', S.year, S.year - 1)
S['off_svc_share'] = S.off_year.map(off.svc_share_inv)
S['off_dom_inv'] = S.off_year.map(off.dom_inv)
S['coverage_dom_vs_off'] = S.n_dom / S.off_dom_inv
for col in ['app1org_dom_ksicstd', 'anyorg_dom_ksicstd', 'app1org_dom', 'c13_anyorg_all']:
    for basis in ['pub', 'appl']:
        m = S.basis == basis
        b = S.loc[m & (S.year == 2015), col].iloc[0]
        S.loc[m, f'd_{col}'] = S.loc[m, col] - b
for basis in ['pub', 'appl']:
    m = S.basis == basis
    b = S.loc[m & (S.year == 2015), 'off_svc_share'].iloc[0]
    S.loc[m, 'd_off'] = S.loc[m, 'off_svc_share'] - b
S['gap_pp'] = (S.d_app1org_dom_ksicstd - S.d_off) * 100          # 주 판정 지표
S['gap_pp_anyorg'] = (S.d_anyorg_dom_ksicstd - S.d_off) * 100    # 보조
S['level_gap_pp'] = (S.app1org_dom_ksicstd - S.off_svc_share) * 100
# 외국 몫 반박: 외국 몫 2015 고정 시 전체 조직 비중
for basis in ['pub', 'appl']:
    m = S.basis == basis
    f15 = S.loc[m & (S.year == 2015), 'foreign_share'].iloc[0]
    s_for = S.loc[m, 'app1org_for']; s_dom = S.loc[m, 'app1org_dom']
    S.loc[m, 'app1org_all_fix_for15'] = f15 * s_for + (1 - f15) * s_dom
    s_all15 = S.loc[m & (S.year == 2015), 'app1org_all'].iloc[0]
    rise = S.loc[m, 'app1org_all'] - s_all15
    S.loc[m, 'foreign_contrib_share'] = np.where(rise.abs() > 1e-9, (S.loc[m, 'app1org_all'] - S.loc[m, 'app1org_all_fix_for15']) / rise, np.nan)
# C13-T4 재현 대조(공개연도, 결측=비조직, 외국 포함)
s6 = json.load(open(S6, encoding='utf-8'))
c13 = {r['y']: r for r in s6['CN']['by_year']}
S['c13_org_doc'] = [c13.get(y, {}).get('org_doc') if b == 'pub' else np.nan for b, y in zip(S.basis, S.year)]
S['c13_multi'] = [c13.get(y, {}).get('multi_org') if b == 'pub' else np.nan for b, y in zip(S.basis, S.year)]
S['c13_docs'] = [c13.get(y, {}).get('docs') if b == 'pub' else np.nan for b, y in zip(S.basis, S.year)]
S.to_csv(OUT, index=False, encoding='utf-8-sig', float_format='%.5f')
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
print(S[['basis', 'year', 'n_docs', 'c13_docs', 'c13_anyorg_all', 'c13_org_doc', 'multi_all', 'c13_multi', 'foreign_share', 'app1org_all', 'app1org_dom', 'app1org_dom_ksicstd', 'off_svc_share', 'gap_pp', 'gap_pp_anyorg', 'level_gap_pp', 'coverage_dom_vs_off', 'foreign_contrib_share', 'multi_dom_ksicstd']].round(4).to_string())
