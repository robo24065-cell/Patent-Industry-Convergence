# NBER-CES(미국 제조업 NAICS2012 6자리, 1958-2018) <-> 50 KSIC 수작업 연결 + 의미 검정
# 가설: 미국 시장의 산업별 수요(출하액·고용) 증가가 1~3년 뒤 그 KSIC의 (미국·5국) 특허 출원 증가로 이어진다.
import pandas as pd, numpy as np, pathlib, statsmodels.formula.api as smf
from scipy.stats import spearmanr
OUT = pathlib.Path(r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext")
W = OUT / 'demand_work'
MAP = {  # ksic: ([naics...], quality)
'15220': (['316210'], 'proxy'), '20202': (['325211'], 'direct'), '20411': (['325510'], 'direct'), '20423': (['325620'], 'direct'),
'20491': (['325992'], 'direct'), '21220': (['325411'], 'proxy'), '23992': (['327910'], 'direct'), '24191': (['331110'], 'proxy'),
'24219': (['331410', '331491'], 'direct'), '25914': (['332119'], 'direct'), '25922': (['332813'], 'direct'), '25924': (['332710'], 'direct'),
'25932': (['332510'], 'direct'), '26111': (['334413'], 'proxy'), '26112': (['334413'], 'proxy'), '26121': (['334413'], 'proxy'),
'26129': (['334413'], 'proxy'), '26211': (['334419'], 'proxy'), '26212': (['334419'], 'proxy'), '26223': (['334412'], 'direct'),
'26224': (['334418'], 'direct'), '26291': (['334416'], 'direct'), '26322': (['334118'], 'proxy'), '26323': (['334118'], 'proxy'),
'26422': (['334220'], 'proxy'), '26429': (['334220'], 'proxy'), '26519': (['334310'], 'proxy'), '26529': (['334310'], 'proxy'),
'27199': (['339112', '339113'], 'proxy'), '27211': (['334511'], 'direct'), '27212': (['334515'], 'direct'), '27301': (['333314'], 'direct'),
'28111': (['335312'], 'direct'), '28122': (['335931'], 'direct'), '28511': (['335210', '335221', '335222'], 'direct'), '28902': (['335991'], 'proxy'),
'29131': (['333911'], 'direct'), '29132': (['333912'], 'direct'), '29133': (['332911', '332912', '332919'], 'direct'), '29150': (['333994'], 'direct'),
'29162': (['333921'], 'direct'), '29210': (['333111'], 'direct'), '29229': (['333517'], 'proxy'), '29241': (['333120', '333131'], 'direct'),
'29280': (['333999'], 'proxy'), '30393': (['336360'], 'direct'), '31202': (['336510'], 'direct'), '31312': (['336411'], 'proxy'),
'33301': (['339920'], 'proxy'), '33309': (['339920'], 'proxy'),
}
d = pd.read_csv(OUT / 'demand_nberces5818v1_n2012.csv', dtype={'naics': str})
rows = [dict(ksic=k, naics=n, quality=q) for k, (ns, q) in MAP.items() for n in ns]
m = pd.DataFrame(rows)
m['in_nber'] = m.naics.isin(set(d.naics))
print('NAICS 매칭 실패:', m[~m.in_nber].to_dict('records'))
m.to_csv(OUT / 'demand_ksic_naics2012_map.csv', index=False, encoding='utf-8-sig')
x = m[m.in_nber].merge(d, on='naics')
k = x.groupby(['ksic', 'year']).agg(emp=('emp', 'sum'), vship=('vship', 'sum'), vadd=('vadd', 'sum'), invest=('invest', 'sum'),
                                    piship=('piship', 'mean'), tfp5=('tfp5', 'mean')).reset_index()
k['rship'] = k.vship / k.piship   # 실질 출하
k.to_parquet(OUT / 'demand_ksic_us_mfg_panel_1958_2018.parquet', index=False)
print('연결 KSIC:', k.ksic.nunique(), '/50')

p = pd.read_parquet(W / 'demand_pat_ksic_cc_year_le2016.parquet'); p['ksic'] = p.ksic.astype(str)
res = []
for cc_set, lab in [(['US'], 'US'), (['KR'], 'KR'), (['KR', 'US', 'JP', 'CN', 'EP'], '5국')]:
    pc = p[p.cc.isin(cc_set)].groupby(['ksic', 'year']).n.sum().unstack(fill_value=0).reindex(columns=range(1995, 2017), fill_value=0)
    for var in ['emp', 'vship', 'rship', 'piship', 'invest']:
        lv = np.log(k.pivot(index='ksic', columns='year', values=var).clip(lower=1e-6))
        rr = []
        for t in range(2001, 2014):
            for kk in lv.index:
                if kk not in pc.index: continue
                past = pc.loc[kk, [t - 2, t - 1, t]].sum(); fut = pc.loc[kk, [t + 1, t + 2, t + 3]].sum()
                if past < 10: continue
                # 시점정합: t년 원점에서 ASM 값은 약 1.5년 시차 -> t-1까지 사용 (t-4 -> t-1 3년 변화)
                rr.append(dict(ksic=kk, t=t, dx=lv.loc[kk, t - 1] - lv.loc[kk, t - 4],
                               gpat=np.log(fut + 1) - np.log(past + 1),
                               gpast=np.log(past + 1) - np.log(pc.loc[kk, [t - 5, t - 4, t - 3]].sum() + 1) if t >= 2000 else np.nan))
        r = pd.DataFrame(rr).dropna()
        for c in ['dx', 'gpat', 'gpast']:
            r[c + '_w'] = r[c] - r.groupby('t')[c].transform('mean')
        rho, pv = spearmanr(r.dx_w, r.gpat_w)
        cl = {'groups': r.ksic.astype('category').cat.codes}
        f1 = smf.ols('gpat_w ~ dx_w + gpast_w', r).fit(cov_type='cluster', cov_kwds=cl)
        f2 = smf.ols('gpat ~ dx + gpast + C(ksic) + C(t)', r).fit(cov_type='cluster', cov_kwds=cl)
        res.append(dict(pat=lab, var=var, n=len(r), n_ksic=r.ksic.nunique(), spearman_yrFE=round(rho, 3), p=round(pv, 4),
                        beta_yrFE_ctrl=round(f1.params['dx_w'], 3), p1=round(f1.pvalues['dx_w'], 4),
                        beta_2wayFE=round(f2.params['dx'], 3), p2=round(f2.pvalues['dx'], 4)))
res = pd.DataFrame(res); pd.set_option('display.width', 250); print(res.to_string())
res.to_csv(W / 'demand_test_us_mfg_lead.csv', index=False, encoding='utf-8-sig')
