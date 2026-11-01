from lab2 import *
M = {k: member_cache(k, v) for k, v in BASE.items()}
ens = combine(list(M.values())); r = rows_from(ens)
r['nb'] = NB.ase.values
g = r.groupby(['ksic', 'cc']).agg(ase=('ase', 'sum'), nb=('nb', 'sum'), bias=('f', 'sum'), y=('y', 'sum'))
g['share'] = g.ase / r.ase.sum(); g['bias'] = (g.bias - g.y) / g.y
print(g.sort_values('ase', ascending=False).head(15).round(3))
r['sgn'] = np.sign(r.f - r.y) * r.ase
print('부호있는 척도오차 평균(양=과대) by cc,h'); print(r.pivot_table(index='cc', columns='h', values='sgn', aggfunc='mean').round(2))
# 과거 1-step 잔차 서프라이즈 s = g_l1 - g8(직전) 와 이후 오차의 관계
for o in FR.VAL_ORIGINS:
    L = np.log1p(D['n'][:, :o - 1999]); s = (L[:, -1] - L[:, -2]) - (L[:, -2] - L[:, -10]) / 8
    le = np.log1p(D['n'][:, o - 1999 + 2]) - np.log1p(ens[o][:, 2])
    print(o, 'corr(surprise, h3 로그오차)', round(np.corrcoef(s, le)[0, 1], 3))
