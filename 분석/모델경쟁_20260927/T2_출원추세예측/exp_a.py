from lab2 import *
M = {k: member_cache(k, v) for k, v in BASE.items()}
for k in M: rep(rows_from(M[k]), k)
ens = combine(list(M.values())); S0 = rep(rows_from(ens), 'v29_repro')
S = {}
S['geo'] = rep(rows_from(combine(list(M.values()), 'geo')), 'geo')
S['median'] = rep(rows_from(combine(list(M.values()), 'median')), 'median')
EB = member_cache('ebdrift', {'k': 8, 'phi': 0.8, 'kc': 8}); rep(rows_from(EB), 'ebdrift')
S['eb4'] = rep(rows_from(combine(list(M.values()) + [EB])), 'ens+eb')
S['eb_rep_td'] = rep(rows_from(combine([M['ridge_lead'], M['flow_um'], EB])), 'eb replaces topdown')
