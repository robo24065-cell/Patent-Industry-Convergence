import pandas as pd, numpy as np, sys
d=pd.read_csv(sys.argv[1]+'/seed_bundles.csv')
for t,g in d.groupby('track',sort=False):
    o=g[g.official_bundle==True].iloc[0]
    se=(o.ci_hi-o.ci_lo)/3.92; sse=(o.strong_ci_hi-o.strong_ci_lo)/3.92
    sd=g.skill.std(ddof=1); ssd=g.skill_vs_strong.std(ddof=1)
    sd0=g.skill.std(ddof=0)
    w=g.ci_hi-g.ci_lo; sw=g.strong_ci_hi-g.strong_ci_lo
    print(t,len(g),'sd',round(sd,4),'sd0',round(sd0,4),'se',round(se,4),'ratio',round(sd/se,3),'| strong sd',round(ssd,4),'se',round(sse,4),'ratio',round(ssd/sse,3))
    print('   ci_lo range',round(g.ci_lo.min(),4),round(g.ci_lo.max(),4),'max |dev from official|',round((g.ci_lo-o.ci_lo).abs().max(),4),'sd(ci_lo)',round(g.ci_lo.std(),4),'width range',round(w.min(),4),round(w.max(),4))
    print('   strong lo range',round(g.strong_ci_lo.min(),4),round(g.strong_ci_lo.max(),4),'max |dev|',round((g.strong_ci_lo-o.strong_ci_lo).abs().max(),4),'sd',round(g.strong_ci_lo.std(),4),'width range',round(sw.min(),4),round(sw.max(),4))
    print('   skill max dev',round((g.skill-o.skill).abs().max(),4),'strongD max dev',round((g.skill_vs_strong-o.skill_vs_strong).abs().max(),4))
