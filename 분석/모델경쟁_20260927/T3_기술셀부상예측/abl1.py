from models import *
D=enrich(H.data())
base=MOM+FOR+NEW
groups={'app(출원인포트폴리오)':['app_mom','app_size','big_sh','top1_sh'],'nb(이웃)':['nb_kmom','nb_tmom','nb_w','nb_deg'],
 'sub(서브클래스)':['sub_kmom','sub_fmom','sub_krsh'],'long(장기이력)':['kr_long','kr_drawdown','kr_vs_long','tot_vs_long'],
 'rel(상대점유율)':['kr_share_rel_ksic','kr_share_rel_sub','kr_share_rel_mg'],
 'US':[c for c in FOR if c.startswith('US')],'CN':[c for c in FOR if c.startswith('CN')],'JP':[c for c in FOR if c.startswith('JP')],'EP':[c for c in FOR if c.startswith('EP')],
 'fx/share':['fx_mom','fx_mom_l1','tot_mom','tot_lvl','kr_share','kr_share_chg','gap_fx_kr']}
R,_=H.rolling(logit(base),D); b=H.summarize(R)['auc']; print('full',round(b,4))
for g,cols in groups.items():
    R,_=H.rolling(logit([c for c in base if c not in cols]),D); a=H.summarize(R)['auc']; print('drop',g,round(a,4),'delta',round(a-b,4),flush=True)
# 추가 후보: STR, GRP
for g,cols in {'STR':STR,'GRP':GRP}.items():
    R,_=H.rolling(logit(base+cols),D); a=H.summarize(R)['auc']; print('add',g,round(a,4),'delta',round(a-b,4))
