# verify_2: s3 재계산(독립 hard() 구현) + 제공 사건 재구성 일치 + 고유 사건(창 겹침 제거) 자카드
import sys, time; sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4\scripts')
import numpy as np, pandas as pd
from lib_cd import load, build, KSIC50, DEV, C
V = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4\verify_2'
T0=time.time()
own, pat, app, docs, orig = load()
assert own.d.max() < '20250101'
lab = pd.read_parquet(C + r'\kr_publab_src.parquet').set_index('appl_no')
lab['mg']=lab.sg.str.split('/').str[0]; lab['sc']=lab.sg.str[:4]
def hard2(t0, levels, nmin):
    tr = lab[lab.appl_year<=t0]
    out = pd.Series(index=lab.index, dtype=object)
    for l in levels:
        s = tr[[l,'k_free']].dropna()
        cnt = s.value_counts().rename('n').reset_index()
        cnt = cnt.sort_values([l,'n','k_free'], ascending=[True,False,True])
        tot = cnt.groupby(l).n.sum()
        md = cnt.drop_duplicates(l).set_index(l).k_free
        md = md[tot.reindex(md.index)>=nmin]
        v = lab[l].map(md)
        fill = out.isna() & v.notna()
        out[fill] = v[fill]
    return out
P = pd.read_parquet(C + r'\dev_pos.parquet')
prov_doc = docs.ksic.where(docs.ksic.isin(KSIC50)); prov_pat = dict(zip(pat.rgst_no, pat.ksic))
rows=[]; EVall={}
for t0 in DEV:
    st,pos,_,_ = build(t0, own, docs, app, orig, prov_doc, prov_pat)
    Ep=set(zip(pos.cd,pos.k)); Ep0=set(zip(P[P.t0==t0].cd,P[P.t0==t0].k))
    assert Ep==Ep0, ('prov mismatch',t0)
    EVall.setdefault('PROV',[]).append(pos.assign(t0=t0)[['t0','cd','k','d','rgst_no']])
    labs={'TXT-FREE':lab.k_free,'FREE-IPC(H-sg-3)':hard2(t0,['sg','mg','sc'],3),'H-sc-1':hard2(t0,['sc'],1)}
    for nm,L in labs.items():
        st,pa,_,_ = build(t0, own, docs, app, orig, docs.appl_no.map(L), dict(zip(pat.rgst_no, pat.appl_no.map(L))))
        Ea=set(zip(pa.cd,pa.k)); i=len(Ea&Ep); u=len(Ea|Ep)
        rows.append(dict(t0=t0,assign=nm,ev_pub=len(Ea),ev_prov=len(Ep),inter=i,jac=round(i/u,4),rec=round(i/len(Ep),4),
             same_rgst_share=None))
        EVall.setdefault(nm,[]).append(pa.assign(t0=t0)[['t0','cd','k','d','rgst_no']])
    print(t0, round(time.time()-T0), flush=True)
R=pd.DataFrame(rows)
pool=R.groupby('assign')[['ev_pub','ev_prov','inter']].sum(); pool['jac']=(pool.inter/(pool.ev_pub+pool.ev_prov-pool.inter)).round(4); pool['rec']=(pool.inter/pool.ev_prov).round(4)
print(pool); print('max per-origin jac', R.jac.max(), R.loc[R.jac.idxmax()].to_dict())
# 고유 사건(원점 무관 (cd,k,첫 이전 rgst_no)) 기준
pv=pd.concat(EVall['PROV']); U_p=set(zip(pv.cd,pv.k))
for nm in ['TXT-FREE','FREE-IPC(H-sg-3)','H-sc-1']:
    a=pd.concat(EVall[nm]); U_a=set(zip(a.cd,a.k))
    print('unique (cd,k)', nm, len(U_a), len(U_p), len(U_a&U_p), round(len(U_a&U_p)/len(U_a|U_p),4))
    # 같은 (t0,cd)에서 매입 특허는 같지만 라벨만 다른 경우 vs 사건 자체 다른 경우
    ka=set(zip(a.t0,a.cd)); kp=set(zip(pv.t0,pv.cd))
    print('  firm-level (t0,cd) jac', round(len(ka&kp)/len(ka|kp),4), 'pub',len(ka),'prov',len(kp))
R.to_csv(V+r'\v2_b_essentiality_recalc.csv',index=False,encoding='utf-8-sig')
for k,v in EVall.items(): pd.concat(v).to_parquet(V+rf'\v2_b_ev_{k.replace("(","_").replace(")","")}.parquet',index=False)
print('done',round(time.time()-T0))
