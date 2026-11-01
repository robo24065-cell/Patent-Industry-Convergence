# 독립 재계산: 저장된 묶음 예측에서 상위 1% 적중률·포착률. 동점 처리 민감도(낙관/비관/평균) 포함.
import sys, glob, os, json
import numpy as np, pandas as pd
base = sys.argv[1]
rows = []
cols = {'n4':('p_champ',None),'n6':('p_champ',None),'n7':('p_champ','p_SB'),'n8':('p_champ','p_SB'),'n9':('p_champ',None)}
def t1(y, p, k):
    p = np.asarray(p, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    thr = np.sort(p)[::-1][k-1]
    above = p > thr; tie = p == thr
    need = k - above.sum()
    h_above = y[above].sum(); ty = y[tie].sum(); tn = tie.sum()
    lo = h_above + max(0, need - (tn - ty)); hi = h_above + min(need, ty)
    exp = h_above + need * ty / tn
    return lo, hi, exp, int(tn)
for t,(c,s) in cols.items():
    for f in sorted(glob.glob(os.path.join(base,'cache',t,'preds_b*.parquet'))):
        b = int(os.path.basename(f)[7:-8])
        d = pd.read_parquet(f)
        y = d.y.values; n = len(y); pos = y.sum()
        k = max(1, int(np.floor(0.01*n + 0.5)))
        lo,hi,exp,tn = t1(y, d[c].values, k)
        r = dict(track=t.upper(), bundle=b, n=n, pos=int(pos), k=k, hits_lo=lo, hits_hi=hi, hits_exp=exp, ties_at_thr=tn,
                 prec_exp=exp/k, cap_exp=exp/pos)
        if s:
            lo2,hi2,exp2,tn2 = t1(y, d[s].values, k)
            r.update(sb_hits_lo=lo2, sb_hits_hi=hi2, sb_cap_exp=exp2/pos, sb_ties=tn2)
        rows.append(r)
R = pd.DataFrame(rows)
pd.set_option('display.width',250)
print(R.to_string())
R.to_csv(os.path.join(base,'verify_1','v1_top1.csv'), index=False, encoding='utf-8-sig')
