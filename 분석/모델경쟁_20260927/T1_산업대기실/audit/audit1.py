# 2단계 감사1: 누수·모집단·중복 점검 (검증 c=2009만, 🔒 홀드아웃 결과 미조회)
import os, sys; os.environ['T1_STRICT'] = '1'; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from final_model import *
from sklearn.metrics import roc_auc_score, average_precision_score
t0 = time.time()
tr, te = load_final()
sc = pd.read_parquet(CACHE/'v31_scores.parquet'); assert (sc.firm.values == te.firm.values).all()
te['s'] = sc.s.values; te['sb'] = sc.sb.values
print('rows', len(te), 'dup (firm,k)', te.duplicated(['firm', 'k']).sum(), 'train dup (cut,firm,k)', tr.duplicated(['cut', 'firm', 'k']).sum())
# 1) 모집단: 가시(등록≤기준시점 연말) 자사특허 수
nvis = np.expm1(te.lsize.values).round()
for thr in (0, 1, 3):
    m = nvis >= thr; firms = te[m].firm.nunique()
    print(f'[pop vis>={thr}] firms={firms} rows={m.sum()} y={te.y[m].sum()} rate={te.y[m].mean():.4f} | v31 auc_all={roc_auc_score(te.y[m], te.s[m]):.4f} strict_base={roc_auc_score(te.y[m], te.sb[m]):.4f}', end='')
    mn = m & te.newc.values
    print(f' | new rows={mn.sum()} y={te.y[mn].sum()} auc_new={roc_auc_score(te.y[mn], te.s[mn]):.4f} pr_new={average_precision_score(te.y[mn], te.s[mn]):.4f}')
g = te.groupby('firm').agg(n=('lsize', 'first'), y=('y', 'max'))
print('firm-level any-entry rate by visible history 0 vs >=3:', g[np.expm1(g.n).round() == 0].y.mean().round(3), g[np.expm1(g.n).round() >= 3].y.mean().round(3))
ntr = np.expm1(tr.lsize.values).round()
print('train rows vis0 share', (ntr == 0).mean().round(3), 'vis<3 share', (ntr < 3).mean().round(3), 'by cut:', tr.assign(v=ntr < 3).groupby('cut').v.mean().round(3).to_dict())
# 2) pa_unreg_n: 스냅샷 등록여부(reg) 사용 여부
a = pd.read_parquet(CACHE/'app_pub.parquet'); a['rgst_date'] = a.rgst_date.fillna('').astype(str); a['pub_date'] = a.pub_date.fillna('').astype(str)
e = a[(a.ay <= 2009) & (a.ay >= 2000) & (((a.pub_date.str.len() == 8) & (a.pub_date <= '20091231')) | ((a.rgst_date.str.len() == 8) & (a.rgst_date <= '20091231')))]
never = ~e.reg; pend = ~((e.rgst_date.str.len() == 8) & (e.rgst_date <= '20091231'))
print('pub by 2009: n', len(e), 'never registered (snapshot reg=False)', int(never.sum()), 'not registered as of 2009-12-31', int(pend.sum()))
# 3) pa_k>0 in risk set
m = te.pa_k > 0
print('risk rows pa_k>0', int(m.sum()), 'y rate', te.y[m].mean().round(4), 'vs', te.y[~m].mean().round(4))
# 4) DART snapshot
print('DART-matched rows', int(te.has_dart.sum()), 'dm4 rate y', te[te.dm4 == 1].y.mean().round(4), 'dm2', te[te.dm2 == 1].y.mean().round(4))
# 5) 결과창 확인
D = load(); ins = D['ins']; fut = ins[(ins.ay > 2009) & (ins.ay <= 2014)]
print('outcome filing years', fut.ay.min(), fut.ay.max(), 'rgst max', fut.rgst_date.max())
print('time', round(time.time() - t0))
