# python logv.py <ver> <kept> <change> <hyp> [extra note]
import sys, json
sys.stdout.reconfigure(encoding='utf-8')
from evalkit import log
v, kept, change, hyp = sys.argv[1:5]; extra = sys.argv[5] if len(sys.argv) > 5 else ''
m = json.load(open(f'cache/metrics_{v}.json')); b = json.load(open('cache/v0_metrics.json'))['y_MAIN']
note = f"PR-AUC {m['prauc']:.4f} (기준선 {b['prauc']:.4f}), p@1% {m['p1']:.4f}; AUC 95%CI(양도인 군집) {m['ci_auc'][0]:.3f}-{m['ci_auc'][1]:.3f}" + (' | ' + extra if extra else '')
log(v, change, hyp, m['auc'], b['auc'], kept, note)
