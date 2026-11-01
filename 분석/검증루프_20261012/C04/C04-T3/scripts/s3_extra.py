# C04-T3 s3: 보충 분석(판정 기준 불변, PREREG 변경 기록 참조)
#  (1) 개발 원점 SAB 비율 추세 외삽  (2) 상위 100셀 구성  (3) 레이블 정의 점검(로그 성장 대 절대 증가)
#  (4) 분할·사명변경 별칭(N3)·기업집단 키(G) 민감도  (5) 층 × 대체 레이블 A  (6) CN 2021 결측 비례 배분 시 새 적격 셀의 층
import os, sys, json, hashlib, html, re
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
W = os.path.join(ROOT, r"검증루프_20261012\C04\C04-T3"); C = os.path.join(W, 'cache')
ac = pd.read_csv(os.path.join(W, 'applicant_concentration.csv'), dtype={'ksic': str})
dev = pd.read_csv(os.path.join(W, 'dev_sab_by_origin.csv'))
out = {}
def auc(y, p): return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan
def sk(y, p, b):
    a, a0 = auc(y, p), auc(y, b); return dict(n=int(len(y)), pos=int(y.sum()), auc=a, auc_base=a0, skill=(a - a0) / (1 - a0))

# (1) 추세
b = np.polyfit(dev.origin, dev.sab_rate_k1, 1); out['dev_trend'] = dict(slope=float(b[0]), extrap_2021=float(np.polyval(b, 2021)),
    resid_sd=float(np.std(dev.sab_rate_k1 - np.polyval(b, dev.origin), ddof=2)), final=float(ac.sab_k1[ac.yb == 1].mean()))
b2 = np.polyfit(dev.origin, dev.pos_krW_le1, 1); out['dev_trend_pos_krW_le1'] = dict(extrap_2021=float(np.polyval(b2, 2021)), final=float((ac.kr_W[ac.yb == 1] <= 1).mean()))
print(out['dev_trend'], out['dev_trend_pos_krW_le1'])

# (2) 상위 100·504 구성
o = np.argsort(-ac.p.values)
for k in [100, 504]:
    t = ac.iloc[o[:k]]; h = t[t.yb == 1]
    out[f'top{k}'] = dict(hits=int(len(h)), hits_krW_le1=int((h.kr_W <= 1).sum()), hits_krW0=int((h.kr_W == 0).sum()), hits_D_le3=int((h.D <= 3).sum()),
                          hits_sab=int(h.sab_k1.sum()), hits_kr_fut_le3=int((h.kr_fut <= 3).sum()), cells_krW_le1=int((t.kr_W <= 1).sum()),
                          hits_D_median=float(h.D.median()), hits_D_ge10=int((h.D >= 10).sum()))
print(out['top100'], out['top504'])

# (3) 레이블 정의: 절대 증가 상위 10%와의 겹침
k = 504; hk = lambda cs: np.array([int(hashlib.md5(s.encode()).hexdigest()[:8], 16) for s in cs], dtype=np.int64)
oa = np.lexsort((hk(ac.cell.values), -ac.y.values, -ac.D.values)); yabs = np.zeros(len(ac), int); yabs[oa[:k]] = 1
out['label_def'] = dict(overlap_logtop_abstop=int(((yabs == 1) & (ac.yb == 1)).sum()), abs_label_krW_le1=int(((yabs == 1) & (ac.kr_W <= 1)).sum()),
                        log_label_krW_le1=int(((ac.yb == 1) & (ac.kr_W <= 1)).sum()), abs_label_min_D=float(ac.D[yabs == 1].min()),
                        model_on_abs_label=sk(yabs, ac.p.values, ac.p_base.values))
print(out['label_def'])

# (4) N3·G 민감도
docs = pd.read_parquet(os.path.join(C, 'kr_docs.parquet')); docs['cell'] = docs.ksic + '|' + docs.mg
def n1(s):
    s = html.unescape(s or '').lower()
    for w in ['주식회사', '(주)', '㈜', '(株)', '（주）']: s = s.replace(w, '')
    return re.sub(r"[\s\.,·\-\(\)\[\]'\"“”‘’&/]+", '', s)
ALIAS = {'엘지에너지솔루션': '엘지화학', '에스케이온': '에스케이이노베이션', '에이치엘만도': '만도', '만도모빌리티솔루션즈': '만도', '두산에너빌리티': '두산중공업',
         '에이치디현대중공업': '현대중공업', '한국조선해양': '현대중공업', '에이치디한국조선해양': '현대중공업', '포스코홀딩스': '포스코', '에이치디현대인프라코어': '두산인프라코어',
         '현대자동차': '현대자동차', '에스케이시그넷': '시그넷이브이'}
GROUP = [('삼성', '삼성'), ('엘지', '엘지'), ('에스케이', '에스케이'), ('현대', '현대'), ('기아', '현대'), ('에이치디현대', '에이치디현대'), ('한화', '한화'), ('포스코', '포스코'),
         ('롯데', '롯데'), ('두산', '두산'), ('씨제이', '씨제이'), ('에이치엘', '에이치엘'), ('만도', '에이치엘'), ('효성', '효성'), ('코오롱', '코오롱'), ('케이티', '케이티')]
names = pd.Series(docs.app_raw.fillna('').unique()); m1 = {x: n1(x) for x in names}
def g_of(k):
    for pre, g in GROUP:
        if k.startswith(pre): return 'G:' + g
    return k
docs['k1'] = docs.app_raw.fillna('').map(m1); docs['k3'] = docs.k1.map(lambda x: ALIAS.get(x, x)); docs['kg'] = docs.k3.map(g_of)
def sab(key, t0=2021, d=ac):
    sub = docs[docs.cell.isin(set(d.cell)) & (docs.year >= t0 - 2) & (docs.year <= t0 + 3)]
    g = sub.assign(w=np.where(sub.year <= t0, -1, 1)).groupby(['cell', key]).w.sum()
    mx = g.groupby('cell').max().reindex(d.cell.values).fillna(0).values
    return (d.yb.values == 1) & (d.D.values > 0) & (mx >= 0.5 * d.D.values)
base = sk(ac.yb.values, ac.p.values, ac.p_base.values)
for key in ['k1', 'k3', 'kg']:
    s = sab(key); assert key != 'k1' or (s == ac.sab_k1.values).all()
    keep = ~s; rA = sk(ac.yb.values[keep], ac.p.values[keep], ac.p_base.values[keep]); yB = ac.yb.values.copy(); yB[s] = 0; rB = sk(yB, ac.p.values, ac.p_base.values)
    out[f'sab_{key}'] = dict(n=int(s.sum()), rate=float(s.sum() / 504), rate_D5=float((s & (ac.D.values >= 5)).sum() / ((ac.yb == 1) & (ac.D >= 5)).sum()),
                             dskill_A=rA['skill'] - base['skill'], dskill_B=rB['skill'] - base['skill'],
                             by_stratum={st: float(s[(ac.stratum == st).values].sum() / ((ac.stratum == st) & (ac.yb == 1)).sum()) for st in ['S0', 'S1', 'S2']})
    print(key, out[f'sab_{key}'])
# 기업집단 키로 본 양성 증가분 중 상위 5 집단 몫
sub = docs[docs.cell.isin(set(ac.cell[ac.yb == 1])) & (docs.year >= 2019)]
gg = sub.assign(w=np.where(sub.year <= 2021, -1, 1)).groupby('kg').w.sum().sort_values(ascending=False)
out['pos_increase_top_groups'] = {k: float(v / ac.D[ac.yb == 1].sum()) for k, v in gg.head(8).items()}
print(out['pos_increase_top_groups'])

# (5) 층 × 대체 레이블 A
out['strata_altA'] = {}
for st in ['S0', 'S1', 'S2']:
    msk = (ac.stratum == st).values; keep = msk & ~ac.sab_k1.values
    out['strata_altA'][st] = dict(orig=sk(ac.yb.values[msk], ac.p.values[msk], ac.p_base.values[msk]), A=sk(ac.yb.values[keep], ac.p.values[keep], ac.p_base.values[keep]))
print(json.dumps(out['strata_altA'], ensure_ascii=False))
# SAB 양성 대 비SAB 양성의 모형 순위분위
r = pd.Series(ac.p.values).rank(pct=True).values
out['rank_pct_pos'] = dict(sab=float(np.median(r[ac.sab_k1.values])), nonsab=float(np.median(r[(ac.yb == 1).values & ~ac.sab_k1.values])))

# (6) CN 2021 결측 비례 배분(가상 A 근사): CN 창(2019–21) 셀 문서 수 × 1/(1−KSIC·연도 결측률)
pan = pd.read_parquet(os.path.join(C, 'panel.parquet')); pan['cell'] = pan.ksic + '|' + pan.mg
cn = pd.read_parquet(os.path.join(C, 'cn_missing_ky.parquet')); cn['f'] = cn.n_all / cn.n_valid.where(cn.n_valid > 0)
w = pan[(pan.year >= 2019) & (pan.year <= 2021)].copy()
w = w.merge(cn[['ksic', 'year', 'f']], on=['ksic', 'year'], how='left'); w['f'] = np.where(w.cc == 'CN', w.f.fillna(1.0), 1.0); w['n2'] = w.n * w.f
cw = w.groupby('cell').agg(tot=('n', 'sum'), tot2=('n2', 'sum'))
kr = w[w.cc == 'KR'].groupby('cell').n.sum(); cw['kr_W'] = kr.reindex(cw.index).fillna(0)
cur = cw[cw.tot >= 50]; new = cw[(cw.tot < 50) & (cw.tot2 >= 50)]
st = lambda s: {'S0': int((s == 0).sum()), 'S1': int((s == 1).sum()), 'S2': int((s >= 2).sum())}
out['cn_realloc'] = dict(n_current=int(len(cur)), n_new=int(len(new)), new_by_stratum=st(new.kr_W), current_by_stratum=st(cur.kr_W),
                         new_share_krW_le1=float((new.kr_W <= 1).mean()), current_share_krW_le1=float((cur.kr_W <= 1).mean()))
print(out['cn_realloc'])
json.dump(out, open(os.path.join(W, 's3_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
