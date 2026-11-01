# -*- coding: utf-8 -*-
"""C08-T3 전향 채점 1회 (PREREG.txt sha256 4c4e9a29… 고정 후).
입력: frozen_scores.parquet(sha 대조), persist_index.csv(sha 대조), 등록원부(c08lib 캐시 = sqlite mode=ro 적재).
출력: prospective_result.json, calibration.csv, industry_rank_check.csv, subgroup.csv, cache/labels.parquet"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
import c08lib as C
t0 = time.time()
W = C.W
FZ_SHA = '21d4743d5015e74af6b7f8ecd2fa471cbd2392dee0e0d811a57926da77f48453'
PI_SHA = '8c0c3a8bc24b33fbcf0e69e99cfa40fd88dc9ffe5ff1c781b20f1c86523b2b1a'
L = C.L0; WEND = np.datetime64('2025-10-31'); SNAP = np.datetime64('2026-08-15')
PAYEND = np.datetime64('2026-08-14'); S2END = np.datetime64('2026-01-13')
OFF = dict(skill=0.2458, lo=0.1332, hi=0.3196)
B = 1000
DRY = os.environ.get('C08_DRYRUN') == '1'   # 기계 점검: 실제 결과 대신 모의 레이블, B=20, 출력은 cache/dryrun
if DRY:
    B = 20; W = os.path.join(C.CACHE, 'dryrun'); os.makedirs(W, exist_ok=True)
assert C.sha(os.path.join(C.W, 'frozen_scores.parquet')) == FZ_SHA
assert C.sha(os.path.join(C.W, 'persist_index.csv')) == PI_SHA
fz = pd.read_parquet(os.path.join(C.W, 'frozen_scores.parquet'))
D = C.load(); pt, pay, rt = D['pt'], D['pay'], D['rt']
pid = fz.pid.values
if DRY:
    rs = np.random.default_rng(99)
    y = (rs.random(len(pid)) < fz.p_haz_win.values).astype(int); y_S1 = y.copy(); y_S2 = (rs.random(len(pid)) < 0.04).astype(int)
    LB = dict(dryrun=True, pos=int(y.sum()))
else:
    # ---------------- 레이블
    lap = C.lapse_events(D)
    rec = rt[(rt.name == '회복등록') & (rt.cd <= SNAP)]
    recmax = rec.groupby('pid').cd.max()
    ew = lap[(lap.E > L) & (lap.E <= WEND) & (lap.cd <= SNAP) & lap.pid.isin(pid)].copy()
    ew['rec_cd'] = ew.pid.map(recmax)
    ew['restored'] = ew.rec_cd.notna() & (ew.rec_cd >= ew.cd)
    y_S1 = np.isin(pid, ew.pid.unique()).astype(int)
    y = np.isin(pid, ew[~ew.restored].pid.unique()).astype(int)
    # 진단: 위험집합 특허의 E<=L 소멸(있으면 안 됨), 창 안 다중 사건
    e_le_L = lap[(lap.E <= L) & (lap.cd > L) & lap.pid.isin(pid)]
    LB = dict(n=len(pid), pos=int(y.sum()), rate=float(y.mean()), pos_S1=int(y_S1.sum()), restored_events=int(ew.restored.sum()),
              window_events_rows=len(ew), patents_multi_event=int((ew.groupby('pid').size() > 1).sum()),
              riskset_lapse_E_le_L_cd_gt_L=int(len(e_le_L)), pos_outside_due_win=int((y & (fz.due_win.values == 0)).sum()),
              pos_in_due_win=int((y & (fz.due_win.values == 1)).sum()))
    # S2 납부기록 레이블
    pth, _ = C.paid_thru(pt, pay, PAYEND)
    p2 = pth[pid]; ex = pt.exp.values[pid]
    ml = rt[rt.name.isin(C.MALSO) & (rt.cd > L) & (rt.cd <= S2END)].pid.unique()
    y_S2 = ((p2 > L) & (p2 <= S2END) & (p2 < ex) & ~np.isin(pid, ml)).astype(int)
    LB['pos_S2'] = int(y_S2.sum()); LB['S2_window'] = f'({L}, {S2END}] pay<= {PAYEND}'
    pd.DataFrame({'pid': pid, 'y': y, 'y_S1': y_S1, 'y_S2': y_S2}).to_parquet(os.path.join(C.CACHE, 'labels.parquet'), index=False)


# ---------------- 가중 AUC (동점 처리, 부트스트랩 가중)
class WA:
    def __init__(s, yy, sc):
        s.y = yy.astype(float); u, s.inv = np.unique(sc, return_inverse=True); s.G = len(u)
    def __call__(s, w=None):
        w = np.ones(len(s.y)) if w is None else w
        pg = np.bincount(s.inv, w * s.y, s.G); ng = np.bincount(s.inv, w * (1 - s.y), s.G)
        P, N = pg.sum(), ng.sum()
        cn = np.cumsum(ng) - ng
        return float((pg * (cn + 0.5 * ng)).sum() / (P * N)) if P > 0 and N > 0 else np.nan


sk = lambda a, b: (a - b) / (1 - b)
acd = fz.acd.values
cl = np.where(acd == 'NA', 'P' + pid.astype(str), acd)
codes, uniq = pd.factorize(cl); K = len(uniq)
SC = {'p': fz.p.values, 'p_base': fz.p_base.values, 'p_strong': fz.p_strong.values, 'p_haz_win': fz.p_haz_win.values,
      'p_cls': fz.p_cls.values, 'p_haz': fz.p_haz.values, 'p_logit': fz.p_logit.values, 'p_hgb': fz.p_hgb.values}
# 부분집합 정의 (사전등록)
atype = fz.atype.values
SUBS = {'all': np.ones(len(pid), bool), 'due_win': fz.due_win.values == 1, 'excl_big10': fz.big10.values == 0, 'big10_only': fz.big10.values == 1,
        'excl_big2': fz.big2.values == 0}
for a in ['corp', 'fcorp', 'indiv', 'univ', 'inst']:
    SUBS['atype_' + a] = atype == a
LABS = {'y': y, 'y_S1': y_S1, 'y_S2': y_S2}
# 평가 조합: (부분집합, 레이블, 점수쌍)
JOBS = [('all', 'y', 'p', 'p_base'), ('all', 'y', 'p', 'p_strong'), ('due_win', 'y', 'p', 'p_base'), ('due_win', 'y', 'p', 'p_strong'),
        ('excl_big10', 'y', 'p', 'p_base'), ('big10_only', 'y', 'p', 'p_base'), ('excl_big2', 'y', 'p', 'p_base'),
        ('all', 'y', 'p_haz_win', 'p_base'), ('due_win', 'y', 'p_haz_win', 'p_base'),
        ('all', 'y_S1', 'p', 'p_base'), ('all', 'y_S2', 'p', 'p_base'), ('all', 'y_S2', 'p', 'p_strong')] + \
       [('atype_' + a, 'y', 'p', 'p_base') for a in ['corp', 'fcorp', 'indiv', 'univ', 'inst']]
prep = {}
for sname, lname, a, b in JOBS:
    m = SUBS[sname]
    for s_ in (a, b):
        k = (sname, lname, s_)
        if k not in prep: prep[k] = WA(LABS[lname][m], SC[s_][m])
pt_est = {}
for sname, lname, a, b in JOBS:
    A1, A0 = prep[(sname, lname, a)](), prep[(sname, lname, b)]()
    pt_est[(sname, lname, a, b)] = (A1, A0, sk(A1, A0))
rng = np.random.default_rng(0)
bs = {j: [] for j in JOBS}
calib_bs = {'ratio_pw': [], 'ratio_hazwin': [], 'ratio_pw_due': []}
pw = 1 - (1 - fz.p.values) ** (122 / 1096)
for it in range(B):
    wc = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[codes]
    cache = {}
    for j in JOBS:
        sname, lname, a, b = j; m = SUBS[sname]
        for s_ in (a, b):
            k = (sname, lname, s_)
            if k not in cache: cache[k] = prep[k](wc[m])
        A1, A0 = cache[(sname, lname, a)], cache[(sname, lname, b)]
        bs[j].append((A1, A0, sk(A1, A0)))
    yw = (wc * y).sum()
    calib_bs['ratio_pw'].append((wc * pw).sum() / yw); calib_bs['ratio_hazwin'].append((wc * fz.p_haz_win.values).sum() / yw)
    dm = SUBS['due_win']; calib_bs['ratio_pw_due'].append((wc[dm] * pw[dm]).sum() / (wc[dm] * y[dm]).sum())
ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
rows = []
for j in JOBS:
    sname, lname, a, b = j; arr = np.array(bs[j]); m = SUBS[sname]
    A1, A0, S = pt_est[j]
    rows.append(dict(subset=sname, label=lname, model=a, baseline=b, n=int(m.sum()), pos=int(LABS[lname][m].sum()),
                     rate=float(LABS[lname][m].mean()), auc_model=A1, auc_base=A0, skill=S,
                     auc_model_ci=ci(arr[:, 0]), auc_base_ci=ci(arr[:, 1]), skill_ci=ci(arr[:, 2])))
sub = pd.DataFrame(rows)
sub2 = sub.copy()
for c in ['auc_model_ci', 'auc_base_ci', 'skill_ci']:
    sub2[c + '_lo'] = sub2[c].str[0]; sub2[c + '_hi'] = sub2[c].str[1]; del sub2[c]
sub2.to_csv(os.path.join(W, 'subgroup.csv'), index=False, encoding='utf-8-sig')
G = lambda s, l, a, b: next(r for r in rows if (r['subset'], r['label'], r['model'], r['baseline']) == (s, l, a, b))
main = G('all', 'y', 'p', 'p_base'); due = G('due_win', 'y', 'p', 'p_base')
# ---------------- 판정
lo, hi = main['skill_ci']; S = main['skill']
if lo > 0 and OFF['lo'] <= S <= OFF['hi']: verdict = '전향 확인'
elif lo > 0 and S > OFF['hi']: verdict = '전향 확인(공식 CI 위)'
elif lo > 0 and S < OFF['lo']: verdict = '약화'
else: verdict = '실패'
qual = []
if due['skill_ci'][0] <= 0: qual.append('납부 시점 정보 의존 — 같은 납부기한 특허끼리는 구별 안 됨')
elif due['skill'] < OFF['lo']: qual.append('같은 납부기한 안 순위는 약화')
st = G('all', 'y', 'p', 'p_strong')
st_v = '우위' if st['skill_ci'][0] > 0 else ('열위' if st['skill_ci'][1] < 0 else '구별 안 됨')
stD = G('due_win', 'y', 'p', 'p_strong')
stD_v = '우위' if stD['skill_ci'][0] > 0 else ('열위' if stD['skill_ci'][1] < 0 else '구별 안 됨')
e10, b10 = G('excl_big10', 'y', 'p', 'p_base'), G('big10_only', 'y', 'p', 'p_base')
big_rule = bool((e10['skill'] - S > 0) and (b10['skill'] < e10['skill']))
# ---------------- 포착·정밀도
def capture(sc, yy, q):
    k = int(round(q * len(yy))); top = np.argsort(-sc, kind='stable')[:k]
    return float(yy[top].sum() / yy.sum()), float(yy[top].mean())
cap = {}
for nm in ['p', 'p_base', 'p_strong', 'p_haz_win']:
    for q in (0.1, 0.2):
        c_, pr_ = capture(SC[nm], y, q); cap[f'{nm}_top{int(q*100)}'] = dict(capture=c_, precision=pr_)
    md = SUBS['due_win']; c_, pr_ = capture(SC[nm][md], y[md], 0.1); cap[f'{nm}_due_top10'] = dict(capture=c_, precision=pr_)
# ---------------- 보정
cal = dict(n_pos=int(y.sum()), sum_pw=float(pw.sum()), ratio_pw=float(pw.sum() / y.sum()), ratio_pw_ci=ci(calib_bs['ratio_pw']),
           sum_hazwin=float(fz.p_haz_win.sum()), ratio_hazwin=float(fz.p_haz_win.sum() / y.sum()), ratio_hazwin_ci=ci(calib_bs['ratio_hazwin']),
           ratio_pw_due=float(pw[SUBS['due_win']].sum() / y[SUBS['due_win']].sum()), ratio_pw_due_ci=ci(calib_bs['ratio_pw_due']),
           mean_p_3y=float(fz.p.mean()), blind='아님(집계 기저율 사전 노출, exposure_disclosure.txt)')
dec = pd.DataFrame({'p': fz.p.values, 'pw': pw, 'phw': fz.p_haz_win.values, 'pb': fz.p_base.values, 'y': y, 'due': fz.due_win.values})
dec['decile'] = 10 - pd.qcut(rankdata(-dec.p.values, method='ordinal'), 10, labels=False)
cr = dec.groupby('decile').agg(n=('y', 'size'), mean_p3y=('p', 'mean'), mean_pw=('pw', 'mean'), mean_p_haz_win=('phw', 'mean'),
                               realized=('y', 'mean'), due_share=('due', 'mean'), events=('y', 'sum')).reset_index()
cr['decile'] = 11 - cr['decile']; cr = cr.sort_values('decile')   # 1 = 위험 최상위
cr['ratio_pw'] = cr.mean_pw / cr.realized; cr['ratio_hazwin'] = cr.mean_p_haz_win / cr.realized
dd = dec[dec.due == 1].copy(); dd['dq'] = pd.qcut(rankdata(-dd.p.values, method='ordinal'), 10, labels=False) + 1
cr2 = dd.groupby('dq').agg(n=('y', 'size'), mean_p3y=('p', 'mean'), mean_pw=('pw', 'mean'), mean_p_haz_win=('phw', 'mean'),
                           realized=('y', 'mean'), events=('y', 'sum')).reset_index().rename(columns={'dq': 'decile'})
cr['set'] = 'all'; cr2['set'] = 'due_win'; cr2['due_share'] = 1.0
cr2['ratio_pw'] = cr2.mean_pw / cr2.realized; cr2['ratio_hazwin'] = cr2.mean_p_haz_win / cr2.realized
pd.concat([cr, cr2], ignore_index=True).to_csv(os.path.join(W, 'calibration.csv'), index=False, encoding='utf-8-sig')
# ---------------- 산업 지수
pi = pd.read_csv(os.path.join(C.W, 'persist_index.csv'), dtype={'ksic': str}).set_index('ksic')
ks = fz.ksic.astype(str).values; kc, kn = pd.factorize(ks); NK = len(kn)
persist = pi.persist_rate.reindex(kn).values; persist_due = pi.persist_rate_due.reindex(kn).values
dmask = SUBS['due_win']
def kstats(w):
    sw = np.bincount(kc, w, NK)
    mp = np.bincount(kc, w * fz.p.values, NK) / sw; mb = np.bincount(kc, w * fz.p_base.values, NK) / sw
    yr = np.bincount(kc, w * y, NK) / sw
    wd = w * dmask; yrd = np.bincount(kc, wd * y, NK) / np.bincount(kc, wd, NK)
    o = dict(rho_model=spearmanr(mp, yr)[0], rho_persist=spearmanr(persist, yr)[0], rho_base=spearmanr(mb, yr)[0],
             rho_model_due=spearmanr(mp, yrd)[0], rho_persist_due=spearmanr(persist_due, yrd)[0])
    for nm, mk in (('excl_big2', fz.big2.values == 0), ('excl_big10', fz.big10.values == 0)):
        w2 = w * mk; s2 = np.bincount(kc, w2, NK)
        o['rho_model_' + nm] = spearmanr(np.bincount(kc, w2 * fz.p.values, NK) / s2, np.bincount(kc, w2 * y, NK) / s2)[0]
        o['rho_persist_' + nm] = spearmanr(persist, np.bincount(kc, w2 * y, NK) / s2)[0]
    o['delta'] = o['rho_model'] - o['rho_persist']; o['delta_due'] = o['rho_model_due'] - o['rho_persist_due']
    o['delta_excl_big2'] = o['rho_model_excl_big2'] - o['rho_persist_excl_big2']
    o['delta_excl_big10'] = o['rho_model_excl_big10'] - o['rho_persist_excl_big10']
    return o, mp, yr, yrd, mb
ind_pt, mp, yr, yrd, mb = kstats(np.ones(len(pid)))
rng1 = np.random.default_rng(1); ib = {k: [] for k in ind_pt}
for it in range(B):
    wc = np.bincount(rng1.integers(0, K, K), minlength=K).astype(float)[codes]
    o = kstats(wc)[0]
    for k in ib: ib[k].append(o[k])
ind = {k: dict(est=float(v), ci=ci(np.array(ib[k]))) for k, v in ind_pt.items()}
dlo, dhi = ind['delta']['ci']
ind_v = '지수 우위' if dlo > 0 else ('지수 열위' if dhi < 0 else '지속 지수와 구별 안 됨')
ind_pos = ind['rho_model']['ci'][0] > 0
ik = pd.DataFrame({'ksic': kn, 'n': np.bincount(kc, minlength=NK), 'events': np.bincount(kc, y.astype(float), NK).astype(int),
                   'index_pred_risk_3y': mp, 'baseline_risk_3y': mb, 'realized_rate_window': yr, 'persist_rate_prev_window': persist,
                   'due_share': np.bincount(kc, dmask.astype(float), NK) / np.bincount(kc, minlength=NK), 'realized_rate_due': yrd,
                   'persist_rate_due': persist_due})
ik['rank_index'] = ik.index_pred_risk_3y.rank(ascending=False); ik['rank_realized'] = ik.realized_rate_window.rank(ascending=False)
ik['rank_persist'] = ik.persist_rate_prev_window.rank(ascending=False); ik['rank_realized_due'] = ik.realized_rate_due.rank(ascending=False)
nm_ = pd.read_csv(os.path.join(C.R2, 'future_forecast_r2.csv'), dtype={'ksic': str}).drop_duplicates('ksic').set_index('ksic').industry
ik.insert(1, 'industry', ik.ksic.map(nm_))
ik.sort_values('rank_index').to_csv(os.path.join(W, 'industry_rank_check.csv'), index=False, encoding='utf-8-sig')
# 대형 2위 출원인 진단(정보)
b2 = fz.big2.values == 1
big2 = dict(n=int(b2.sum()), mean_p=float(fz.p.values[b2].mean()), mean_pw=float(pw[b2].mean()), realized=float(y[b2].mean()),
            due_share=float(fz.due_win.values[b2].mean()), realized_due=float(y[b2 & dmask].mean()) if (b2 & dmask).any() else None,
            mean_pw_due=float(pw[b2 & dmask].mean()) if (b2 & dmask).any() else None)
top10 = []
for a in pd.Series(acd[fz.big10.values == 1]).value_counts().index:
    m = acd == a
    top10.append(dict(acd=a, n=int(m.sum()), mean_p=float(fz.p.values[m].mean()), mean_pw=float(pw[m].mean()), realized=float(y[m].mean()),
                      events=int(y[m].sum()), auc_model=WA(y[m], fz.p.values[m])() if 0 < y[m].sum() < m.sum() else None,
                      auc_base=WA(y[m], fz.p_base.values[m])() if 0 < y[m].sum() < m.sum() else None))
res = dict(task='C08-T3', when=time.strftime('%Y-%m-%d %H:%M:%S'), prereg_sha256='4c4e9a292ebf3a636484aff1f82f41481f9f3774f277ba96b043f5d865ac970e',
           frozen_scores_sha256=FZ_SHA, official_ref=OFF, window=f'E in ({L}, {WEND}], cd<= {SNAP}', labels=LB,
           main=main, verdict=verdict, qualifiers=qual, verdict_full=verdict + (' (' + '; '.join(qual) + ')' if qual else ''),
           due_win=due, strong=dict(all=st, all_verdict=st_v, due_win=stD, due_verdict=stD_v),
           big_applicant=dict(excl_big10=e10, big10_only=b10, excl_big2=G('excl_big2', 'y', 'p', 'p_base'), rule_limit_persists=big_rule,
                              big2=big2, top10=top10),
           capture=cap, calibration=cal, industry=dict(stats=ind, verdict=ind_v, rho_model_ci_lo_gt_0=bool(ind_pos), n_ksic=NK),
           components_auc={k: WA(y, v)() for k, v in SC.items()},
           bootstrap=dict(B=B, seed_main=0, seed_industry=1, clusters=int(K), cluster='1순위 출원인코드(없으면 특허)'),
           secs=round(time.time() - t0, 1), peak_gb=C.peak_gb())
C.jdump(res, os.path.join(W, 'prospective_result.json'))
print(json.dumps({k: res[k] for k in ['labels', 'main', 'verdict_full', 'due_win', 'strong', 'calibration', 'industry', 'components_auc', 'secs', 'peak_gb']},
                 ensure_ascii=False, indent=1, default=str))
