# -*- coding: utf-8 -*-
"""R5 레드팀(T4) 분석: 앙상블 재합성, 위약(층 내 레이블 섞기), 강화 기준선, 대형 출원인 제외, 하위집단, 상위 k, 보정, 시점 누수 점검.
동결 예측 npy·ev.parquet·frame_cls.parquet(사본)은 읽기만 한다. 모형은 고치지 않는다."""
import os, sys, json, sqlite3
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
HERE = os.path.dirname(os.path.abspath(__file__)); TR = os.path.dirname(HERE)
FD = os.path.join(TR, 'R2', 'runs_final', 'F22')
FS = json.load(open(os.path.join(TR, 'R2', 'final_spec_r2.json'), encoding='utf-8')); SPEC = FS['spec']
A = lambda y, p: float(roc_auc_score(y, p))
sk = lambda a, b: (a - b) / (1 - b)
R = {}
ev = pd.read_parquet(os.path.join(FD, 'ev.parquet'))
y = ev.y.values.astype(int)
P = {k: np.load(os.path.join(FD, 'pred', f)) for k, f in [('cls', 'cls_99f270b3cd.npy'), ('haz', 'haz_0568d4297b.npy'), ('logit', 'logit_3008b8b195.npy'), ('hgb', 'hgb_b4bd0fa46a.npy'), ('base', 'base_46526c56d8.npy')]}
lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
ws = SPEC['ensemble']['weights']
z = sum(w * lg(P[k]) for k, w in ws.items()) / sum(ws.values())
pc = 1 / (1 + np.exp(-z)); pb = P['base']
lab = pd.read_parquet(os.path.join(HERE, 'cache', 'labels_indep.parquet'))
assert (lab.set_index('pid').y_indep.reindex(ev.pid).values == y).all()
R['repro'] = dict(n=len(y), pos=int(y.sum()), auc_model=A(y, pc), auc_base=A(y, pb), skill=sk(A(y, pc), A(y, pb)),
                  frozen=json.load(open(os.path.join(TR, 'R2', 'final_result.json'), encoding='utf-8'))['skill_auc'],
                  comps={k: A(y, v) for k, v in P.items()}, labels_from_raw_match=True)
print('repro', R['repro'], flush=True)


def boot(pm, pb_, B=300, seed=3):
    cl = np.where(ev.acd.values == 'NA', 'P' + ev.pid.astype(str).values, ev.acd.values)
    codes, uniq = pd.factorize(cl); K = len(uniq); rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        w = np.bincount(rng.integers(0, K, K), minlength=K).astype(float)[codes]
        a = roc_auc_score(y, pm, sample_weight=w); b = roc_auc_score(y, pb_, sample_weight=w); out.append(sk(a, b))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


# ---------------- 위약: 층(KSIC × 경과연수 구간 × 출원인 포트폴리오 규모 3분위)
ageb = pd.cut(ev.age, [-1, 3, 6, 9, 12, 99], labels=False).astype(str)
port = ev.port_n.fillna(0)
sizeb = pd.qcut(port.rank(method='first'), 3, labels=False).astype(str)
strata = ev.ksic.astype(str) + '_' + ageb + '_' + sizeb
idx = pd.Series(np.arange(len(y))).groupby(strata.values).apply(np.array)
pl = []
for s in range(100):
    rng = np.random.default_rng(300 + s); yp = y.copy()
    for ix in idx:
        yp[ix] = rng.permutation(y[ix])
    pl.append((sk(A(yp, pc), A(yp, pb)), A(yp, pc), A(yp, pb)))
pl = np.array(pl)
full = np.array([sk(A(yp, pc), A(yp, pb)) for yp in (np.random.default_rng(700 + s).permutation(y) for s in range(100))])
R['placebo'] = dict(strata='KSIC × 경과연수 5구간 × 출원인 포트폴리오 규모 3분위', n_strata=int(strata.nunique()), seeds=100,
                    skill_mean=float(pl[:, 0].mean()), p2_5=float(np.percentile(pl[:, 0], 2.5)), p97_5=float(np.percentile(pl[:, 0], 97.5)),
                    auc_model_mean=float(pl[:, 1].mean()), auc_base_mean=float(pl[:, 2].mean()), p_ge_obs=float((pl[:, 0] >= R['repro']['skill']).mean()),
                    full_mean=float(full.mean()), full_p2_5=float(np.percentile(full, 2.5)), full_p97_5=float(np.percentile(full, 97.5)))
print('placebo', R['placebo'], flush=True)

# ---------------- 강화 기준선 (학습 프레임 frame_cls 사본, 원점 시점 피처만)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
fr = pd.read_parquet(os.path.join(HERE, 'F22', 'frame_cls.parquet'), columns=['y', 'age', 'claims', 'rgst_year', 'atype', 'ksic', 'next_annl', 'paid_ahead', 'n_due_win', 'tier_in_win', 'a_lapse_rate', 'a_lapse_rate2', 'n_late', 'yrs_since_pay', 'L'])
fr = fr.sample(n=min(600_000, len(fr)), random_state=0)
def strong(num):
    ct = ColumnTransformer([('n', make_pipeline(SimpleImputer(strategy='median'), StandardScaler()), num),
                            ('c', OneHotEncoder(handle_unknown='ignore', min_frequency=20), ['atype', 'ksic'])])
    X = fr[num + ['atype', 'ksic']].copy(); X['claims'] = np.log1p(X.claims.astype(float))
    for c in ['atype', 'ksic']: X[c] = X[c].astype(str)
    m = make_pipeline(ct, LogisticRegression(C=1.0, max_iter=2000)).fit(X, fr.y.values)
    E = ev[num + ['atype', 'ksic']].copy(); E['claims'] = np.log1p(E.claims.astype(float))
    for c in ['atype', 'ksic']: E[c] = E[c].astype(str)
    return m.predict_proba(E)[:, 1]
B0 = ['age', 'claims', 'rgst_year']
SB = {'기준선+납부일정(next_annl·paid_ahead·n_due_win·tier_in_win)': strong(B0 + ['next_annl', 'paid_ahead', 'n_due_win', 'tier_in_win']),
      '기준선+출원인 과거 포기율(a_lapse_rate·a_lapse_rate2)': strong(B0 + ['a_lapse_rate', 'a_lapse_rate2']),
      '기준선+납부일정+출원인 포기율+납부행동(n_late·yrs_since_pay)': strong(B0 + ['next_annl', 'paid_ahead', 'n_due_win', 'tier_in_win', 'a_lapse_rate', 'a_lapse_rate2', 'n_late', 'yrs_since_pay'])}
R['strong_baseline'] = {'공식 기준선(로지스틱)': dict(auc_base=A(y, pb), skill=sk(A(y, pc), A(y, pb)), ci95=json.load(open(os.path.join(TR, 'R2', 'final_result.json'), encoding='utf-8'))['bootstrap_95ci']['skill'])}
for k, v in SB.items():
    R['strong_baseline'][k] = dict(auc_base=A(y, v), skill=sk(A(y, pc), A(y, v)), ci95=boot(pc, v))
print('strong', R['strong_baseline'], flush=True)
np.save(os.path.join(HERE, 'cache', 'strong_full.npy'), list(SB.values())[-1])


# ---------------- 대형 출원인 제외 / 하위집단
def sub(mask, label):
    yy = y[mask]
    if yy.sum() < 20 or yy.sum() == len(yy):
        return dict(label=label, n=int(mask.sum()), note='양성 부족')
    a1, a0 = A(yy, pc[mask]), A(yy, pb[mask])
    return dict(label=label, n=int(mask.sum()), rate=float(yy.mean()), auc_model=a1, auc_base=a0, skill=sk(a1, a0))
cnt = ev[ev.acd != 'NA'].acd.value_counts()
ex = [sub(~ev.acd.isin(cnt.index[:k]).values, f'특허 수 상위 {k} 출원인 제외') for k in (1, 2, 5, 10, 20)]
ex.append(sub(~ev.acd.isin([cnt.index[1]]).values, '2위 출원인(LCD·OLED)만 제외'))
ex.append(sub(ev.acd.isin(cnt.index[:10]).values, '상위 10 출원인만'))
R['exclude_large'] = ex
R['top_share'] = dict(top1=float(cnt.iloc[0] / len(ev)), top10=float(cnt.iloc[:10].sum() / len(ev)))
subs = [sub((ev.atype == a).values, f'출원인유형 {a}') for a in ['corp', 'fcorp', 'indiv', 'univ', 'inst']]
for lo, hi, lb in [(0, 3, '경과 0–3년'), (4, 6, '4–6년'), (7, 9, '7–9년'), (10, 12, '10–12년'), (13, 99, '13년+')]:
    subs.append(sub(ev.age.between(lo, hi + 0.999).values, lb))
k2 = ev.ksic.astype(str).str[:2]
for g in sorted(k2.unique()):
    if (k2 == g).sum() >= 10000:
        subs.append(sub((k2 == g).values, f'KSIC {g}xxx'))
for q, lb in [(0, '포트폴리오 하위 1/3'), (1, '중간 1/3'), (2, '상위 1/3')]:
    subs.append(sub((sizeb == str(q)).values, lb))
R['subgroups'] = subs
# ---------------- 상위 k
n = len(y)
def tk(p, k): return int(y[np.argsort(-p)[:k]].sum())
R['topk'] = [dict(k=k, model_hits=tk(pc, k), base_hits=tk(pb, k), strong_hits=tk(list(SB.values())[-1], k), model_prec=tk(pc, k) / k, base_prec=tk(pb, k) / k,
                  model_recall=tk(pc, k) / y.sum(), lift=tk(pc, k) / k / y.mean()) for k in [100, 1000, int(0.01 * n), int(0.05 * n), int(0.10 * n), int(0.20 * n)]]
R['pr_auc'] = dict(model=float(average_precision_score(y, pc)), base=float(average_precision_score(y, pb)), base_rate=float(y.mean()))
# ---------------- 보정
def calib(p):
    d = pd.DataFrame({'p': p, 'y': y}); d['dec'] = pd.qcut(d.p.rank(method='first'), 10, labels=False)
    t = d.groupby('dec').agg(mean_pred=('p', 'mean'), obs=('y', 'mean')).reset_index()
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(lg(p).reshape(-1, 1), y)
    return dict(brier=float(brier_score_loss(y, p)), mean_pred=float(p.mean()), obs=float(y.mean()), slope=float(lr.coef_[0][0]), intercept=float(lr.intercept_[0]), deciles=t.round(4).to_dict('records'))
R['calibration'] = {'model': calib(pc), 'base': calib(pb)}
# ---------------- 시점 누수 점검: 사후 스냅샷 서지(청구항·존속만료)
DB = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\수집파이프라인\registry_full.sqlite"
c = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
rt = pd.read_sql("select rgst_no, rgstCsDate, rgstCsName from right where rgstCsName like '%일부말소%' or rgstCsName like '%연장%' or rgstCsName like '%정정%'", c)
c.close()
lab2 = lab.set_index('pid').rgst_no.reindex(ev.pid).values
rt['d'] = pd.to_datetime(rt.rgstCsDate.astype(str), format='%Y%m%d', errors='coerce')
post = rt[rt.d > '2022-01-01']
kinds = {}
for nm, pat in [('일부말소(청구항 감소)', '일부말소'), ('존속기간 연장', '연장'), ('정정', '정정')]:
    s = set(post[post.rgstCsName.str.contains(pat)].rgst_no)
    m = np.isin(lab2, list(s))
    kinds[nm] = dict(n_test=int(m.sum()), share=float(m.mean()), rate=float(y[m].mean()) if m.sum() else None)
R['leak_post_snapshot_events'] = kinds
json.dump(R, open(os.path.join(HERE, 'v3_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
for s in R['exclude_large'] + R['subgroups']: print(s)
for s in R['topk']: print(s)
print(R['pr_auc'], R['top_share'], R['leak_post_snapshot_events'])
print({k: {kk: vv for kk, vv in v.items() if kk != 'deciles'} for k, v in R['calibration'].items()})
print(R['calibration']['model']['deciles'])
