# -*- coding: utf-8 -*-
"""C06-T3 s2: N7 — RL 경로(권리목록 사업자번호) 8곳 레이블 재구성, 공식 단위 재채점, 원고 숫자 대조, 보안과제 민감도.
모형 재학습 없음. 예측은 공식 동결(final_test_pred_r5) + VERIFY_R6 재현 예측(preds_orig/alias) + C05-T2 층 정의(U_N7)."""
import sqlite3, os, sys, io, json
import pandas as pd, numpy as np, pyarrow.parquet as pq
from scipy.stats import rankdata
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = ROOT + r"\분석\모델경쟁_20260927"; SH = MC + r"\_shared\ext"; N7 = MC + r"\N7_특허기업_정부RnD첫수행"
OUT = ROOT + r"\분석\검증루프_20261012\C06\C06-T3"
o = {}
def auc(y, p):  # Mann-Whitney 순위식 (sklearn 미사용)
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    r = rankdata(p); n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
sk = lambda a, b: (a - b) / (1 - b)
def topk(y, p, fr):
    y = np.asarray(y); p = np.asarray(p); k = max(1, int(round(fr * len(y))))
    idx = np.argsort(-p, kind='mergesort')[:k]
    return k, float(y[idx].mean()), float(y[idx].sum() / y.sum())
def boot(y, pm, pb, cl, n=1000, seed=0):
    rng = np.random.default_rng(seed); y = np.asarray(y); pm = np.asarray(pm); pb = np.asarray(pb)
    u, inv = np.unique(pd.Series(cl).astype(str).values, return_inverse=True)
    groups = pd.Series(np.arange(len(y))).groupby(inv).apply(lambda s: s.values).values
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(u), len(u)); i = np.concatenate([groups[j] for j in pick])
        if y[i].min() == y[i].max(): continue
        out.append(sk(auc(y[i], pm[i]), auc(y[i], pb[i])))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]
# --- 재구성 레이블 + RL 8곳
m = pd.read_parquet(OUT + r"\cache\n7_merge.parquet")
lo = m[m._merge == 'left_only'].copy()
o['official_only_units'] = lo[['cd', 'bizno', 'y_off', 'y1_off']].astype(str).values.tolist()
con = sqlite3.connect("file:" + ROOT + r"\수집파이프라인\business_rights.sqlite?mode=ro", uri=True)
q = "select applicant_cd, business_no from right_list where applicant_cd in (%s)" % ",".join("'%s'" % c for c in lo.cd)
rl = pd.read_sql(q, con); con.close()
rl['bz'] = rl.business_no.fillna('').astype(str).str.replace(r'\D', '', regex=True)
rlu = rl[rl.bz.str.len() == 10].groupby('applicant_cd').bz.agg(lambda s: sorted(set(s)))
o['rl_bizno_by_code'] = {k: v for k, v in rlu.items()}
# 이 8곳이 등록원부 첫 자리 1·등록 ≤2022·국세청·폐업·NTIS 조건을 만족하는지 직접 확인
st = pd.read_parquet(SH + r"\r3nts_status.parquet", columns=['bizno', 'registered', 'end_dt']).drop_duplicates('bizno').set_index('bizno')
nt = pq.read_table(SH + r"\ntis_project_raw.parquet", columns=['year', 'corp_rgst_no']).to_pandas()
nt['bz'] = nt.corp_rgst_no.fillna('').astype(str).str.replace(r'\D', '', regex=True)
firsty = nt[(nt.bz.str.len() == 10) & (nt.year >= 2010)].groupby('bz').year.min()
chk = []
for _, r in lo.iterrows():
    bzs = rlu.get(r.cd, [])
    bz = bzs[0] if len(bzs) == 1 else None
    fy = firsty.get(bz, np.nan) if bz else np.nan
    reg = bool(st.registered.get(bz, False)) if bz else False
    ed = st.end_dt.get(bz, pd.NaT) if bz else pd.NaT
    yr = int(fy in (2023, 2024)) if bz else None
    chk.append(dict(cd=r.cd, bz_off=str(r.bizno), rl_bz=bzs, bz_match=(bz == str(r.bizno)), nts_registered=reg, end_dt=str(ed),
                    first_ntis=None if pd.isna(fy) else int(fy), y_rb=yr, y_off=int(r.y_off), y1_rb=int(fy == 2023) if bz else None, y1_off=int(r.y1_off)))
o['rl_units_check'] = chk
lab = m[m._merge != 'right_only'][['cd', 'bizno', 'y_off', 'y1_off', 'y_rb', 'y1_rb']].copy()
fix = {c['cd']: (c['y_rb'], c['y1_rb']) for c in chk}
for c, (a, b) in fix.items():
    lab.loc[lab.cd == c, 'y_rb'] = a; lab.loc[lab.cd == c, 'y1_rb'] = b
lab['route'] = np.where(lab.cd.isin(fix.keys()), 'RL(권리목록 사업자번호)', 'CA')
o['label_agree_all12286'] = float((lab.y_rb == lab.y_off).mean()); o['label1_agree_all12286'] = float((lab.y1_rb == lab.y1_off).mean())
lab.to_csv(OUT + r"\n7_label_rebuild.csv", index=False, encoding='utf-8-sig')
# --- 재채점 (공식 동결 예측)
fr = pd.read_parquet(N7 + r"\final_test_pred_r5.parquet"); fr['cd'] = fr.applicantCd.astype(str)
d = fr.merge(lab[['cd', 'y_rb', 'y1_rb']], on='cd', how='left')
assert d.y_rb.notna().all()
y = d.y_rb.astype(int).values
J = json.load(open(N7 + r"\final_result_r5.json", encoding='utf-8'))['results'][0]
R = {}
R['n'] = len(d); R['pos'] = int(y.sum()); R['base_rate'] = float(y.mean()); R['n_clusters'] = int(d.bizno.nunique())
for nm, col in (('champ', 'p_champ'), ('M0', 'p_M0'), ('SB', 'p_SB')):
    R['auc_' + nm] = auc(y, d[col])
R['skill_vs_M0'] = sk(R['auc_champ'], R['auc_M0']); R['skill_vs_SB'] = sk(R['auc_champ'], R['auc_SB'])
for nm, col in (('champ', 'p_champ'), ('M0', 'p_M0'), ('SB', 'p_SB')):
    for fr_, tag in ((0.01, 'top1pct'), (0.05, 'top5pct'), (0.10, 'top10pct')):
        k, pr, rc = topk(y, d[col].values, fr_); R[f'{nm}_{tag}'] = dict(k=k, precision=pr, recall=rc)
# 공식 JSON 대비 차
cmp = {'n': (R['n'], J['n']), 'pos': (R['pos'], J['pos']), 'base_rate': (R['base_rate'], J['base_rate']), 'n_clusters': (R['n_clusters'], J['n_clusters']),
       'auc_champ': (R['auc_champ'], J['auc']['champ']), 'auc_M0': (R['auc_M0'], J['auc']['M0']), 'auc_SB': (R['auc_SB'], J['auc']['SB']),
       'skill_vs_M0': (R['skill_vs_M0'], J['skill_vs_M0']), 'skill_vs_SB': (R['skill_vs_SB'], J['skill_vs_SB'])}
for nm in ('champ', 'M0', 'SB'):
    for tag in ('top1pct', 'top5pct', 'top10pct'):
        cmp[f'{nm}_{tag}_prec'] = (R[f'{nm}_{tag}']['precision'], J['topk'][nm][tag]['precision'])
        cmp[f'{nm}_{tag}_rec'] = (R[f'{nm}_{tag}']['recall'], J['topk'][nm][tag]['recall'])
R['cmp_vs_json'] = {k: dict(rebuild=a, json=b, absdiff=abs(float(a) - float(b))) for k, (a, b) in cmp.items()}
R['max_absdiff'] = max(v['absdiff'] for v in R['cmp_vs_json'].values())
# CI: (i) 공식 절차 재현(군집=사업자번호, seed=t0=2022, 2023) (ii) 독립 시드
R['ci_skill_M0_seed2022'] = boot(y, d.p_champ, d.p_M0, d.bizno, 1000, 2022)
R['ci_skill_SB_seed2023'] = boot(y, d.p_champ, d.p_SB, d.bizno, 1000, 2023)
R['ci_skill_M0_seed777'] = boot(y, d.p_champ, d.p_M0, d.bizno, 1000, 777)
R['ci_json'] = dict(M0=J['skill_vs_M0_ci95'], SB=J['skill_vs_SB_ci95'])
o['rescore_official'] = R
# --- VERIFY_R6 예측(공식 재현 패널 orig / NPS 별칭 alias)과 원고 숫자
V = {}
for pan in ('orig', 'alias'):
    p = pd.read_parquet(N7 + rf"\VERIFY_R6\cache\preds_{pan}.parquet"); p['cd'] = p.applicantCd.astype(str)
    p = p.merge(lab[['cd', 'y_rb', 'y1_rb']], on='cd', how='left'); yy = p.y_rb.astype(int).values
    v = dict(label_eq_file=float((p.y_rb == p.y).mean()))
    for c in ('champ', 'M0', 'SB', 'pilot', 'M0_ens', 'SB_ens', 'EXT_ens', 'EXT_xgb'):
        v['auc_' + c] = auc(yy, p['p_' + c])
    for b in ('M0', 'SB', 'M0_ens', 'EXT_ens'):
        v['skill_' + b] = sk(v['auc_champ'], v['auc_' + b])
    v['ci_M0_seed2022'] = boot(yy, p.p_champ, p.p_M0, p.bizno, 1000, 2022)
    v['ci_SB_seed2023'] = boot(yy, p.p_champ, p.p_SB, p.bizno, 1000, 2023)
    v['ci_EXT_ens_seed2025'] = boot(yy, p.p_champ, p.p_EXT_ens, p.bizno, 1000, 2025)
    v['ci_EXT_ens_seed777'] = boot(yy, p.p_champ, p.p_EXT_ens, p.bizno, 1000, 777)
    y1 = p.y1_rb.astype(int).values
    v['y1_pos'] = int(y1.sum()); v['y1_auc_champ'] = auc(y1, p.p_champ_y1); v['y1_auc_M0'] = auc(y1, p.p_M0_y1); v['y1_skill'] = sk(v['y1_auc_champ'], v['y1_auc_M0'])
    k, pr, rc = topk(yy, p.p_champ.values, 0.01); v['top1_champ'] = [k, pr, int(round(pr * k))]
    k, pr, rc = topk(yy, p.p_M0.values, 0.01); v['top1_M0'] = [k, pr]
    k, pr, rc = topk(yy, p.p_EXT_ens.values, 0.01); v['top1_EXT'] = [k, pr, int(round(pr * k))]
    k, pr, rc = topk(yy, p.p_champ.values, 0.10); v['top10_champ'] = [k, pr, rc]
    nn = p.nps_present.values == 0
    v['nonnps'] = dict(n=int(nn.sum()), pos=int(yy[nn].sum()), auc_champ=auc(yy[nn], p.p_champ[nn]), auc_M0=auc(yy[nn], p.p_M0[nn]))
    if pan == 'orig':
        v['p_champ_eq_frozen'] = float(np.abs(p.set_index('cd').p_champ.reindex(fr.cd).values - fr.p_champ.values).max())
    V[pan] = v
o['verify_r6_preds'] = V
# --- C05-T2 층(S2: 기준일 공개 자료로 사업자번호 확인)
U = pd.read_parquet(ROOT + r"\분석\검증루프_20261012\C05\C05-T2\cache\U_N7.parquet", columns=['applicantCd', 'S2'])
U['cd'] = U.applicantCd.astype(str)
ds = d.merge(U[['cd', 'S2']], on='cd', how='left')
o['S2_missing'] = int(ds.S2.isna().sum())
for sv in (1, 0):
    s = ds[ds.S2 == sv]; ys = s.y_rb.astype(int).values
    a1, a0 = auc(ys, s.p_champ), auc(ys, s.p_M0)
    o[f'S2_{sv}'] = dict(n=len(s), share=len(s) / len(ds), pos=int(ys.sum()), auc=a1, auc_M0=a0, skill=sk(a1, a0),
                         ci_seed777=boot(ys, s.p_champ, s.p_M0, s.bizno, 1000, 777))
# --- 보안과제: 마스킹 규모(원천) + 민감도
ntf = pq.read_table(SH + r"\ntis_project_raw.parquet", columns=['year', 'corp_rgst_no', 'secret_flag', 'perform_grp', 'research_agency', 'gov_funds']).to_pandas()
sec = ntf[ntf.secret_flag]
o['secret'] = dict(rows_by_year={int(k): int(v) for k, v in sec.year.value_counts().sort_index().items() if k >= 2015},
                   corp_rgst_nonnull=int(sec.corp_rgst_no.notna().sum()), research_agency_nonnull=int(sec.research_agency.notna().sum()),
                   perform_grp=sec.perform_grp.value_counts().to_dict(), share_rows_2023_24=float(ntf[ntf.year.isin([2023, 2024])].secret_flag.mean()))
# 레이블 처리: 보안과제로만 첫 진입한 기업은 2023–24에 사업자번호가 안 나오므로 y=0(음성)으로 남는다. 2010–22 보안과제만 수행한 기업은 모집단에 잔류.
rng = np.random.default_rng(11)
neg = np.where(y == 0)[0]; base = sk(auc(y, d.p_champ), auc(y, d.p_M0))
sens = {}
pc = d.p_champ.values; rk = rankdata(pc) / len(pc)
for r_ in (0.049, 0.074, 0.235):
    add = int(round(R['pos'] * r_ / (1 - r_)))  # 관측 양성이 전체의 (1-r)이라 가정
    res = {'random': [], 'score_prop': [], 'score_low': []}
    for it in range(200):
        for mode in res:
            if mode == 'random': pick = rng.choice(neg, add, replace=False)
            elif mode == 'score_prop':
                w = rk[neg] / rk[neg].sum(); pick = rng.choice(neg, add, replace=False, p=w)
            else:
                w = (1 - rk[neg]) / (1 - rk[neg]).sum(); pick = rng.choice(neg, add, replace=False, p=w)
            y2 = y.copy(); y2[pick] = 1
            res[mode].append(sk(auc(y2, d.p_champ), auc(y2, d.p_M0)) - base)
    sens[str(r_)] = dict(add=add, **{k: [float(np.mean(v)), float(np.percentile(v, 5)), float(np.percentile(v, 95))] for k, v in res.items()})
o['secret_sensitivity_dskill'] = sens
json.dump(o, open(OUT + r"\n7_rescore.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str)[:12000])
