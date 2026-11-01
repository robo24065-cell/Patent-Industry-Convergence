# s6: 잠금창 1회 채점. verify_0 판정 통과 파일이 있어야만 실행된다. 잠금 레이블을 한 번 계산해 곧바로 채점한다(재실행 금지: 결과 파일이 있으면 중단).
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from libC import *
from libB import Prep
T0 = time.time()
RES = os.path.join(W, 'final_result_C05T4.json')
assert not os.path.exists(RES), '이미 채점함 — 재채점 금지'
vd = json.load(open(os.path.join(W, 'verify_0', 'verify_verdict.json'), encoding='utf-8'))
assert vd['pass'] is True, '독립 검토 미통과 — 동결 보류'
spec = json.load(open(os.path.join(W, 'final_spec_C05T4.json'), encoding='utf-8'))
fz_path = os.path.join(W, 'frozen_scores.parquet')
assert sha256(fz_path) == open(os.path.join(W, 'frozen_scores.sha256'), encoding='utf-8').read().split()[0]
fz = pd.read_parquet(fz_path)
champ = spec['champion']
ref = REF[LOCK]; end = add_months(ref, 6)
assert (ref, end) == ('20241231', '20250630')
ca = load_ca(); nm2b6, _ = name_map(ca); ca_names = set(ca.nname)
docs = load_docs(); docs['b6'] = docs.nname.map(nm2b6)
first_vis_link = docs[docs.b6.notna() & docs.vis.notna()].groupby('nname').vis.min()
first_vis_all = docs[docs.vis.notna()].groupby('nname').vis.min()
# ---- 잠금 레이블(1회)
w = docs[docs.pub.notna() & (docs.pub > ref) & (docs.pub <= end)]
wl = w[w.b6.notna()].copy(); wl = wl[wl.nname.map(first_vis_link) > ref]; wl['key'] = wl.b6 + '|' + wl.nname
pos = set(wl.key) & set(fz.key)
fz['y'] = fz.key.isin(pos).astype(np.int8)
fz[['key', 'y']].to_parquet(os.path.join(C, 'lock_labels.parquet'), index=False)
print('lock positives', int(fz.y.sum()), 'of', len(fz), flush=True)
# ---- 사전 점검(보고만): 창 안 첫 공개 국내 법인 표지 상호의 CA 존재율(C04 s6 정의: 첫 보인 날짜는 전체 문서 기준)
wk = w[w.kor_corp]; wk = wk[wk.nname.map(first_vis_all) > ref]
nmset = pd.Index(wk.nname.unique())
ca_rate = float(nmset.isin(ca_names).mean())
dev_rates = {}
for s0 in DEV:
    r0 = REF[s0]; e0 = add_months(r0, 6)
    ww = docs[docs.pub.notna() & (docs.pub > r0) & (docs.pub <= e0) & docs.kor_corp]
    ww = ww[ww.nname.map(first_vis_all) > r0]
    dev_rates[s0] = round(float(pd.Index(ww.nname.unique()).isin(ca_names).mean()), 4)
pre = dict(lock_first_time_kor_corp_names=int(len(nmset)), lock_ca_presence=round(ca_rate, 4), dev_ca_presence=dev_rates,
           dev_mean=round(float(np.mean(list(dev_rates.values()))), 4), warn_lt_0_90=bool(ca_rate < 0.90),
           lock_window_docs=int(len(w)), lock_window_kor_corp_docs=int(w.kor_corp.sum()))
print('precheck', pre, flush=True)
# ---- 채점
y = fz.y.values
cols = {m: f'score_{m}' for m in ['M0', 'M1', 'M1_pub_TXT', 'M1_pub_IPC'] if f'score_{m}' in fz}
cols['champion'] = 'score_champion'
pt = {}
preps = {m: Prep(y, fz[c].values) for m, c in cols.items()}
one = np.ones(len(fz))
for m, p in preps.items():
    a = p.auc(one); ap, top = p.ap_top(one)
    pt[m] = dict(auc=a, pr_auc=ap, top1000=top)
rng = np.random.default_rng(2025); NB = 1000
BS = {m: {'auc': [], 'pr_auc': [], 'top1000': []} for m in preps}
n = len(fz)  # 잠금은 단일 원점이라 기업키 = 행
for b in range(NB):
    wb = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
    for m, p in preps.items():
        a = p.auc(wb); ap, top = p.ap_top(wb)
        BS[m]['auc'].append(a); BS[m]['pr_auc'].append(ap); BS[m]['top1000'].append(top)
ci = lambda x: [round(float(np.quantile(x, 0.025)), 5), round(float(np.quantile(x, 0.975)), 5)]
sk = lambda a, a0: (np.asarray(a) - np.asarray(a0)) / (1 - np.asarray(a0))
out = dict(champion=champ, n=int(n), pos=int(y.sum()), base_rate=round(float(y.mean()), 6), precheck=pre, seed=2025, n_boot=NB, models={})
for m in preps:
    out['models'][m] = dict(auc=round(pt[m]['auc'], 5), auc_ci=ci(BS[m]['auc']), pr_auc=round(pt[m]['pr_auc'], 6), pr_auc_ci=ci(BS[m]['pr_auc']),
                            top1000=round(pt[m]['top1000'], 5), top1000_ci=ci(BS[m]['top1000']),
                            skill_vs_M0=round(float(sk(pt[m]['auc'], pt['M0']['auc'])), 5), skill_vs_M0_ci=ci(sk(BS[m]['auc'], BS['M0']['auc'])))
for pm in ['M1_pub_TXT', 'M1_pub_IPC']:
    if pm in preps:
        out[f'dskill_M1_vs_{pm}'] = dict(point=round(float(sk(pt['M1']['auc'], pt[pm]['auc'])), 5), ci=ci(sk(BS['M1']['auc'], BS[pm]['auc'])))
c = out['models']['champion']
out['verdict'] = dict(predictable=bool(c['auc_ci'][0] > 0.5),
                      M1_added_effect_none=bool(out['models']['M1']['skill_vs_M0_ci'][0] <= 0 <= out['models']['M1']['skill_vs_M0_ci'][1]))
# ---- N9 문서 겹침(보고만)
fl = pd.read_parquet(os.path.join(EXT, 'ntis_patent_flag.parquet'), columns=['appl_no', 'gov_rnd', 'min_project_year', 'max_project_year'])
pd_ = wl[wl.key.isin(pos)][['appl_no', 'key', 'ay']].merge(fl.drop_duplicates('appl_no'), on='appl_no', how='left')
pd_['g'] = pd_.gov_rnd.fillna(False).astype(bool)
n9w = pd_.g & pd_.ay.isin([2022, 2023]) & (pd_.min_project_year <= 2022) & (pd_.max_project_year >= 2022)
out['n9_overlap'] = dict(pos_docs=int(len(pd_)), gov_share=round(float(pd_.g.mean()), 4) if len(pd_) else None,
                         firm_any_gov_share=round(float(pd_.groupby('key').g.any().mean()), 4) if len(pd_) else None,
                         docs_in_n9_final_window_approx=int(n9w.sum()))
pd_[['appl_no', 'key', 'ay', 'g']].to_parquet(os.path.join(C, 'lock_pos_docs.parquet'), index=False)
out['elapsed_s'] = round(time.time() - T0)
json.dump(out, open(RES, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
