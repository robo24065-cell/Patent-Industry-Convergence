# -*- coding: utf-8 -*-
"""s05: S5 [보조, 재학습 없음] 평가 모집단의 시점 의존 — 동결 build_panel(cap=None)의 '특허 ≥3건' 모집단은 원점 후 등록분(2020–25 등록, 출원 ≤2019)과
 2026 목록 범위밖 행으로 센다. 원점 연말까지 등록된 행만으로도 ≥3건인 법인 쌍만 남겨 공식 예측·공식 6기준선을 다시 채점 → cache/s5_out.json"""
import sys, json, time
from c14lib import *
from sklearn.metrics import roc_auc_score
from r3 import mets, skill
from final_run_r3 import blend
T0 = time.time(); OUT = {}
D = lib.load(); ins, oos = D['ins'], D['oos']
ip = ins[ins.ay <= 2019].drop_duplicates(['applicant_cd', 'rgst_no']); op = oos[oos.ay <= 2019].drop_duplicates(['applicant_cd', 'rgst_no'])
cnt = pd.concat([ip.applicant_cd, op.applicant_cd]).value_counts(); pop = set(cnt[cnt >= 3].index)
ce = '20191231'
cs = pd.concat([ip[ip.rgst_date <= ce].applicant_cd, op[op.rgst_date <= ce].applicant_cd]).value_counts(); strict = set(cs[cs >= 3].index)
ci = pd.concat([ip.applicant_cd]).value_counts(); ins_only = set(ci[ci >= 3].index)
LAB = B/'분석/검증루프_20261012/C10/C10-T1/cache/final_labels.parquet'
o = pd.read_parquet(R3/'runs/FROZEN_final/o2019.parquet'); L = pd.read_parquet(LAB)
assert (o.firm.values == L.firm.values).all()
keep = ~L.prior_any.values; o = o[keep].reset_index(drop=True); L = L[keep].reset_index(drop=True)
bs = pd.read_parquet(R3/'cache_final/o2019/baseline_scores.parquet').loc[keep].reset_index(drop=True)
firms = set(o.firm.unique())
OUT['pop_firms_build_panel_rule'] = len(pop); OUT['eval_firms_with_pairs'] = len(firms); OUT['eval_firms_subset_of_pop'] = len(firms - pop) == 0
OUT['eval_firms_strict_ge3'] = len(firms & strict); OUT['eval_firms_need_post_origin'] = len(firms - strict)
s = blend(o); y = L.y_u.values; nc = L.newc.values.astype(bool)
def sc(m_):
    if nc[m_].sum() == 0:      # 신규진입자(범위안 0건) 없는 부분집합: 전체 AUC만
        a = roc_auc_score(y[m_], s[m_]); b = max(roc_auc_score(y[m_], bs[k].values[m_]) for k in bs.columns)
        return dict(n=int(m_.sum()), pos=int(y[m_].sum()), auc=a, base=b, skill=skill(a, b), skill_new=None)
    mm = mets(y[m_], s[m_], nc[m_]); bm = {k: mets(y[m_], bs[k].values[m_], nc[m_]) for k in bs.columns}
    b = max(v['auc_all'] for v in bm.values()); bn = max(v['auc_new'] for v in bm.values())
    return dict(n=int(m_.sum()), pos=int(y[m_].sum()), auc=mm['auc_all'], base=b, skill=skill(mm['auc_all'], b), skill_new=skill(mm['auc_new'], bn))
full = np.ones(len(o), bool); sub = o.firm.isin(strict).values; out_ = ~sub
OUT['full'] = sc(full); OUT['strict_subset'] = sc(sub)
OUT['excluded_pairs'] = int(out_.sum()); OUT['excluded_pos'] = int(y[out_].sum()); OUT['excluded_pos_rate'] = float(y[out_].mean()) if out_.any() else None
OUT['d_skill_strict_minus_full'] = OUT['strict_subset']['skill'] - OUT['full']['skill']
# 범위안 ≥3만으로(범위밖 2026 목록 행을 전혀 안 쓴 모집단) — 참고
sub2 = o.firm.isin(ins_only).values; OUT['ins_only_subset'] = sc(sub2); OUT['d_skill_insonly_minus_full'] = OUT['ins_only_subset']['skill'] - OUT['full']['skill']
json.dump(OUT, open(HERE/'cache/s5_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(OUT, ensure_ascii=False, indent=1, default=str)); print('done', round(time.time() - T0))
