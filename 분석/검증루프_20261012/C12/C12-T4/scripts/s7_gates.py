# 게이트 집계 → gates.json, candidates.csv (P0 실패이므로 P2–P5는 '판정 밖')
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib12 import *

p01 = json.load(open(C + r'\p0p1.json'))
em = json.load(open(C + r'\eval_main.json')); es = json.load(open(C + r'\eval_sub.json'))
ck = json.load(open(C + r'\checks.json'))
info = {f'{cc}_{Y}': json.load(open(C + rf'\info_{cc}_{Y}.json')) for cc in CCS for Y in ALL_Y}
# P5 귀무 모으기
nulls = {}
real = None
for f in glob.glob(C + r'\p5null_*.json'):
    j = json.load(open(f))
    if 'G_real_recomputed' in j:
        real = j
    for n in j.get('nulls', []):
        nulls[n['b']] = n['G']
G = np.array([nulls[b] for b in sorted(nulls)])
g_real = real['G_real_recomputed'][0]
q95 = float(np.percentile(G, 95)) if len(G) else None
p5a = {'G_real': g_real, 'G_real_per_origin': real['G_real_recomputed'][1], 'ctx_recompute_maxabs': real['ctx_recompute_maxabs'],
       'n_null': int(len(G)), 'null_mean': float(G.mean()), 'null_sd': float(G.std(ddof=1)), 'null_q95': q95, 'null_max': float(G.max()),
       'z': float((g_real - G.mean()) / G.std(ddof=1)), 'p_emp': float((1 + (G >= g_real).sum()) / (1 + len(G))),
       'pass_offjudgment': bool(g_real > q95), 'gap_real_minus_nullmean': float(g_real - G.mean()),
       'share_of_ctx_gain_reproduced_by_null': float(G.mean() / g_real)}
pos_lt_ad = sum(info[k]['pos_docs_pub_lt_ad'] for k in info if int(k[-4:]) in EVAL_Y)
pos_docs = sum(info[k]['pos_docs'] for k in info if int(k[-4:]) in EVAL_Y)
gates = {
    'task': 'C12-T4', 'verdict': 'P0에서 기각(잠금 채점 없음, 원고 미반영)',
    'P0': {'pass': p01['P0']['pass'], 'a': p01['P0a'], 'b': p01['P0b'], 'judged': True},
    'P1': {'pass': p01['P1']['pass'], 'judged': False, 'note': 'P0 실패 뒤 판정 밖', 'table': p01['P1']['table']},
    'P2': {'judged': False, **em['P2']},
    'P3': {'judged': False, 'a_asserts': '스크립트 assert 통과(피처 공개일<t0, 학습 결과창<평가 t0, 캐시<2025-01-01)',
           'b_pos_docs': pos_docs, 'b_pub_lt_appl': pos_lt_ad, 'c_lock_observability': '실행 안 함(P0 실패로 잠금 구간 집계를 더 보지 않음)'},
    'P4': {'judged': False, **em['P4']},
    'P5': {'judged': False, 'a': p5a, 'b_top10_share_dev': ck['dev_all']['top10_firm_share'], 'b_top5_sc': ck['dev_all']['top5_sc'],
           'b_top5_sc_pos_share': ck['dev_all']['top5_sc_pos_share'], 'b_top5_sc_risk_share': ck['dev_all']['top5_sc_risk_share'],
           'b_cc_share': ck['dev_all']['cc_pos_share'], 'c_vk_rate': ck['dev_all']['vk_rate']},
    'refute_offjudgment': {'i': em['ref_i'], 'ii': em['ref_ii'], 'iii': {'refuted': es['ref_iii']['refuted'], 'ex_top10': {k: es['ex_top10'][k] for k in ['D', 'ci_firm_D', 'd_v']},
                                                                     'ex_top1pct': {k: es['ex_top1pct'][k] for k in ['D', 'ci_firm_D', 'd_v']}}},
    'sensitivity_offjudgment': {k: {kk: es[k][kk] for kk in ['D', 'ci_firm_D', 'd_v', 'prov_vs_pub', 'n_pos']} for k in ['ex_missc', 'CN', 'EP']},
    'auc_by_model': em['all']['auc'],
}
json.dump(gates, open(W + r'\gates.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
rows = [dict(candidate='C12-T4 해외(CN·EP) 활성 출원인 새 IPC 서브클래스 첫 공개 × 제공 KSIC 맥락', gate_failed='P0(b)',
             P0a_exp_lock_pos=round(p01['P0a']['exp_pos']), P0b_power_two=round(p01['P0b']['power']['two'], 4), P0b_power_firm=round(p01['P0b']['power']['firm'], 4),
             P0b_SE_two=round(p01['P0b']['SE']['two'], 5),
             P2_dskill_vs_PI=round(em['P2']['mean'], 4), P4_D=round(em['P4']['D'], 5), P4_ci_firm=str([round(x, 5) for x in em['P4']['ci_firm']]),
             P4_ci_two=str([round(x, 5) for x in em['P4']['ci_two']]), P4_pos_origins=em['P4']['n_pos'],
             P5a_G_real=round(g_real, 5), P5a_null_q95=round(q95, 5), P5a_n_null=len(G),
             ref_i=em['ref_i']['refuted'], ref_ii=em['ref_ii']['refuted'], ref_iii=es['ref_iii']['refuted'],
             lock_scored=False, manuscript='원고 미반영')]
pd.DataFrame(rows).to_csv(W + r'\candidates.csv', index=False, encoding='utf-8-sig')
out(json.dumps(p5a, ensure_ascii=False))
