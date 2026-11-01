# s5: P3(매핑·연결 편향) + 데이터 편향 점검. 입력: s2 산출(개발 원점만, 2025 이전 owner만).
import sys, json
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C02\C02-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from lib_cd import *
R = pd.read_parquet(C + r'\dev_rows.parquet'); P = pd.read_parquet(C + r'\dev_pos.parquet'); CE = pd.read_parquet(C + r'\dev_cells.parquet')
st = json.load(open(C + r'\dev_stats.json', encoding='utf-8'))
out, bias = {}, []


def smd(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    s = np.sqrt((x.var(ddof=1) + y.var(ddof=1)) / 2)
    return float((x.mean() - y.mean()) / s) if s > 0 else 0.0


def duncan(a, b):
    return round(float(0.5 * (a.value_counts(normalize=True).reindex(KSIC50, fill_value=0) - b.value_counts(normalize=True).reindex(KSIC50, fill_value=0)).abs().sum()), 4)


# --- P3 매핑률
R['mapped'] = R['map'] == 'unique'
out['rows_once'] = len(R)
out['map_rate_all'] = round(float(R.mapped.mean()), 4)
out['map_share'] = R['map'].value_counts(normalize=True).round(4).to_dict()
out['map_rate_corp'] = round(float(R[R.corp].mapped.mean()), 4)
out['map_rate_corp_noninst'] = round(float(R[R.corp & ~R.inst].mapped.mean()), 4)
out['map_rate_inst'] = round(float(R[R.inst].mapped.mean()), 4)
out['map_rate_noncorp_noninst'] = round(float(R[~R.corp & ~R.inst].mapped.mean()), 4)
out['map_rate_by_year'] = R.groupby(R.d.str[:4]).mapped.mean().round(4).to_dict()
out['mapped_to_prefix1_share_of_mapped'] = round(float((R[R.mapped].cd.str[0] == '1').mean()), 4)
m, u = R[R.mapped], R[~R.mapped]
sm = {'year': smd(m.d.str[:4].astype(int), u.d.str[:4].astype(int))}
for cs in ['양도', '일부양도', '지분양도', '신탁계약에 의한 양도']:
    sm['cs_' + cs] = smd(m.cs == cs, u.cs == cs)
for c in ['bulk', 'corp', 'inst', 'group']:
    sm[c] = smd(m[c].astype(float), u[c].astype(float))
sm['natl_KR_or_null'] = smd(m.natl.isna() | (m.natl == '대한민국'), u.natl.isna() | (u.natl == '대한민국'))
sm['seller_inst'] = smd(m.seller_type == 'inst', u.seller_type == 'inst')
out['smd_mapped_vs_unmapped'] = {k: round(v, 3) for k, v in sm.items()}
out['ksic_duncan_mapped_vs_unmapped'] = duncan(m.k, u.k)
mc, uc = R[R.corp & R.mapped], R[R.corp & ~R.mapped]
out['smd_corp_only'] = {'year': round(smd(mc.d.str[:4].astype(int), uc.d.str[:4].astype(int)), 3),
                        'bulk': round(smd(mc.bulk.astype(float), uc.bulk.astype(float)), 3),
                        'group': round(smd(mc.group.astype(float), uc.group.astype(float)), 3),
                        'cs_양도': round(smd(mc.cs == '양도', uc.cs == '양도'), 3)}
out['ksic_duncan_corp_only'] = duncan(mc.k, uc.k)
# --- NPS 연결/비연결
CE['units_n'] = CE.units - CE.units_l; CE['pos_n'] = CE.pos - CE.pos_l
out['cells_total'] = len(CE)
out['cells_both_units'] = int(((CE.units_l > 0) & (CE.units_n > 0)).sum())
out['cells_both_units_share'] = round(out['cells_both_units'] / len(CE), 4)
cp = CE[CE.pos > 0]
out['cells_with_pos'] = len(cp)
out['cells_both_pos_share_of_pos_cells'] = round(float(((cp.pos_l > 0) & (cp.pos_n > 0)).mean()), 4)
rl = CE.pos_l.sum() / CE.units_l.sum(); rn = CE.pos_n.sum() / CE.units_n.sum()
out['pos_rate_linked'] = float(f'{rl:.3e}'); out['pos_rate_unlinked'] = float(f'{rn:.3e}')
out['pos_rate_ratio_linked_over_unlinked'] = round(float(rl / rn), 3)
out['link_rate_units'] = round(float(CE.units_l.sum() / CE.units.sum()), 4)
out['link_rate_pos'] = round(float(CE.pos_l.sum() / CE.pos.sum()), 4)
byt = CE.groupby('t0')[['units', 'units_l', 'pos', 'pos_l']].sum()
out['pos_rate_ratio_by_t0'] = ((byt.pos_l / byt.units_l) / ((byt.pos - byt.pos_l) / (byt.units - byt.units_l))).round(3).to_dict()
# --- P3 판정
out['P3_decision_cells_both_ge_0.95'] = bool(out['cells_both_units_share'] >= 0.95)
warn = []
if out['map_rate_corp'] < 0.5: warn.append('법인 표지 매핑률<0.5')
for k, v in out['smd_mapped_vs_unmapped'].items():
    if abs(v) > 0.25: warn.append(f'|SMD| {k}={v}')
if not (0.5 <= out['pos_rate_ratio_linked_over_unlinked'] <= 2): warn.append('양성률 비 범위 밖')
out['P3_warnings'] = warn


# --- 데이터 편향(양성 기준)
def add(metric, value, note=''):
    bias.append(dict(metric=metric, value=value, note=note))


N = len(P)
add('dev_pos_total_sum_over_origins', N, '원점 합(19개월 창이 7개월 겹침 -> 중복 포함)')
add('dev_pos_unique_events_cd_k_rgst_d', int(P.drop_duplicates(['cd', 'k', 'rgst_no', 'd']).shape[0]), '겹침 제거 고유 사건')
add('dev_pos_unique_cd_k', int(P.drop_duplicates(['cd', 'k']).shape[0]), '')
add('dev_pos_unique_buyers', int(P.cd.nunique()), '')
top_buy = P.cd.value_counts()
add('buyer_top1_share_of_pos', round(float(top_buy.iloc[0] / N), 4), '한 매입자 집중')
add('buyer_top10_share_of_pos', round(float(top_buy.iloc[:10].sum() / N), 4), '')
add('bulk_any_share_of_pos', round(float(P.bulk_any.mean()), 4), '같은 날 같은 매입자 >=10건 포함 사건')
add('group_internal_share_of_pos', round(float(P.group.mean()), 4), '판매자·매입자 공통 접두 규칙')
for k, v in P.seller_type.value_counts(normalize=True).round(4).items():
    add(f'seller_type_share_{k}', v, '')
for k, v in P.cs.value_counts(normalize=True).round(4).items():
    add(f'reason_share_{k}', v, '첫 사건 사유')
add('pos_n_rows_ge2_share', round(float((P.n_rows >= 2).mean()), 4), '같은 (f,k) 창 안 2건 이상')
q = P.buyer_docs.quantile([.1, .25, .5, .75, .9]).round(1).to_dict()
add('buyer_docs_quantiles_pos', json.dumps(q), 't0 말 제공범위 문서 수')
pe = P[P.t0 >= 2015]
add('buyer_nps_emp_quantiles_pos_2015plus', json.dumps(pe.n_emp.quantile([.1, .25, .5, .75, .9]).round(1).to_dict()), f'결측 {pe.n_emp.isna().mean():.3f}')
add('buyer_nps_linked_share_pos', round(float(P.nps.mean()), 4), '')
add('rg_in_window_share_pos', round(float(P.rg_in_window.mean()), 4), '창 안 등록 특허 매입 비중(잠금 창은 재고 동결로 낮아짐)')
kp = P.k.value_counts(normalize=True)
add('ksic_top5_share_pos', round(float(kp.iloc[:5].sum()), 4), json.dumps(kp.iloc[:5].round(3).to_dict()))
add('ksic_n_with_pos', int(P.k.nunique()), '50 중')
units_k = CE.groupby('k').units.sum(); uk = units_k / units_k.sum()
add('ksic_top5_share_risk_units', round(float(uk.sort_values(ascending=False).iloc[:5].sum()), 4), '위험집합 기준(비교)')
for s in st:
    add(f'pos_rate_t0_{s["t0"]}', float(f'{s["pos"] / s["risk_units"]:.3e}'), f'pos={s["pos"]}, units={s["risk_units"]}, eligible={s["eligible_firms"]}')
rates = np.array([s['pos'] / s['risk_units'] for s in st]); xs = np.array([s['t0'] for s in st])
add('pos_rate_log_slope_per_year', round(float(np.polyfit(xs, np.log(rates), 1)[0]), 4), '기저율 이동(위험집합 증가로 하락)')
add('pos_count_by_t0', json.dumps({s['t0']: s['pos'] for s in st}), '')
for s in st:
    add(f'map_unique_share_window_t0_{s["t0"]}', round(s['win_map_unique'] / s['win_acc_rows'], 4), '창 전체 인정 사유 행 기준')
B = pd.DataFrame(bias)
B.to_csv(W + r'\bias_checks.csv', index=False, encoding='utf-8-sig')
json.dump(out, open(C + r'\p3_result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str)); print(B.to_string())
