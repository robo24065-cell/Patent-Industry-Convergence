# C18-T4 s8: power_map.csv(설계×후보×학습기, verify 독립 추정량 열 포함), p0_results.csv 조립. 캐시·verify 결과만 읽는다.
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib18 import *
from scipy.stats import norm
P = pd.read_csv(C + r'\s4_power_rows.csv', encoding='utf-8-sig')
V = {}
for fn in glob.glob(W + r'\verify_0\v1_K*.json'):
    J = json.load(open(fn)); t = J['tag'].split('_')
    V[(t[0], t[2] if len(t) > 2 else 'xgb')] = J['lock']
for col in ['SE_lock_IF', 'power_IF', 'SE_lock_JK', 'power_JK', 'relDiff_boot_vs_IF', 'verify_note']:
    P[col] = np.nan if col != 'verify_note' else ''
for i, r in P.iterrows():
    lk = V.get((r.cand, r.learner))
    if lk is None:
        continue
    tgt = r.target
    if r.design in ('D0', 'D2'):
        se_if = lk['D0_IF']['SE_lock']; se_jk = lk['D0_JK']['SE_lock']
    elif r.design == 'D1a':
        se_if = lk['D1a_IF']['SE_lock']; se_jk = np.nan
    else:
        continue
    P.at[i, 'SE_lock_IF'] = se_if; P.at[i, 'power_IF'] = norm.cdf(tgt / se_if - 1.959964)
    if se_jk == se_jk:
        P.at[i, 'SE_lock_JK'] = se_jk; P.at[i, 'power_JK'] = norm.cdf(tgt / se_jk - 1.959964)
    rd = abs(r.SE_lock - se_if) / se_if
    P.at[i, 'relDiff_boot_vs_IF'] = rd
    g_if = norm.cdf(tgt / se_if - 1.959964) >= 0.8
    note = []
    if rd > 0.25:
        note.append('경고: 부트스트랩 게이트 SE와 IF 상대차 >25%')
    if g_if != bool(r.G1) and not str(r.design).startswith('D3b'):
        note.append(f'IF 추정량이면 G1 {"통과" if g_if else "실패"}(판정과 다름; 판정은 사전 등록 SE_gate)')
    P.at[i, 'verify_note'] = '; '.join(note)
P['classification'] = ''
for i, r in P.iterrows():
    if str(r.design).startswith('D3b'):
        P.at[i, 'classification'] = '투영(게이트 아님)'; continue
    if r.G1:
        P.at[i, 'classification'] = '검정력 충분'
    elif r.MDE_inf > r.target:
        P.at[i, 'classification'] = '(가) 구조적: MDE_inf > 목표'
    else:
        P.at[i, 'classification'] = '(나) 규모 조건부: 잠금 ×%.1f 필요' % r['need_lock_multiple_for_0.8']
P.to_csv(W + r'\power_map.csv', index=False, encoding='utf-8-sig', float_format='%.6g')
out(P[['cand', 'learner', 'design', 'target', 'obs_delta_dev', 'SE_lock', 'power_lock', 'MDE_lock', 'MDE_inf', 'need_lock_multiple_for_0.8', 'SE_lock_IF', 'power_IF', 'power_JK', 'relDiff_boot_vs_IF', 'classification', 'verify_note']].to_string())

# p0_results.csv
B = json.load(open(C + r'\s6_p0b.json'))['rows']
pb = {(r['cand'], r['learner'], r['stat'][:6]): r for r in B}
rows = []
for K in ['K1', 'K2', 'K3', 'K4']:
    for D in ['D0', 'D1a', 'D1b', 'D2']:
        g = P[(P.cand == K) & (P.design == D)]
        gx = g[g.learner == 'xgb'].iloc[0]; gl = g[g.learner == 'lr']
        g1x = bool(gx.G1); g1l = bool(gl.iloc[0].G1) if len(gl) else None
        st = 'strati' if D == 'D1b' else 'pooled'
        bx = pb.get((K, 'xgb', st)); bl = pb.get((K, 'lr', st))
        r = {'cand': K, 'design': D, 'target': gx.target, 'G1_power_xgb': round(gx.power_lock, 4), 'G1_power_lr': (round(gl.iloc[0].power_lock, 4) if len(gl) else 'LR 미학습(XGB G1 실패로 불필요)'),
             'G1': 'pass' if (g1x and (g1l is None or g1l)) and (g1l is not None) else ('fail' if not g1x or g1l is False else 'pending'),
             'P0a_E_lock_pos': round(E_LOCK[K], 1), 'P0a': 'pass' if E_LOCK[K] >= 300 else 'fail'}
        if r['G1'] == 'pass':
            r['P0b_xgb'] = 'pass' if bx['P0b_pass'] else 'fail'
            r['P0b_lr'] = 'pass' if bl['P0b_pass'] else 'fail'
            r['P0b_detail'] = (f"XGB 최선공개 {bx['best_public']} 평균Δ " + ', '.join('%s %+.5f' % (m, bx[f'd_{m}_mean']) for m in PUB_ARMS[K]) + f" (>0 원점 몫 {bx['frac_origins_pos_vs_best']:.2f}) | "
                               f"LR 최선공개 {bl['best_public']} 평균Δ " + ', '.join('%s %+.5f' % (m, bl[f'd_{m}_mean']) for m in PUB_ARMS[K]) + f" (>0 원점 몫 {bl['frac_origins_pos_vs_best']:.2f})")
            r['P0c'] = '미실행(첫 실패 규칙)' if not (bx['P0b_pass'] and bl['P0b_pass']) else 'run'
            r['P0d'] = '미실행(첫 실패 규칙)'
            r['first_fail'] = 'P0(b) LR' if not bl['P0b_pass'] else ('P0(b) XGB' if not bx['P0b_pass'] else '')
        else:
            r.update({'P0b_xgb': '미실행', 'P0b_lr': '미실행', 'P0b_detail': '', 'P0c': '미실행', 'P0d': '미실행', 'first_fail': 'G1(검정력)'})
        r['verdict'] = '기각(P0 중단, 잠금 채점 없음)'
        rows.append(r)
pd.DataFrame(rows).to_csv(W + r'\p0_results.csv', index=False, encoding='utf-8-sig')
out(pd.DataFrame(rows)[['cand', 'design', 'G1_power_xgb', 'G1_power_lr', 'G1', 'P0a', 'P0b_xgb', 'P0b_lr', 'first_fail']].to_string())
