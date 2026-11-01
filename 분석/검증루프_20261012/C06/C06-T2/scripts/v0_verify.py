# C06-T2 verify_0: (1) docx XML 직접 파싱으로 diff 재확인 (2) 그림 3개 독립 재계산(fig02 AUC·skill, fig09 상위 k, fig05 RTA)
import re, json, os, sys, zipfile, difflib, glob, math
import xml.etree.ElementTree as ET
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding='utf-8')
REP = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\보고서_20261012"
MC = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927"
OUT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C06\C06-T2"
V0 = os.path.join(OUT, 'verify_0'); os.makedirs(V0, exist_ok=True)
CACHE = os.path.join(OUT, 'cache')
sys.path.insert(0, os.path.join(OUT, 'scripts'))
from s1_docx_diff import expected_paras  # 기대 문단(md2docx 규칙)은 같은 함수 사용, 추출만 독립
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
norm = lambda s: re.sub(r'\s+', ' ', s).strip()
res = {}

# ---------- (1) XML 직접 파싱 ----------
def xml_paras(path):
    root = ET.fromstring(zipfile.ZipFile(path).read('word/document.xml'))
    body = root.find(W + 'body'); out = []
    def ptext(p):
        s = ''
        for el in p.iter():
            if el.tag == W + 't' and el.text: s += el.text
            elif el.tag == W + 'tab': s += '\t'
            elif el.tag in (W + 'br', W + 'cr'): s += '\n'
        return s
    for ch in body:
        if ch.tag == W + 'p':
            t = ptext(ch)
            if norm(t): out.append(norm(t))
        elif ch.tag == W + 'tbl':
            for tr in ch.iter(W + 'tr'):
                for tc in tr.findall(W + 'tc'):
                    t = '\n'.join(ptext(p) for p in tc.findall(W + 'p'))
                    if norm(t): out.append(norm(t))
    return out
pairs = {'v7': (os.path.join(REP, 'manuscript_v3.md'), os.path.join(REP, '보고서_초안_v7.docx'), 's1_v7docx_vs_md_v3'),
         'v6': (os.path.join(REP, 'manuscript_v3.md'), os.path.join(REP, '보고서_초안_v6.docx'), 's1_v6docx_vs_md_v3')}
for v, (md, dx, key) in pairs.items():
    xp = xml_paras(dx)
    pd_list = [norm(t) for _, t in json.load(open(os.path.join(CACHE, key + '.json'), encoding='utf-8'))['got']]
    exp = [norm(t) for (ln, k, t) in expected_paras(md)[0] if norm(t) and k != 'figure']
    sm = difflib.SequenceMatcher(None, exp, xp, autojunk=False)
    diff_lines = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op != 'equal': diff_lines.append((op, i1, i2, j1, j2))
    exp_ln = [ln for (ln, k, t) in expected_paras(md)[0] if norm(t) and k != 'figure']
    md_lines_diff = sorted(set(exp_ln[i] for op, i1, i2, j1, j2 in diff_lines for i in range(i1, i2)))
    res['xml_' + v] = {'n_xml': len(xp), 'n_python_docx': len(pd_list), 'xml_equals_python_docx': xp == pd_list,
                       'n_diff_blocks_vs_md_v3': len(diff_lines), 'md_lines_differing': md_lines_diff,
                       'docx_extra_paras': [xp[j][:80] for op, i1, i2, j1, j2 in diff_lines for j in range(j1, j2) if op == 'insert']}

# ---------- (2a) fig02: 예측 파일로 AUC·skill 독립 재계산(순위합) ----------
def auc(y, p):
    y = np.asarray(y).astype(int); p = np.asarray(p, float)
    r = pd.Series(p).rank(method='average').values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
sk = lambda a, b: (a - b) / (1 - b)
spec = {
 'T3': (os.path.join(MC, 'T3_기술셀부상예측', 'R3', 'preds', 'FINAL_R3_t0_2021.parquet'), 'yb', 'p', 'p_base'),
 'N8': (os.path.join(MC, 'N8_특허분쟁표적레이더', 'final_preds_r6.parquet'), 'y', 'p_champ', 'p_M0'),
 'N4': (os.path.join(MC, 'N4_특허행태_기업폐업조기경보', 'final_test_pred_r4.parquet'), 'y', 'p_champ', 'p_M0'),
 'N5': (os.path.join(MC, 'N5_표준특허선언레이더', 'final_preds_r4.parquet'), 'y', 'p_champ', 'p_M0'),
 'N9': (os.path.join(MC, 'N9_정부RnD과제_특허성과예측', 'final_preds_r6.parquet'), 'y', 'p_champ', 'p_M0_L2'),
 'N6': (os.path.join(MC, 'N6_정부RnD기업_첫특허전환', 'final_preds_r5.parquet'), 'y', 'p_champ', 'p_m0'),
 'N7': (os.path.join(MC, 'N7_특허기업_정부RnD첫수행', 'final_test_pred_r5.parquet'), 'y', 'p_champ', 'p_M0'),
}
f2 = pd.read_csv(os.path.join(REP, 'figs', 'fig02_scorecard_9tracks.csv'), encoding='utf-8-sig').set_index('트랙')
fig02 = []
for t, (f, yc, pc, bc) in spec.items():
    d = pd.read_parquet(f, columns=[yc, pc, bc]).dropna()
    am, ab = auc(d[yc], d[pc]), auc(d[yc], d[bc]); s = sk(am, ab)
    row = f2.loc[t]
    fig02.append({'track': t, 'n': len(d), 'pos': int(d[yc].sum()), 'auc_model': am, 'auc_base': ab, 'skill': s,
                  'fig_auc_model': row.AUC_모형, 'fig_auc_base': row.AUC_기준선, 'fig_skill': row.skill,
                  'match_3dp': (round(am, 3) == round(row.AUC_모형, 3)) and (round(ab, 3) == round(row.AUC_기준선, 3)) and (round(s, 3) == round(row.skill, 3))})
# 보수적 skill 일부(예측 파일에 기준선 있는 것): N8 SB, N9 strong_L2
d = pd.read_parquet(spec['N8'][0]); fig02.append({'track': 'N8 보수(SB)', 'skill': sk(auc(d.y, d.p_champ), auc(d.y, d.p_SB)), 'fig_skill': 0.312})
d = pd.read_parquet(spec['N9'][0]); fig02.append({'track': 'N9 보수(strong_L2)', 'skill': sk(auc(d.y, d.p_champ), auc(d.y, d.p_strong_L2)), 'fig_skill': 0.356})
res['fig02'] = fig02

# ---------- (2b) fig09: 상위 k 적중률·상위 100건 ----------
d = pd.read_parquet(spec['N9'][0]); n = len(d)
cols = ['p_champ', 'p_M0_L2', 'p_strong_L2', 'p_pilot_L2', 'p_M0_orig', 'p_strong_orig', 'p_pilot_orig']
f9 = []
for c in cols:
    o = d.sort_values(c, ascending=False, kind='mergesort').y.values
    r = {'model': c}
    for q in (0.01, 0.05, 0.10):
        k_round = int(round(n * q)); k_ceil = math.ceil(n * q)
        r[f'top{int(q*100)}_round(k={k_round})'] = o[:k_round].mean(); r[f'top{int(q*100)}_ceil(k={k_ceil})'] = o[:k_ceil].mean()
    r['hit100'] = int(o[:100].sum()); r['base_rate'] = d.y.mean()
    f9.append(r)
res['fig09'] = f9

# ---------- (2c) fig05: 출원 2017–19 발명특허 RTA(5개국 동일가중) ----------
cnt = {}
for f in sorted(glob.glob(os.path.join(MC, '_shared', 'meta', '*.parquet'))):
    t = pq.read_table(f, columns=['cc', 'appl_no', 'appl_year', 'ksic'], filters=[('appl_year', '>=', 2017), ('appl_year', '<=', 2019)]).to_pandas()
    t['cc'] = t.cc.astype(str); a = t.appl_no.astype(str)
    inv = np.where(t.cc == 'KR', a.str[:2] == '10',
          np.where(t.cc == 'CN', a.str.len().isin([12, 14]) & a.str[4].isin(['1', '8']),
          np.where(t.cc == 'JP', a.str.len().isin([8, 17]), True)))
    g = t[inv].groupby(['ksic', 'cc'], observed=True).size()
    for (k, c), v in g.items(): cnt[(str(k), c)] = cnt.get((str(k), c), 0) + int(v)
    del t
m = pd.Series(cnt).unstack()[['KR', 'US', 'JP', 'CN', 'EP']]
share = m / m.sum(0)
rta = share.div(share.mean(1), axis=0)
f5 = pd.read_csv(os.path.join(REP, 'figs', 'fig05_rta_map_5x50.csv'), encoding='utf-8-sig')
f5['ksic'] = f5.ksic.astype(str); f5 = f5.set_index('ksic')[['KR', 'US', 'JP', 'CN', 'EP']]
rta = rta.reindex(f5.index)
lead = rta.idxmax(1)
res['fig05'] = {'max_abs_diff_rta': float((rta - f5).abs().max().max()), 'n_kr_gt1': int((rta.KR > 1).sum()),
                'leaders': lead.value_counts().to_dict(), 'n_gap_lt0.7': int((rta.KR / rta.max(1) < 0.7).sum()),
                'kr_2dp': {k: round(float(rta.loc[k, 'KR']), 2) for k in ['20423', '26212', '31312']},
                'hanyak_kr_over_cn': round(float(rta.loc['21220', 'KR'] / rta.loc['21220'].max()), 3) if '21220' in rta.index else None,
                'printer_kr_over_jp': round(float(rta.loc['26323', 'KR'] / rta.loc['26323'].max()), 3),
                'oven_29150_kr': float(rta.loc['29150', 'KR']), 'n_docs_2017_19': int(m.values.sum())}
json.dump(res, open(os.path.join(V0, 'v0_results.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if k.startswith('xml')}, ensure_ascii=False, indent=1))
for r in res['fig02']: print(r)
for r in res['fig09']: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
print(res['fig05'])
