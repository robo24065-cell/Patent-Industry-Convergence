# C19-T3 s11: facts 패치에서 '원고 부록 셀을 베낀' 순환 18행 + v20 부록 변경 동기 행의 숫자 토큰을 원천 파일에서 직접 찾는다(E-C18T3-F (3)).
# 판정: 토큰마다 원천 파일 안의 수 x 가운데 표기 자릿수로 반올림해 같아지는 것(x, x×100, |x|)이 있거나, 원천 텍스트에 같은 문자열이 있으면 '근거 있음'.
# 구조 토큰(연도·자리수 정의·원고 줄 번호 등)은 행별 화이트리스트로 따로 표시한다. 근거 없는 숫자가 남는 행은 패치에서 빼고 '근거 미확보'.
import os, sys, re, csv, json, sqlite3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib19 import *
A = json.load(open(C + r'\s1_app.json', encoding='utf-8'))['APP']
FA = MC + r'\_shared_report\fusion_advantage'
LINKIO_NEW = None  # s10에서 만든 조각을 쓴다(아래에서 facts 원문으로 구성)
import io
facts_rows = {r['id']: r for r in csv.DictReader(io.StringIO(open(FACTS + r'\facts.csv', encoding='utf-8-sig').read()))}
orig_io = facts_rows['LINK-IO']['정확한 문장 조각']
LINKIO_NEW = orig_io.replace('한국은행 산업연관표 2010(통합소분류 165)', '한국은행 2010–2015–2020 접속 산업연관표의 2010년 표(2020 분류, 통합소분류 165; 파일 수정 2026-06-30)')
# registry 집계(mode=ro): 등록번호 접두 10/20
REG = AN.replace('\\분석', '') + r'\수집파이프라인\registry_full.sqlite'
reg_txt = ''
try:
    con = sqlite3.connect('file:' + REG + '?mode=ro', uri=True)
    cols = [r[1] for r in con.execute('pragma table_info(patent)')]
    rc = 'rgst_no' if 'rgst_no' in cols else cols[0]
    q = con.execute(f"select substr({rc},1,2), count(distinct {rc}) from patent group by 1").fetchall()
    reg_txt = json.dumps(dict(q)); con.close()
except Exception as e:
    reg_txt = 'ERR ' + str(e)
s0 = json.load(open(C + r'\s0_v1e_check.json', encoding='utf-8'))
p0rows = {cc: n for cc, n in s0['rows']['P0']}
tot = sum(p0rows.values())
p0_txt = json.dumps(dict(rows=p0rows, total=tot, share={cc: n / tot for cc, n in p0rows.items()}))
SRC = {
 'LINK-IO': [MC + r'\T1_산업대기실\R2\ext_link_summary.csv', VL + r'\C10\C10-T1\data_check.txt', VL + r'\C10\C10-T1\verify_2\v2_io_guide.txt'],
 'T3NC-MG': [MC + r'\T3_기술셀부상예측\TXT_R9\summary_mg_r9.json', VL + r'\C11\C11-T2\res\s5_summary_mg__reg.json'],
 'C-LAND-free_F': [FA + r'\c5_R10\c2_results.json', VL + r'\C11\C11-T2\verify_5\v5_out.json', VL + r'\C11\C11-T2\verify_5\v5_frac.json', VL + r'\C11\C11-T2\res\s7_landscape.json'],
 'C-LAND-txt_F': [FA + r'\c5_R10\c2_results.json', VL + r'\C11\C11-T2\res\s7_landscape.json', VL + r'\C11\C11-T2\res\c11t2_results.json'],
 'C-AGREE-US': [FA + r'\c5_R10\c1_label_agreement.csv', VL + r'\C10\C10-T3\res\s2_agree.csv', VL + r'\C14\C14-T2\pkr_ipc_altA.csv'],
 'C-AGREE-JP': [FA + r'\c5_R10\c1_label_agreement.csv', VL + r'\C10\C10-T3\res\s2_agree.csv', VL + r'\C14\C14-T2\pkr_ipc_altA.csv'],
 'C-AGREE-EP': [FA + r'\c5_R10\c1_label_agreement.csv', VL + r'\C10\C10-T3\res\s2_agree.csv', VL + r'\C14\C14-T2\pkr_ipc_altA.csv'],
 'C-AGREE-RANGE': [FA + r'\c5_R10\c1_label_agreement.csv', VL + r'\C10\C10-T3\res\s2_agree.csv'],
 'C-RCA-TXT': [VL + r'\C10\C10-T3\res\s3_rho.json', VL + r'\C10\C10-T3\res\s3_rho_table.csv', FA + r'\c5_R10\c2_results.json'],
 'C-RCA-FREE-IPC': [VL + r'\C10\C10-T3\res\s3_rho.json', VL + r'\C10\C10-T3\res\s3_rho_table.csv', FA + r'\c5_R10\c2_results.json'],
 'C-T3-AUCS': [FA + r'\c5_R10\c5_t3_summary.json', VL + r'\C10\C10-T3\res\s6_t3_summary.json', VL + r'\C15\C15-T3\res\s5_t3_summary.json'],
 'C-T3-RET': [FA + r'\c5_R10\c5_t3_summary.json', VL + r'\C10\C10-T3\res\s6_t3_summary.json', VL + r'\C15\C15-T3\res\s5_t3_summary.json'],
 'C-T3-drop_ros-hyb_txt': [FA + r'\c5_R10\c5_t3_summary.json', VL + r'\C10\C10-T3\res\s6_t3_summary.json'],
 'C-T3-drop_ros-hyb_free': [FA + r'\c5_R10\c5_t3_summary.json', VL + r'\C10\C10-T3\res\s6_t3_summary.json'],
 'DATA-REG-EVENTS': [MC + r'\_shared\README.md', VL + r'\C15\C15-T1\cache\s2_out.json', VL + r'\C02\C02-T3\data_check.txt', VL + r'\C02\C02-T3\verify_0\v1_transfer_out.txt'],
 'DATA-KR-REG': [MC + r'\_shared\ext\ntis_link_rates.csv', VL + r'\C15\C15-T1\cache\s2_out.json', '<registry>'],
 'LINK-NTIS': [MC + r'\_shared\ext\ntis_link_rates.csv', VL + r'\C16\C16-T3\kr_prefix.csv', VL + r'\C16\C16-T3\summary.txt'],
 'C-COUNTRY-SHARE': [FA + r'\c5_R10\c1_label_agreement.csv', '<s0_P0>', VL + r'\C16\C16-T3\kr_prefix.csv'],
 # v20 부록 변경 동기 행(순환 아님, 원천 대조)
 'LINK-KOSIS5': [MC + r'\T1_산업대기실\R2\ext_link_summary.csv', VL + r'\C07\C07-T1\official_compare.csv', VL + r'\C07\C07-T1\summary.txt', VL + r'\C07\C07-T1\data_check.txt'],
 'C13-SEED6': [VL + r'\C13\C13-T2\seed_bundles.csv', VL + r'\C13\C13-T2\variance_decomp.csv', VL + r'\C15\C15-T2\seed_bundles.csv', VL + r'\C15\C15-T2\variance_decomp.csv', VL + r'\C15\C15-T2\summary.txt', VL + r'\C13\C13-T2\summary.txt'],
 'C18-T1-REGGAP': [VL + r'\C18\C18-T1\monthly_capture.csv', VL + r'\C18\C18-T1\sens_results.csv', VL + r'\C18\C18-T1\summary.txt'],
 'R11-PKR': [MC + r'\_shared_report\label_origin_R10\foreign_public_R11\res_main.csv', VL + r'\C14\C14-T2\summary.txt'],
 'NTIS-T3-SUM': [MC + r'\_shared_report\ntis_t3_cohort_lapse_transfer.csv', VL + r'\C15\C15-T1\cache\s4_horizon.csv', VL + r'\C15\C15-T1\summary.txt', VL + r'\C02\C02-T1\_tab_e_lapse.json'],
}
# 구조 토큰(행별): 연도·자리수 정의·단계 번호·원고 줄 번호 등 — 원천 수치가 아님
STRUCT = {
 'LINK-IO': {'2010', '2015', '2020', '5', '4', '3', '165', '2026', '06', '30', '50'},
 'C-LAND-free_F': {'168', '7'}, 'C-LAND-txt_F': {'11', '2', '7'},
 'C-AGREE-US': {'2015', '19', '10', '3', '14', '2'}, 'C-AGREE-JP': {'2015', '19', '10', '3', '14', '2'}, 'C-AGREE-EP': {'2015', '19', '10', '3', '14', '2'},
 'C-AGREE-RANGE': {'4', '5', '10', '3'}, 'C-RCA-TXT': {'10', '3'}, 'C-RCA-FREE-IPC': {'10', '3'},
 'C-T3-AUCS': {'9', '15', '3'}, 'C-T3-RET': {'15', '3'}, 'C-T3-drop_ros-hyb_txt': {'10', '3'}, 'C-T3-drop_ros-hyb_free': {'10', '3'},
 'DATA-REG-EVENTS': {'2'}, 'T3NC-MG': {'10', '11', '2'},
 'LINK-KOSIS5': {'5', '1001', '10', '2008', '2020', '1101', '11', '2024', '2026', '13', '1', '50'},
 'C13-SEED6': {'4', '6', '7', '8', '9', '13', '15', '1', '3', '10', '3.92', '2', '5'},
 'C18-T1-REGGAP': {'2024', '2025', '1', '7', '01', '08', '02'}, 'R11-PKR': {'14', '2'}, 'NTIS-T3-SUM': {'2007', '2016', '5', '7', '10', '15', '6', '1'},
 'LINK-NTIS': set(), 'DATA-KR-REG': set(), 'C-COUNTRY-SHARE': set(),
}
NUM = re.compile(r'(?<![A-Za-z0-9_])[−\-+]?\d[\d,]*(?:\.\d+)?')
def frag_of(i):
    if i == 'LINK-IO': return LINKIO_NEW
    return A[i]['val']
def src_numbers(paths):
    txt = ''; nums = []
    for p in paths:
        if p == '<registry>': t = reg_txt
        elif p == '<s0_P0>': t = p0_txt
        elif os.path.exists(p):
            t = open(p, encoding='utf-8', errors='replace').read()
        else:
            t = ''
        txt += '\n' + t
    # 쉼표를 지우지 않은 원문(CSV 칸 구분 유지)과 천 단위 쉼표만 합친 판을 모두 쓴다
    for src_t in (txt, re.sub(r'(?<=\d),(?=\d{3}(?!\d))', '', txt)):
        for m in re.finditer(r'-?\d+(?:\.\d+)?(?:[eE]-?\d+)?', src_t):
            try: nums.append(float(m.group(0)))
            except Exception: pass
    return txt, nums
rows = []; verdict = {}
for i, paths in SRC.items():
    fr = frag_of(i); txt, nums = src_numbers(paths)
    miss_paths = [p for p in paths if not p.startswith('<') and not os.path.exists(p)]
    toks = []
    for m in NUM.finditer(fr.replace('–', ' – ').replace('~', ' ~ ').replace('/', ' / ')):
        t = m.group(0).replace('−', '-').replace('+', '').rstrip(',')
        toks.append(t)
    found_all = True
    for t in dict.fromkeys(toks):
        tn = t.replace(',', '').lstrip('-')
        if tn in STRUCT.get(i, set()):
            rows.append(dict(id=i, 토큰=t, 근거='구조 토큰(정의·연도·참조)', 원천=''))
            continue
        d = len(tn.split('.')[1]) if '.' in tn else 0
        try: v = float(tn)
        except Exception: continue
        hit = ''
        # 우선순위: (1) 같은 값 (2) 표기 자릿수 반올림 (3) ×100 반올림(비율→%)
        for x in nums:
            if abs(abs(x) - v) < 1e-9: hit = f'{x}(같은 값)'; break
        if not hit:
            for x in nums:
                if abs(round(abs(x), d) - v) < 1e-9: hit = f'{x}(반올림)'; break
        if not hit:
            for x in nums:
                if abs(round(abs(x) * 100, d) - v) < 1e-9: hit = f'{x}×100'; break
        if not hit and (t in txt or tn in txt.replace(',', '')):
            hit = '문자열 일치'
        if not hit: found_all = False
        rows.append(dict(id=i, 토큰=t, 근거=(hit if hit else '없음'), 원천=';'.join(os.path.relpath(p, AN) if not p.startswith('<') else p for p in paths)))
    verdict[i] = dict(근거=('있음' if found_all and not miss_paths else '미확보'), 없는토큰=[r['토큰'] for r in rows if r['id'] == i and r['근거'] == '없음'], 없는파일=miss_paths)
with open(W + r'\facts_work\circular_evidence.csv', 'w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['id', '토큰', '근거', '원천']); w.writeheader(); w.writerows(rows)
jdump(dict(verdict=verdict, linkio_new=LINKIO_NEW, registry=reg_txt), 's11_evidence.json')
for k, v in verdict.items(): out(k, v)
