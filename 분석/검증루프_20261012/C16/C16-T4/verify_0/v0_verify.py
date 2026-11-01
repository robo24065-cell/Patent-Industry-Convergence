# C16-T4 verify_0 — 1차 코드(s1_audit·s3_facts·s4_residual)를 쓰지 않는 독립 재대조
# 파서: pandas.read_csv(1차는 csv 모듈), JSON은 원문 텍스트 정규식 추출(1차는 json.load), 원고 숫자는 원고 줄에서 정규식으로 직접 뽑음(1차는 기대 문자열 포함 검사)
import re, sys, random, json, os, sqlite3, glob
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
AN = BASE + r"\분석"; R = AN + r"\보고서_20261012"; MC = AN + r"\모델경쟁_20260927"; L = AN + r"\검증루프_20261012"
OUT = L + r"\C16\C16-T4"; V = OUT + r"\verify_0"
rng = random.Random(20261017)
ms = open(R + r"\manuscript_v3.md", encoding='utf-8', newline='').read().splitlines()
def raw(p): return open(p, encoding='utf-8-sig').read()
def jnum(text, *keys):
    # 중첩 키 경로를 정규식으로 따라가 첫 숫자 반환(json.load 미사용)
    pos = 0
    for k in keys:
        m = re.compile(r'"' + re.escape(k) + r'"\s*:\s*').search(text, pos)
        if not m: return None
        pos = m.end()
    m = re.compile(r'-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?').match(text, pos)
    return float(m.group(0)) if m else None
def R_(x, nd):  # 사사오입(소수 nd자리), 부동소수 오차 보정
    from decimal import Decimal, ROUND_HALF_UP
    return float(Decimal(str(round(x, 12))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))
def mnums(line):
    return [float(t.replace('−', '-').replace(',', '')) for t in re.findall(r'[−-]?\d[\d,]*\.?\d*', line.replace('–', ' ')) if re.search(r'\d', t)]
def present(line, val, nd):
    # 원고 줄에서 뽑은 숫자 중 val을 nd자리로 반올림한 값과 같은 것이 있는가(부호 무시)
    target = R_(abs(val), nd)
    return any(abs(abs(x) - target) < 10 ** (-nd) / 2 for x in mnums(line))

# ---------------- A. 부록 블록 독립 재계산 ----------------
appx = [482, 483, 485, 488, 498, 499, 501, 503, 519, 534, 542, 577, 638, 651]
sample_appx = sorted(rng.sample(appx, 5))   # 5/14 = 35.7%
sb15 = pd.read_csv(L + r"\C15\C15-T2\seed_bundles.csv"); sb13 = pd.read_csv(L + r"\C13\C13-T2\seed_bundles.csv")
vd = pd.concat([pd.read_csv(L + r"\C13\C13-T2\variance_decomp.csv"), pd.read_csv(L + r"\C15\C15-T2\variance_decomp.csv")])
t3s = raw(L + r"\C15\C15-T3\res\s5_t3_summary.json"); t3s_raw = t3s[t3s.find('"auc_raw"'):]
v5f = raw(L + r"\C11\C11-T2\verify_5\v5_frac.json"); v5o = raw(L + r"\C11\C11-T2\verify_5\v5_out.json")
pk = pd.read_csv(L + r"\C14\C14-T2\pkr_ipc_altA.csv"); ag = pd.read_csv(L + r"\C10\C10-T3\res\s2_agree.csv"); c1 = pd.read_csv(MC + r"\_shared_report\fusion_advantage\c5_R10\c1_label_agreement.csv")
s04 = raw(L + r"\C14\C14-T3\cache\s04_out.json"); hz = pd.read_csv(L + r"\C15\C15-T1\cache\s4_horizon.csv"); nt = pd.read_csv(MC + r"\_shared_report\ntis_t3_cohort_lapse_transfer.csv")
f1 = raw(MC + r"\T1_산업대기실\R3\final_result_r3.json"); f3 = raw(MC + r"\T3_기술셀부상예측\R3\final_result_r3.json")
con = sqlite3.connect('file:' + BASE + r"\수집파이프라인\registry_full.sqlite" + '?mode=ro', uri=True)
n10 = con.execute("select count(*) from patent where rgst_no like '10%'").fetchone()[0]
n20 = con.execute("select count(*) from patent where rgst_no like '20%'").fetchone()[0]
con.close()
T1 = sb15[sb15.track == 'T1']; T3 = sb15[sb15.track == 'T3']
w = '출원2015-19_발명'
def agrow(cc): return ag[(ag.window == w) & (ag.cc == cc)].iloc[0]
def c1row(cc): return c1[(c1.window == w) & (c1.cc == cc)].iloc[0]
kr = jnum(t3s_raw, 'kr_rebuild'); base = jnum(t3s_raw, 'drop_ros')
def ret(k): return (jnum(t3s_raw, k) - kr) / (base - kr)
def hzv(h, g): return float(hz[(hz.h == h) & (hz.group == g)]['regdate'].iloc[0])
X = {  # 행: [(이름, 값, 자릿수)]
 482: [('ρ asc', jnum(v5f, 'asc_free_F', 'rho'), 3), ('ρ desc 표만', jnum(v5f, 'desc_free_F', 'rho'), 3), ('ρ desc 전부', jnum(v5o, 'REG_tie_desc_full', 'max_raw'), 3), ('RTA>1', jnum(v5f, 'asc_free_F', 'kr_rta_gt1'), 0)],
 483: [('T3 d2f 최소', T3.dev2final.min(), 3), ('T3 d2f 최대', T3.dev2final.max(), 3), ('T3 d2f 공식', T3[T3.official_bundle].dev2final.iloc[0], 3)],
 485: [('T3 공식', jnum(f3, 'summary', 'mean_skill'), 3), ('T3 최소', T3.skill.min(), 3), ('T3 최대', T3.skill.max(), 3)],
 488: [('T1 공식', jnum(f1, 'skill', 'auc_all'), 3), ('T1 최소', T1.skill.min(), 3), ('T1 최대', T1.skill.max(), 3), ('하한 최소', T1.ci_lo.min(), 3), ('하한 최대', T1.ci_lo.max(), 3)],
 498: [('US PKR', float(pk[pk.cc == 'US'].대안A_REG.iloc[0]), 1), ('US TXT', agrow('US').txt5_reg * 100, 1), ('US FREE', agrow('US').free5_reg * 100, 1), ('US 그룹', c1row('US').cw_group * 100, 1)],
 499: [('JP PKR', float(pk[pk.cc == 'JP'].대안A_REG.iloc[0]), 1), ('JP TXT', agrow('JP').txt5_reg * 100, 1), ('JP FREE', agrow('JP').free5_reg * 100, 1), ('JP 5자리', c1row('JP').cw5 * 100, 1)],
 501: [('EP PKR', float(pk[pk.cc == 'EP'].대안A_REG.iloc[0]), 1), ('EP TXT', agrow('EP').txt5_reg * 100, 1), ('EP FREE', agrow('EP').free5_reg * 100, 1), ('EP n', c1row('EP').n, 0)],
 503: [('계', n10 + n20, 0), ('특허', n10, 0), ('실용', n20, 0)],
 519: [(k, jnum(t3s_raw, k), 4) for k in ['drop_ros', 'kr_rebuild', 'hyb_txt_reg@C10', 'hyb_free_reg@C10', 'hyb_pkr_reg']],
 534: [('하한>0 묶음', float((sb13.ci_lo > 0).sum() + (sb15.ci_lo > 0).sum()), 0), ('총 묶음', float(len(sb13) + len(sb15)), 0),
       ('T1 SBB 하한 최소', T1.strong_ci_lo.min(), 3), ('T1 SBB 하한 최대', T1.strong_ci_lo.max(), 3), ('T3 SB3 하한 최소', T3.strong_ci_lo.min(), 3), ('T3 SB3 하한 최대', T3.strong_ci_lo.max(), 3),
       ('T4 강화 하한 최소', sb13[sb13.track == 'T4'].strong_ci_lo.min(), 3), ('T4 강화 하한 최대', sb13[sb13.track == 'T4'].strong_ci_lo.max(), 3),
       ('sd/SE 최소', vd.ratio_seed_sd_to_boot_se.min(), 2), ('sd/SE 최대', vd.ratio_seed_sd_to_boot_se.max(), 2)],
 542: [('행', jnum(s04, 'rl_total_rows'), 0), ('범위밖', jnum(s04, 'rl_patent_unique_out_of_scope'), 0), ('비율', jnum(s04, 'rl_patent_unique_out_of_scope') / jnum(s04, 'rl_patent_unique') * 100, 1)],
 577: [('공식 상위1000', jnum(f1, 'topk', 'model', '1000', 'precision') * 100, 1), ('시드 최소', T1.topk2.min() / 10, 1), ('시드 최대', T1.topk2.max() / 10, 1),
       ('공식 순위', float(1 + (T1.topk2 > T1[T1.official_bundle].topk2.iloc[0]).sum()), 0)],
 638: [('txt 보존', ret('hyb_txt_reg@C10') * 100, 0), ('free 보존', ret('hyb_free_reg@C10') * 100, 0), ('pkr 보존', ret('hyb_pkr_reg') * 100, 0), ('상실 상한', (1 - ret('hyb_free_reg@C10')) * 100, 0)],
 651: [('5년 기원', float(nt[nt.group == '정부R&D 기원(전체)'].lapse5.iloc[0]) * 100, 1), ('5년 비기원', float(nt[nt.group == '정부R&D 기원 아님'].lapse5.iloc[0]) * 100, 1),
       ('7년 기원', hzv(7, '정부R&D 기원') * 100, 1), ('7년 비기원', hzv(7, '비기원') * 100, 1), ('10년 기원', hzv(10, '정부R&D 기원') * 100, 1), ('10년 비기원', hzv(10, '비기원') * 100, 1)],
}
resA = []
for ln in appx:
    line = ms[ln - 1]
    for name, val, nd in X[ln]:
        if name == '공식 순위':
            ok = (val == 2) and ('두 번째로 높음' in line)
        elif name in ('하한>0 묶음', '총 묶음'):
            ok = ('76/76' in line) and val == 76
        else:
            ok = present(line, val, nd)
        resA.append(dict(행=ln, 표본=ln in sample_appx, 항목=name, 원천값=round(float(val), 6), 자릿수=nd, 원고에있음=ok))
dA = pd.DataFrame(resA)
dA.to_csv(V + r"\v0_appendix.csv", index=False, encoding='utf-8-sig')

# ---------------- B. facts 패치 행 표본 ----------------
log = pd.read_csv(OUT + r"\facts_patch_consolidated.csv", dtype=str).fillna('')
idx = sorted(rng.sample(range(len(log)), 31))   # 31/102 = 30.4%
def readf(d, name):
    return pd.read_csv(d + '\\' + name, dtype=str, keep_default_na=False, on_bad_lines='skip', engine='python')
orig = {n: readf(R + r"\facts", n) for n in ['facts.csv', 'facts_supplement.csv']}
work = {n: readf(OUT + r"\facts_work", n) for n in ['facts.csv', 'facts_supplement.csv', 'facts_supplement_R12.csv']}
approw = {}
for i, s in enumerate(ms):
    if s.startswith('| ') and i > 440:
        c = [x.strip() for x in s.strip('|').split(' | ')]
        if len(c) >= 4: approw[re.sub(r'\s*\(보충.*\)$', '', c[0])] = ' | '.join(c[2:-1])
ERR = raw(L + r"\ERRATA.txt"); SYNC = raw(L + r"\C14\C14-T2\facts_sync.csv"); FPATCH = raw(L + r"\C15\C15-T3\facts_patch.csv")
rho = pd.read_csv(L + r"\C15\C15-T3\res\s4_rho_table.csv")
A_ = t3s[t3s.find('"A"'):]
SRC = {  # 원천 숫자(독립 재계산) — id: 기대 토큰 목록
 'C-T3-drop_ros-hyb_pkr': [f"{R_(jnum(A_, 'drop_ros−hyb_pkr_reg', 'dAUC_raw'), 4):.4f}"],
 'C-T3-drop_ros-hyb_txt': [f"{R_(jnum(A_, 'drop_ros−hyb_txt_reg@C10', 'dAUC_raw'), 4):.4f}"],
 'C-T3-drop_ros-hyb_free': [f"{R_(jnum(A_, 'drop_ros−hyb_free_reg@C10', 'dAUC_raw'), 4):.4f}"],
 'C-T3-hyb_txt-kr_rebuild': [f"{R_(jnum(A_, 'hyb_txt_reg@C10−kr_rebuild', 'dAUC_raw'), 4):.4f}"],
 'C-T3-hyb_free-kr_rebuild': [f"{R_(jnum(A_, 'hyb_free_reg@C10−kr_rebuild', 'dAUC_raw'), 4):.4f}"],
 'C-RCA-PKR': [f"{float(rho[rho.arm == 'reg'].rho_pkr_raw.iloc[0]):.3f}"],
 'C-LAND-pkr_H': [f"{R_(float(rho[rho.arm == 'reg'].land_rho_KR_raw.iloc[0]), 3):.3f}"],
 'C-T3-AUCS': [f"{R_(jnum(t3s_raw, 'hyb_pkr_reg'), 4):.4f}"],
 'C-T3-RET': [f"{R_(ret('hyb_pkr_reg') * 100, 0):.0f}"],
 'DATA-KR-REG': [f"{n10:,}", f"{n20:,}"],
 'C-AGREE-US': [f"{R_(float(pk[pk.cc == 'US'].대안A_REG.iloc[0]), 1):.1f}"], 'C-AGREE-JP': [f"{R_(float(pk[pk.cc == 'JP'].대안A_REG.iloc[0]), 1):.1f}"],
 'C-AGREE-EP': [f"{R_(float(pk[pk.cc == 'EP'].대안A_REG.iloc[0]), 1):.1f}"],
}
def cell(tbl, fid, col):
    hit = tbl[tbl['id'] == fid]
    return None if hit.empty else hit.iloc[0][col]
resB = []
for k in idx:
    r = log.iloc[k]; f = r['파일']; fid = r['id'].split('→')[-1]; col = r['열']
    kind = r['종류']
    integ = ''
    if col in ('정확한 문장 조각', '조건·경고', '수치', '출처 파일 경로'):
        now = cell(work[f], fid, col) if f in work else None
        integ = '일치' if now == r['새값'] else f'불일치(사본={now!r})'
        if f in orig:
            o = cell(orig[f], r['id'].split('→')[0], col)
            if o is not None and o != r['현재값']: integ += f'; 원본 현재값 불일치'
    elif kind in ('행 추가', '행 대체'):
        integ = '일치' if not work[f][work[f]['id'] == fid].empty else '사본에 행 없음'
    elif kind == '형식':
        integ = '일치' if not work[f][work[f]['id'] == fid].empty else '형식 행 없음'
    else:
        integ = '변경 없음'
    # 근거 대조: 새값 숫자 토큰이 (부록 같은 ID 행 ∪ 제안문(ERRATA·facts_sync·facts_patch)) 안에 있는가 + 원천 재계산
    new = r['새값']
    toks = set(t.replace('−', '-').lstrip('+-') for t in re.findall(r'[−+\-]?\d[\d,]*(?:\.\d+)?', new))
    pool = (approw.get(fid, '') + ' ' + ERR + ' ' + SYNC + ' ' + FPATCH)
    ptoks = set(t.replace('−', '-').lstrip('+-') for t in re.findall(r'[−+\-]?\d[\d,]*(?:\.\d+)?', pool))
    miss = sorted(t for t in toks if t not in ptoks and not re.fullmatch(r'\d{3}', t))
    src = SRC.get(fid); src_ok = ''
    B_ = t3s[t3s.find('"B"'):]
    SRC_APP = {('C-RCA-PKR', '조건·경고'): [f"{float(rho[rho.arm == 'pure'].rho_pkr_raw.iloc[0]):.3f}", f"{float(rho[rho.arm == 'stored'].rho_pkr_raw.iloc[0]):.3f}"],
               ('C-T3-drop_ros-hyb_pkr', '조건·경고'): [f"{R_(jnum(B_, 'drop_ros−hyb_pkr_pure', 'dAUC_raw'), 4):.4f}", f"{R_(jnum(t3s[t3s.find('"stored"'):], 'drop_ros−hyb_pkr@stored', 'dAUC_raw'), 4):.4f}"]}
    if col == '수치':
        frag = cell(work[f], fid, '정확한 문장 조각') or ''
        parts = [x for x in re.split(r'[/ ]', new.replace('−', '-').lstrip('+')) if x]
        src_ok = '수치가 문장 조각 안에 있음' if all(x.lstrip('+-') in frag.replace('−', '-') for x in parts) else f'수치-문장 불일치({new} / {frag[:60]})'
    elif col == '출처 파일 경로':
        added = [x.strip() for x in new.replace(r[ '현재값'], '').strip(' /;').split(';')]
        paths = [a.split('(')[0].strip() for a in added if a.strip().startswith('분석')]
        ex = [os.path.exists(os.path.join(BASE, q.replace(', ', ',').split(',')[0])) for q in paths]
        src_ok = f'경로 존재 {sum(ex)}/{len(ex)}'
    elif (fid, col) in SRC_APP:
        exp = SRC_APP[(fid, col)]
        src_ok = '원천 일치' if all(x.lstrip('+-') in new for x in exp) else f'원천 불일치 {exp}'
    elif src and col in ('정확한 문장 조각', '(행 추가)'):
        src_ok = '원천 일치' if all(x in new or x.lstrip('+-') in new for x in src) else f'원천 불일치 {src}'
    resB.append(dict(로그행=k + 2, 파일=f, id=r['id'], 열=col, 종류=kind, 사본무결성=integ, 제안문밖숫자=' '.join(miss), 원천재계산=src_ok))
dB = pd.DataFrame(resB); dB.to_csv(V + r"\v0_facts_sample.csv", index=False, encoding='utf-8-sig')

# ---------------- C. USER_REQUESTS 누락 재검색 ----------------
ur = raw(OUT + r"\user_requests_final.txt")
hits = []
for p in sorted(glob.glob(L + r"\C*\C*-T*\summary.txt")):
    for i, s in enumerate(open(p, encoding='utf-8', errors='replace').read().splitlines(), 1):
        if re.search(r'USER_REQUESTS|회원가입|로그인', s):
            hits.append((os.path.relpath(p, L), i, s.strip()[:160]))
NEG = re.compile(r'없다|없음|없었|않았|않음|추가 없음|적을 것 없음|필요 없|로그인 없음|로그인 없이|로그인 불필요|로그인·회원가입 없음|미작성|쓰지 않았|산출물')
resC = []
for f, i, s in hits:
    task = f.split('\\')[1]
    req_like = not NEG.search(s)
    mapped = task in ur
    resC.append(dict(파일=f, 행=i, 요청성=('요청 후보' if req_like else '요청 아님(부정·완료 문구)'), 완결본에_바퀴표기=mapped, 내용=s))
dC = pd.DataFrame(resC); dC.to_csv(V + r"\v0_user_requests_scan.csv", index=False, encoding='utf-8-sig')
miss_C = dC[(dC.요청성 == '요청 후보') & (~dC.완결본에_바퀴표기)]
summary = dict(
    A_sample=sample_appx, A_sample_items=int(dA[dA.표본].shape[0]), A_sample_ok=int(dA[dA.표본].원고에있음.sum()),
    A_all_items=int(dA.shape[0]), A_all_ok=int(dA.원고에있음.sum()), A_bad=dA[~dA.원고에있음].to_dict('records'),
    B_sample_n=len(idx), B_integrity_ok=int((dB.사본무결성.isin(['일치', '변경 없음'])).sum()), B_out_of_proposal=dB[dB.제안문밖숫자 != ''][['id', '열', '제안문밖숫자']].to_dict('records'),
    B_src_checked=int((dB.원천재계산 != '').sum()), B_src_bad=dB[dB.원천재계산.str.startswith('원천 불일치')][['id', '원천재계산']].to_dict('records'),
    C_hits=len(dC), C_request_like=int((dC.요청성 == '요청 후보').sum()), C_missing=miss_C.to_dict('records'))
json.dump(summary, open(V + r"\v0_summary.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(summary, ensure_ascii=False, indent=1, default=str)[:4000])
