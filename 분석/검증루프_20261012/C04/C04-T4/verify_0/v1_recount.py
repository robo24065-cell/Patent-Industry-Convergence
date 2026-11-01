# verify_0 반박 검증: P1 건수와 as-of 연결을 독립 재구현으로 다시 센다(libA 미사용).
# A: 비겹침 원점 4개, 기업키 일치만(사업장 동일성 추적 없이) 결과 재계산 → s1 값과 대조.
# B: 원점 9개 양성 재계산 + NTIS 이름 불일치로 남는 N6 잔여 겹침(biz6만 일치) 추정.
# as-of: (1) 연결에 쓰인 문서의 보인 날짜 ≤ 기준일 (2) 연결/탐지가 '나중 등록 여부'(사후)에 좌우되는지(C03-C 반박 유형).
import sys, re, csv, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
EXT = B + r'\분석\모델경쟁_20260927\_shared\ext'
W = B + r'\분석\검증루프_20261012\C04\C04-T4'
out = {}
rx_suf = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
rx_p = re.compile(r"[\s\(\)\[\]·\.,\-&'\"/]")
def nz(s):
    return rx_p.sub('', rx_suf.sub('', str(s or '')).upper())
# CA 독립 파싱
name_b6 = {}; name_bad = set()
with open(B + r'\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('¶')
        if len(p) < 5: continue
        n = nz(p[1]); bz = re.sub(r'\D', '', p[4])
        if len(n) < 2: continue
        if len(bz) != 10:
            name_bad.add(n); continue
        b6 = bz[:6]
        if n in name_b6 and name_b6[n] != b6: name_bad.add(n)
        name_b6.setdefault(n, b6)
name_ok = {n: b for n, b in name_b6.items() if n not in name_bad}
corp_tc = {str(x) for x in range(81, 89)}
name_ok = {n: b for n, b in name_ok.items() if b[3:5] in corp_tc}
out['name_ok_count'] = len(name_ok)
k = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원번호', '공개일자', '등록일자', '출원인명', '세세분류코드', '우선권국가'])
pub = k['공개일자'].fillna('').astype(str).str.strip(); rg = k['등록일자'].fillna('').astype(str).str.strip()
pub = pub.where(pub.str.len() == 8, '99999999'); rg = rg.where(rg.str.len() == 8, '99999999')
k['vis'] = np.where(pub < rg, pub, rg); k['pub'] = pub; k['rg'] = rg
k['n'] = k['출원인명'].map(nz)
k['b6'] = k.n.map(name_ok)
REF = {'201512': '20151130', '201612': '20161130', '201712': '20171130', '201812': '20181130', '201912': '20191130',
       '202012': '20201231', '202112': '20211231', '202212': '20221231', '202312': '20231231'}
S1 = {'201512': '201712', '201712': '201912', '201912': '202112', '202112': '202312'}
def nps(s):
    assert s not in ('202512', '202608')
    d = pd.read_parquet(EXT + rf'\firm_nps_wkpl_{s}.parquet', columns=['name', 'biz6', 'stat', 'n_emp'])
    d = d[(d.stat.astype(str) == '1') & d.biz6.str[3:5].isin(corp_tc)]
    return (d.biz6 + '|' + d.name.map(nz)).values, d.n_emp.values
fv = k[k.b6.notna()].groupby('n').vis.min()
# ---- A
A = {}
for s0, s1 in S1.items():
    ref = REF[s0]
    names = set(fv[fv <= ref].index)
    keys = {name_ok[n] + '|' + n for n in names}
    kk, ee = nps(s0); e0 = pd.Series(ee).groupby(kk).sum()
    units = e0[(e0.index.isin(keys)) & (e0 >= 5)]
    k1, e1v = nps(s1); e1 = pd.Series(e1v).groupby(k1).sum().reindex(units.index).fillna(0)
    y = (e1 >= 1.3 * units) & (e1 - units >= 10)
    A[s0] = dict(units=int(len(units)), pos_key_only=int(y.sum()), rate=round(float(y.mean()), 4))
    # as-of: 연결 상호의 보인 문서가 모두 기준일 이전인지(정의상) + 나중 등록 여부별 연결률
    dv = k[(k.vis <= ref) & k.b6.notna()]
    assert dv.vis.max() <= ref
    lr = dv.assign(key=dv.b6 + '|' + dv.n).assign(lk=lambda x: x.key.isin(set(units.index)))
    later = lr.rg.between(str(int(ref) + 1), '99999998')
    A[s0]['doc_link_rate_later_reg_vs_not'] = [round(float(lr[later].lk.mean()), 4), round(float(lr[~later].lk.mean()), 4)]
s1c = pd.read_parquet(W + r'\cache\A_units_dev.parquet')
for s0 in S1:
    A[s0]['s1_pos_with_ba'] = int(s1c[s1c.s0 == s0].yA.sum()); A[s0]['s1_units'] = int((s1c.s0 == s0).sum())
A['nonoverlap_sum_key_only'] = int(sum(A[s]['pos_key_only'] for s in S1))
out['A'] = A; print(json.dumps(A, ensure_ascii=False), flush=True)
# ---- B
nt = pd.read_parquet(EXT + r'\ntis_project_raw.parquet', columns=['year', 'corp_rgst_no', 'research_agency'])
nt['bz'] = nt.corp_rgst_no.fillna('').str.replace(r'\D', '', regex=True)
nt = nt[nt.bz.str.len() == 10]
def add6(ref):
    y, m = int(ref[:4]), int(ref[4:6]); m += 6; y += (m - 1) // 12; m = (m - 1) % 12 + 1
    import calendar
    return f'{y:04d}{m:02d}{min(int(ref[6:]), calendar.monthrange(y, m)[1]):02d}'
Bres = {}; seen = set(); tot = 0; resid = 0
for s0, ref in REF.items():
    t0 = int(s0[:4]); end = add6(ref)
    assert end <= '20240630'
    kk, ee = nps(s0); e0 = pd.Series(ee).groupby(kk).sum(); e0 = e0[e0 >= 5]
    has = {name_ok[n] + '|' + n for n in fv[fv <= ref].index}
    ntk = set(nt[nt.year.between(t0 - 5, t0 - 3)].pipe(lambda d: d.bz.str[:6] + '|' + d.research_agency.map(nz)))
    risk = set(e0.index) - has - ntk
    w = k[(k.pub > ref) & (k.pub <= end) & k.b6.notna()]
    w = w[w.n.map(fv) > ref]
    cand = set(w.b6 + '|' + w.n) & risk
    new = cand - seen; seen |= cand; tot += len(new)
    nt6 = set(nt[nt.year.between(t0 - 2, t0)].bz.str[:6]) | set(nt[nt.year.between(t0 - 5, t0 - 3)].bz.str[:6])
    r6 = sum(1 for c in cand if c.split('|')[0] in set(nt[nt.year.between(t0 - 5, t0 - 3)].bz.str[:6]))
    # 탐지가 창 문서의 '나중 등록'(사후)에 좌우되는지: 첫 공개 상호(국내 법인 표지) 문서 기준
    wa = k[(k.pub > ref) & (k.pub <= end)]
    wa = wa[wa['출원인명'].fillna('').str.contains(r'주식회사|\(주\)|㈜|유한회사')]
    fva = k.groupby('n').vis.min()
    wa = wa[wa.n.map(fva) > ref]
    det = (wa.b6 + '|' + wa.n).isin(cand)
    lreg = wa.rg != '99999999'
    Bres[s0] = dict(pos=len(cand), new_unique=len(new), residual_ntis_biz6_only=int(r6),
                    det_rate_later_reg_vs_not=[round(float(det[lreg].mean()), 4) if lreg.any() else None, round(float(det[~lreg].mean()), 4)],
                    later_reg_share=round(float(lreg.mean()), 4))
    print(s0, Bres[s0], flush=True)
Bres['unique_total'] = tot
s4 = pd.read_parquet(W + r'\cache\B_dev_pos.parquet')
Bres['s4_per_origin'] = s4.groupby('s0').size().to_dict(); Bres['s4_unique'] = int(s4.key.nunique())
out['B'] = Bres
json.dump(out, open(W + r'\verify_0\v1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({'A_sum': A['nonoverlap_sum_key_only'], 'B_unique': tot, 's4_unique': Bres['s4_unique']}))
