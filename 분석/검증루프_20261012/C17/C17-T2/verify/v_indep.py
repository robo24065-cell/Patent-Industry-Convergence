# -*- coding: utf-8 -*-
"""C17-T2 verify: 독립 코더(PREREG §8). 주 판정(s1)과 다른 구현·다른 규칙으로 표본 30%(시드 20261018)를 3범주(정확/오매칭/불가)로 재판정.
 다른 점: (1) 권리목록은 T1 캐시 rl_pat.parquet(추출본), (2) 실체 동일성은 법인등록번호(rl_pat corp_rgstno·CA jurir) 우선,
          (3) 이름 비교는 정규화 문자열 포함이 아니라 음절 2-gram 자카드(≥0.6), (4) 등록원부는 owner 표 seq0(등록 당시 첫 권리자)·최종 권리자를 따로 본다,
          (5) 근거 문서는 kr_patents·app_pub에서 직접 다시 뽑는다(C16 alias_docs 미사용). s1 산출물·c17common 미사용.
 산출: verify/v_indep_out.json, verify/v_indep_pairs.csv"""
import sys, pathlib, json, re, sqlite3, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent/'scripts'))
import guard17 as G
import numpy as np, pandas as pd
t0 = time.time()
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
HERE = pathlib.Path(__file__).resolve().parent.parent; OUT = HERE/'verify'
TC = B/'분석/모델경쟁_20260927/T1_산업대기실/cache'
ro = lambda p: sqlite3.connect(pathlib.Path(p).as_uri() + '?mode=ro', uri=True)
SUFFIX = ['주식회사', '(주)', '㈜', '유한회사', '재단법인', '사단법인', '학교법인', '가부시키가이샤', '가부시끼가이샤', '코포레이션', '인코포레이티드', '인코퍼레이티드', '리미티드', '엘티디', '인크', '컴퍼니', '게엠베하', '에스에이', '아게']


def clean(s):
    s = str(s or '')
    for x in SUFFIX: s = s.replace(x, '')
    return re.sub(r'[^0-9A-Za-z가-힣]', '', s).lower()


def bg(s):
    s = clean(s); return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) > 1 else ({s} if s else set())


def sim(a, b):
    A, Bb = bg(a), bg(b)
    return len(A & Bb) / len(A | Bb) if A and Bb else 0.0


S = pd.read_csv(HERE/'alias_audit_v2.csv', dtype={'firm': str, 'appl_no': str})
sub = S.sample(frac=0.30, random_state=20261018).reset_index(drop=True)
# 근거 문서 직접 재추출: app_pub(이름매칭 결과)에서 그 법인×KSIC, 출원 2020–23, 공개·등록 ≤ 2025-08-22
a = pd.read_parquet(TC/'app_pub.parquet'); a['applicant_cd'] = a.applicant_cd.astype(str)
K = sorted(pd.read_parquet(TC/'ins_pat.parquet', columns=['ksic']).ksic.astype(str).unique()); ki = {k: i for i, k in enumerate(K)}
a['k'] = a.ksic.astype(str).map(ki)
def d8(x):
    x = str(x or '')[:8]; return int(x) if len(x) == 8 and x.isdigit() else 99999999
a['v'] = [min(d8(p), d8(r)) for p, r in zip(a.pub_date, a.rgst_date)]
a = a[(a.ay >= 2020) & (a.ay <= 2023) & (a.v <= 20250822)]
docs = a.merge(sub[['firm', 'k']], left_on=['applicant_cd', 'k'], right_on=['firm', 'k'])[['firm', 'k', 'appl_no']].astype({'appl_no': str})
kp = pd.read_parquet(B/'공모전데이터/작업데이터/kr_patents.parquet', columns=['출원번호', '등록번호', '출원인명']); kp['출원번호'] = kp['출원번호'].astype(str)
kp = kp[kp['출원번호'].isin(set(docs.appl_no))].set_index('출원번호')
docs['rg'] = docs.appl_no.map(kp['등록번호']).fillna('').astype(str); docs['names'] = docs.appl_no.map(kp['출원인명']).fillna('')
firms = pd.read_parquet(TC/'firms.parquet'); firms['applicant_cd'] = firms.applicant_cd.astype(str); fname = dict(zip(firms.applicant_cd, firms.applicant_name))
# 문서 안에서 법인 이름과 가장 비슷한 출원인명
docs['nm'] = [max(str(n).split('|'), key=lambda x: sim(x, fname.get(f, ''))) for n, f in zip(docs.names, docs.firm)]
# 등록원부
con = ro(B/'수집파이프라인/registry_full.sqlite')
pat = pd.read_sql('select rgst_no, applNo from patent', con); a2r = dict(zip(pat.applNo.astype(str), pat.rgst_no))
docs['rk'] = np.where(docs.rg.str.len() > 5, docs.rg, docs.appl_no.map(a2r).fillna(''))
rks = tuple(set(docs.rk) - {''}) or ('x',)
q = ','.join('?' * len(rks))
app = pd.read_sql(f'select rgst_no, applicantCd from applicant where rgst_no in ({q})', con, params=rks); app['applicantCd'] = app.applicantCd.astype(str)
own = pd.read_sql(f'select rgst_no, seq, finalOwnerYn, ownerName from owner where rgst_no in ({q})', con, params=rks)
con.close()
# 법인등록번호: rl_pat(추출본) + CA
rl = pd.read_parquet(TC/'rl_pat.parquet', columns=['applicant_cd', 'appl_no', 'corp_rgstno', 'ainfo', 'hinfo']); rl['applicant_cd'] = rl.applicant_cd.astype(str); rl['appl_no'] = rl.appl_no.astype(str)
corp = {}
for c, j in zip(rl.applicant_cd, rl.corp_rgstno):
    j = re.sub(r'\D', '', str(j or ''))
    if len(j) == 13: corp.setdefault(c, j)
ca = pd.read_parquet(B/'공모전데이터/외부데이터/DART/corp_applicant.parquet', columns=['출원인코드', 'jurir'])
for c, j in zip(ca['출원인코드'].astype(str), ca.jurir):
    j = re.sub(r'\D', '', str(j or ''))
    if len(j) == 13: corp.setdefault(c, j)
rlh = rl[rl.appl_no.isin(set(docs.appl_no))]
same = lambda c1, c2: c1 == c2 or (corp.get(c1) and corp.get(c1) == corp.get(c2))


def doc_v(r):
    if r.rk and (app.rgst_no == r.rk).any():
        codes = set(app[app.rgst_no == r.rk].applicantCd)
        return 'correct' if any(same(r.firm, c) for c in codes) else 'wrong'
    g = rlh[rlh.appl_no == r.appl_no]
    if len(g) == 0: return 'undet'
    H = set(g.applicant_cd)
    if any(same(r.firm, h) for h in H):
        ok = any(sim(r.nm, p) >= 0.6 for x in g.ainfo.dropna() for p in re.split(r',\s(?=[^,]{6,})|,(?=\S)', str(x))) or any(sim(fname.get(r.firm, ''), p) >= 0.6 for x in g.ainfo.dropna() for p in str(x).split(','))
        # 쉼표 분할이 이름을 자를 수 있어 전체 문자열에 대한 포함도 허용
        ok = ok or any(clean(r.nm) and clean(r.nm) in clean(x) for x in g.ainfo.dropna())
        return 'correct' if ok else 'wrong'
    # 법인이 보유자가 아님: 다른 보유자의 번호가 법인과 다르고 그 보유자 이름(hinfo 안)이 문서 이름과 비슷하면 오매칭
    for h in H:
        if corp.get(h) and corp.get(r.firm) and corp.get(h) != corp.get(r.firm):
            hn = g[g.applicant_cd == h].hinfo.dropna()
            if any(sim(r.nm, p) >= 0.8 for x in hn for p in str(x).split(',')): return 'wrong'
    if any(clean(r.nm) and clean(r.nm) in clean(x) for x in g.ainfo.dropna()): return 'correct'
    return 'undet'


docs['iv'] = [doc_v(r) for r in docs.itertuples(index=False)]
pv = docs.groupby(['firm', 'k']).iv.agg(lambda v: 'correct' if (v == 'correct').any() else ('wrong' if (v == 'wrong').all() else 'undet')).rename('indep')
sub = sub.join(pv, on=['firm', 'k']); sub['indep'] = sub.indep.fillna('undet(문서 없음)')
cats = ['correct', 'wrong', 'undet']
x = sub.pv.values; yv = np.where(sub.indep.str.startswith('undet'), 'undet', sub.indep.values)
agree = float(np.mean(x == yv))
cm = pd.crosstab(pd.Categorical(x, cats), pd.Categorical(yv, cats), dropna=False)
po = np.trace(cm.values) / cm.values.sum(); pe = (cm.values.sum(1) * cm.values.sum(0)).sum() / cm.values.sum() ** 2
kappa = (po - pe) / (1 - pe)
# 판정 가능(둘 다 undet 아님) 쌍에서의 정확/오매칭 일치
both = (x != 'undet') & (yv != 'undet')
out = dict(n_sub=int(len(sub)), docs=int(len(docs)), docs_per_pair_check=dict(indep=int(len(docs)), main_sum_n_docs=int(sub.n_docs.sum())),
           agree=agree, kappa=float(kappa), confusion=cm.astype(int).to_dict(), both_determinable=int(both.sum()), agree_both_det=float(np.mean(x[both] == yv[bool(1)*both])) if both.any() else None,
           rl_pat_coverage_of_hits=float(docs.appl_no.isin(set(rlh.appl_no)).mean()), verdict=('수용' if kappa >= 0.6 else '경고'), secs=round(time.time() - t0), peak=G.rss_peak())
sub[['cell', 'firm', 'k', 'n_docs', 'pv', 'indep', 'key_doc_verdict', 'name_raw']].to_csv(OUT/'v_indep_pairs.csv', index=False, encoding='utf-8-sig')
json.dump(out, open(OUT/'v_indep_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(out, ensure_ascii=False, indent=1, default=str))
