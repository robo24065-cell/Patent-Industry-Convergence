# verify_1/r1: 발견 '특허 법인 벤처는 대부분 재확인' 반박 시도. 원천(내려받은 CSV, CORP_APPLICANT, kr_patents)에서 독립 재계산.
# 사용: python -I r1_recheck.py
import sys, re, io, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, duckdb
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
W = B + r'\분석\검증루프_20261012\C06\C06-T4'
O = W + r'\verify_1'
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
PUN = re.compile(r"[\s\(\)\[\]·\.,\-&'\"/]")
nz = lambda s: PUN.sub('', SUF.sub('', str(s or '')).upper())
out = {}
# 1) 벤처 원본
v = pd.read_csv(io.StringIO(open(W + r'\download\venture_20260521\venture.bin', 'rb').read().decode('utf-8-sig')), dtype=str)
v = pd.DataFrame({'n': v['업체명'].map(nz), 's': v['벤처유효시작일'].str.replace('-', ''), 'e': v['벤처유효종료일'].str.replace('-', ''),
                  'nr': v['신규_재확인'].str.strip()})
out['rows'] = len(v); out['nr'] = v.nr.value_counts().to_dict()
mon = v.assign(m=v.s.str[:6]).groupby(['m', 'nr']).size().unstack(fill_value=0)
mon.to_csv(O + r'\r1_monthly.csv', encoding='utf-8-sig')
pre = mon.loc[(mon.index >= '202305') & (mon.index <= '202402'), '재확인']
post = mon.loc[mon.index >= '202403', '재확인']
out['re_pre_min_max'] = [int(pre.min()), int(pre.max())]
out['re_post_min_max'] = [int(post.min()), int(post.max())]
out['re_post_min_max_excl_202604'] = [int(post[post.index < '202604'].min()), int(post[post.index < '202604'].max())]
# 일 단위: 2024-02 재확인 시작일 분포(2021-02-12 시행 + 3년 → 2024-02-12 이후 집중?)
d2402 = v[(v.nr == '재확인') & (v.s.str[:6] == '202402')].s
out['re_202402_before_0212'] = int((d2402 < '20240212').sum()); out['re_202402_from_0212'] = int((d2402 >= '20240212').sum())
# 2) CA 유일 법인 상호
rows = []
with open(B + r'\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt', encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip('\r\n').split('¶')
        if len(p) >= 5:
            rows.append((nz(p[1].strip()), re.sub(r'\D', '', p[4])))
ca = pd.DataFrame(rows, columns=['n', 'biz'])
ca = ca[ca.n.str.len() >= 2]
ca['hb'] = ca.biz.str.len() == 10
ca['b6'] = np.where(ca.hb, ca.biz.str[:6], None)
ca['ct'] = ca.hb & ca.biz.str[3:5].between('81', '88')
g = ca.groupby('n').agg(c=('hb', 'size'), nb=('hb', 'sum'), u6=('b6', 'nunique'), ct=('ct', 'max'))
ok = set(g[(g.c == g.nb) & (g.u6 == 1) & g.ct].index)
out['ok_names'] = len(ok)
# 3) 문서 첫 노출
k = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원번호', '공개일자', '등록일자', '출원인명']).drop_duplicates('출원번호')
for c in ('공개일자', '등록일자'):
    k[c] = k[c].fillna('').astype(str).str.strip(); k.loc[k[c].str.len() != 8, c] = '99999999'
k['fv'] = np.where(k['공개일자'] <= k['등록일자'], k['공개일자'], k['등록일자'])
k = k[k.fv != '99999999']
k['n'] = k['출원인명'].map(nz)
k = k[k.n.isin(ok)]
fv = k.groupby('n').fv.min()
out['doc_names'] = int(len(fv)); out['fv_max'] = str(fv.max())
# 4) 정의별 신규 비율
v['ok'] = v.n.isin(ok); v['doc'] = v.n.isin(fv.index); v['fv'] = v.n.map(fv)
def share(m, lab):
    sub = v[m]; rest = v[~m]
    out[lab] = dict(rows=int(len(sub)), share_list=round(len(sub) / len(v), 4), new_share=round(float((sub.nr == '신규').mean()), 4),
                    new_share_rest=round(float((rest.nr == '신규').mean()), 4))
share(v.ok, 'D1_CA_unique_corp_name_any')
share(v.doc, 'D2_doc_visible_any_time')
share(v.doc & (v.fv < v.s), 'D3_doc_visible_before_start')
share(v.doc & (v.fv >= v.s), 'D3b_doc_first_visible_on_after_start')
# 5) 원점 위험집합(s1 규칙): fv<=t0, t0에 명단상 유효 없음
T = ['20230430', '20231031', '20240430', '20241031', '20250430', '20251031']
E = ['20231031', '20240430', '20241031', '20250430', '20251031', None]
union = set(); po = []
vd = v[v.doc]
for a, b in zip(T, E):
    names = set(fv[fv <= a].index)
    valid = set(vd[(vd.s <= a) & (vd.e >= a)].n)
    R = names - valid
    union |= R
    if b:
        w = vd[(vd.s > a) & (vd.s <= b)]
        pn = len(R & set(w[w.nr == '신규'].n)); pa = len(R & set(w.n))
        uv = len(R & set(vd[(vd.nr == '재확인') & (vd.s > a)].n)) / len(R)
        # 대안 진단: t0 이후 '재확인'이 t0+36개월 이내(종전 확인이 t0에 유효했을 수 있는 범위)
        po.append(dict(t0=a, n=len(R), pos_new=pn, pos_any=pa, unknown_valid=round(uv, 4)))
    else:
        po.append(dict(t0=a, n=len(R)))
out['origins'] = po
pa_ = [x['pos_any'] for x in po if 'pos_any' in x]
out['pos_any_ratio_max_min'] = round(max(pa_) / min(pa_), 2)
share(v.n.isin(union), 'D4_union_risk_sets(finding)')
# 6) 연결 벤처 중 신규의 첫 노출 vs 시작일: D2 중 신규인 행이 D4에 빠지는 이유
d2new = v[v.doc & (v.nr == '신규')]
out['D2_new_rows'] = int(len(d2new)); out['D2_new_in_union'] = int(d2new.n.isin(union).sum())
out['D2_new_fv_after_start'] = int((d2new.fv >= d2new.s).sum())
# 7) 시작연도별 신규 비율(D2 vs 나머지) — 시점 구성 영향
for lab, m in [('D2', v.doc), ('rest', ~v.doc)]:
    sub = v[m]
    out['by_year_new_share_' + lab] = sub.groupby(sub.s.str[:4]).nr.apply(lambda s: round(float((s == '신규').mean()), 3)).to_dict()
json.dump(out, open(O + r'\r1_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
