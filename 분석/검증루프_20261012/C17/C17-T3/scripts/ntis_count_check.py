# C17-T3 s2: NTIS 2026 스냅샷의 연도별 과제 수와 공식 과제 수(KIST 보고서 PDF·보도 원문) 대조 — grade_rule 2절 (ii) 비선택 근거 재계산.
# 원천 직접 재계산(작업자 단독 F33 대체용). 원천 PDF·HTML은 C02-T1 download(기존 공개 수집본)를 읽기만 한다.
import os, sys, re, json, glob
import pandas as pd, fitz
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석"
EXT = os.path.join(ROOT, "모델경쟁_20260927", "_shared", "ext")
DL = os.path.join(ROOT, "검증루프_20261012", "C02", "C02-T1", "download")
W = os.path.join(ROOT, "검증루프_20261012", "C17", "C17-T3")
d = pd.read_parquet(os.path.join(EXT, 'ntis_project_raw.parquet'), columns=['year', 'pjt_no', 'secret_flag'])
g = d.groupby('year').agg(n=('pjt_no', 'size'), n_uniq=('pjt_no', 'nunique'), n_secret=('secret_flag', 'sum')).reset_index()
g = g[g.year.between(2010, 2025)]
# 공식 값: 원문에서 직접 찾기
texts = {}
for f in glob.glob(os.path.join(DL, 'kist_pdf', '*.pdf')):
    doc = fitz.open(f); texts[os.path.basename(f)] = ''.join(p.get_text() for p in doc)
for f in glob.glob(os.path.join(DL, 'news_*', '*.html')):
    texts[os.path.basename(f)] = re.sub(r'<[^>]+>', ' ', open(f, encoding='utf-8', errors='replace').read())
cand = {2018: 63697, 2019: 70327, 2020: 73501, 2021: 74745, 2022: 76052, 2023: 71804, 2024: 60696}
found = {}
for y, v in cand.items():
    pats = [f'{v:,}', str(v), f'{v/10000:.4f}'.rstrip('0') + '만']
    found[y] = {k: any(p in t for p in pats[:2]) for k, t in texts.items()}
g['official_n'] = g.year.map(cand)
g['found_in_source'] = g.year.map(lambda y: ';'.join(k for k, v in found.get(y, {}).items() if v) or '')
g['ratio_n'] = g.n / g.official_n
g['ratio_uniq'] = g.n_uniq / g.official_n
g.to_csv(os.path.join(W, 'cache', 'ntis_count_check.csv'), index=False, encoding='utf-8-sig')
print(g.to_string())
ok = g[g.official_n.notna()]
print(json.dumps(dict(min_ratio=float(ok.ratio_n.min()), max_ratio=float(ok.ratio_n.max()), all_found=bool((ok.found_in_source != '').all())), ensure_ascii=False))
# 2010–2017 보강: 금액 포착률(비보안 정부연구비 + 보안 총연구비) / e-나라 1330 집행액(원문 파싱본) — 과제 수 공식값이 없는 해의 대체 근거
off = pd.read_csv(os.path.join(ROOT, "검증루프_20261012", "C02", "C02-T1", "official_enara_1330_long.csv"))
off = off[off.kind.isin(['amount', 'value'])].copy(); off['value'] = pd.to_numeric(off.value, errors='coerce'); off['year'] = off.year.astype(int)
tot = off[(off.stts_cd == 133002) & (off.item == '계')].set_index('year').value
d2 = pd.read_parquet(os.path.join(EXT, 'ntis_project_raw.parquet'), columns=['year', 'secret_flag', 'gov_funds', 'total_funds'])
a = (d2[~d2.secret_flag].groupby('year').gov_funds.sum() + d2[d2.secret_flag].groupby('year').total_funds.sum()) / 1e8
amt = pd.DataFrame(dict(collected_eok=a, official_exec_eok=tot)).dropna(); amt['ratio'] = amt.collected_eok / amt.official_exec_eok
amt = amt.loc[2010:2024]; amt.to_csv(os.path.join(W, 'cache', 'ntis_amount_check.csv'), encoding='utf-8-sig')
print(amt.round(4).to_string())
