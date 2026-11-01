# -*- coding: utf-8 -*-
"""s03: 범위안 특허(등록원부에 원출원인 코드·권리자 이력이 있음)를 아날로그로 써서 right_list(2026 스냅샷)의 성격을 확인.
 (1) 원출원인 F가 넘긴(최종 권리자에 F 없음) 특허가 F의 right_list에 남아 있는가 → 목록이 '현재 보유자' 기준인지(생존 편향)
 (2) right_list에 F 이름으로 있지만 F가 원출원인이 아닌 특허(양수분)의 이전등록일 분포 → 원점 이후 이전 비율
 (3) 범위안 특허의 이전 유출률(원출원인 F인데 최종 권리자가 다른 법인) 등록연도별
 → cache/s03_out.json, cache/acq_pairs.parquet((원출원인 코드 A, 양수 법인 F) 최초 이전일 — S2 이전일 추정용)"""
import pathlib, re, time, json, sqlite3
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent.parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; CACHE = T1/'cache'
t0 = time.time(); OUT = {}
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
def norm(s): return STRIP.sub('', str(s)).lower()
firms = pd.read_parquet(CACHE/'firms.parquet'); ok = firms[firms.status == 'OK']; OKS = set(ok.applicant_cd)
app = pd.read_parquet(CACHE/'ins_applicant.parquet', columns=['rgst_no', 'applicantCd', 'applicantName']); app['rgst_no'] = app.rgst_no.astype(str)
al = pd.concat([ok[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                app[app.applicantCd.isin(OKS)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
alias = al.groupby('applicant_cd').n.apply(list).to_dict()
def has_alias(cd, txt):
    a = norm(txt)
    for n in alias.get(cd, []):
        if len(n) >= 3 and n in a: return True
        if len(n) < 3 and any(norm(p) == n for p in str(txt).split(',')): return True
    return False
con = sqlite3.connect((B/'수집파이프라인/registry_full.sqlite').as_uri() + '?mode=ro', uri=True)
ow = pd.read_sql('select rgst_no, seq, finalOwnerYn, ownerName, ownerRgstCsDate, ownerRgstCsName from owner', con); ow['rgst_no'] = ow.rgst_no.astype(str)
print('owner rows', len(ow), round(time.time() - t0), flush=True)
maxseq = ow.groupby('rgst_no').seq.max()
fin = ow[ow.seq == ow.rgst_no.map(maxseq)].groupby('rgst_no').ownerName.apply(lambda s: ','.join(map(str, s)))
chg = ow[ow.seq > 0].groupby('rgst_no').ownerRgstCsDate.min()          # 첫 권리자 변경일
rl = pd.read_parquet(CACHE/'rl_pat.parquet', columns=['applicant_cd', 'rgst_no', 'pert', 'rgst_date']); rl['rgst_no'] = rl.rgst_no.astype(str)
rl = rl[rl.pert == '권리자란']; rlset = set(zip(rl.applicant_cd, rl.rgst_no))
pat = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no', 'rgst_date']).drop_duplicates('rgst_no'); pat['rgst_no'] = pat.rgst_no.astype(str)
# (1)(3) 원출원인 F(OK 법인)의 범위안 특허
x = app[app.applicantCd.isin(OKS)][['rgst_no', 'applicantCd']].drop_duplicates()
x = x.merge(pat, on='rgst_no', how='inner')
x['final'] = x.rgst_no.map(fin).fillna('')
x['has_owner_rec'] = x.rgst_no.isin(maxseq.index)
x = x[x.has_owner_rec].copy()
x['retained'] = [has_alias(c, f) for c, f in zip(x.applicantCd.values, x.final.values)]
x['in_rl'] = [(c, r) in rlset for c, r in zip(x.applicantCd.values, x.rgst_no.values)]
x['ry'] = x.rgst_date.astype(str).str[:4].astype(int)
x['chg_date'] = x.rgst_no.map(chg)
print('analog rows', len(x), round(time.time() - t0), flush=True)
OUT['analog_n'] = int(len(x))
OUT['retained_share'] = float(x.retained.mean())
OUT['in_rl_if_retained'] = float(x[x.retained].in_rl.mean())
OUT['in_rl_if_transferred_away'] = float(x[~x.retained].in_rl.mean())
OUT['n_transferred_away'] = int((~x.retained).sum())
by = x.groupby('ry').agg(n=('retained', 'size'), away=('retained', lambda s: float(1 - s.mean())), in_rl=('in_rl', 'mean'))
OUT['away_by_rgst_year'] = {int(k): dict(n=int(v.n), away=round(v.away, 4), in_rl=round(float(v.in_rl), 4)) for k, v in by.iterrows() if 2000 <= k <= 2025}
# 넘긴 시점이 원점 2019 이후인 비율(넘긴 특허 중)
aw = x[~x.retained]
OUT['away_after_2019_share_of_away'] = float((aw.chg_date.fillna('0') > '20191231').mean())
# (2) right_list 상 F 보유인데 F가 원출원인 아님(양수분) — 이전등록일 분포
orig = app.groupby('rgst_no').applicantCd.agg(lambda s: set(s)).to_dict()
r2 = rl[rl.rgst_no.isin(pat.rgst_no)].copy()
r2['is_orig'] = [a in orig.get(r, set()) for a, r in zip(r2.applicant_cd.values, r2.rgst_no.values)]
OUT['rl_inscope_rows'] = int(len(r2)); OUT['rl_inscope_nonorig_share'] = float(1 - r2.is_orig.mean())
acq = r2[~r2.is_orig].copy()
# 그 법인으로의 이전 기록(새 권리자명 별칭 매칭) 최초일
ow2 = ow[ow.seq > 0][['rgst_no', 'ownerName', 'ownerRgstCsDate', 'ownerRgstCsName']]
acq = acq.merge(ow2, on='rgst_no', how='left')
acq['m'] = [isinstance(n, str) and has_alias(c, n) for c, n in zip(acq.applicant_cd.values, acq.ownerName.values)]
first = acq[acq.m].groupby(['applicant_cd', 'rgst_no']).ownerRgstCsDate.min()
k = r2[~r2.is_orig].set_index(['applicant_cd', 'rgst_no'])
k['tdate'] = first.reindex(k.index).values
k = k.reset_index()
OUT['acq_rows'] = int(len(k)); OUT['acq_dated_share'] = float(k.tdate.notna().mean())
kd = k.dropna(subset=['tdate']); kd['ty'] = kd.tdate.astype(str).str[:4].astype(int)
OUT['acq_transfer_year_dist'] = {int(a): int(b) for a, b in kd.ty.value_counts().sort_index().items()}
OUT['acq_after_2019_share_dated'] = float((kd.tdate > '20191231').mean())
# 원점별: 원점 연말 이전 등록된 양수분 중 이전일 > 원점 연말 비율
for c in (2012, 2013, 2014, 2015, 2019):
    v = k[k.rgst_date.astype(str) <= f'{c}1231']
    OUT[f'acq_reg_le_{c}_n'] = int(len(v))
    OUT[f'acq_reg_le_{c}_transfer_after_or_undated'] = float(((v.tdate.isna()) | (v.tdate > f'{c}1231')).mean())
    OUT[f'acq_reg_le_{c}_transfer_after_dated'] = float((v.dropna(subset=['tdate']).tdate > f'{c}1231').mean()) if v.tdate.notna().any() else None
# (A 원출원인 코드, F 양수 법인) 최초 이전일 표 — 범위안 특허 이전 기록(동결 src_acq와 같은 정의를 다시 만듦)
pairs = []
for (f, r), td in first.items():
    for a in orig.get(r, set()):
        if a != f: pairs.append((a, f, td))
pp = pd.DataFrame(pairs, columns=['orig_cd', 'applicant_cd', 'tdate']).groupby(['orig_cd', 'applicant_cd']).tdate.agg(['min', 'size']).reset_index()
pp.columns = ['orig_cd', 'applicant_cd', 'tdate_min', 'n_pat']
pp.to_parquet(HERE/'cache/acq_pairs.parquet', index=False)
OUT['acq_pairs'] = int(len(pp))
json.dump(OUT, open(HERE/'cache/s03_out.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k2: v for k2, v in OUT.items() if not isinstance(v, dict)}, ensure_ascii=False, indent=1))
print('done', round(time.time() - t0))
