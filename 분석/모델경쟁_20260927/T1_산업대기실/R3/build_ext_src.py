# T1 R3 — 새 외부 연계 원천표 구축(원점 무관, 1회). 모든 DB는 읽기 전용.
#  (a) KIPRIS 디자인권(30*)·범위밖 실용신안(20*) — 자체 수집 business_rights.right_list(법인 53,967개 포트폴리오)
#      → R3/cache/src_design.parquet, src_um.parquet (applicant_cd, rgst_no, appl_date, rgst_date, ay, title)
#  (b) 특허권 이전(양도·합병 등) — 자체 수집 registry_full.owner(권리자 이력) → 새 권리자명을 법인 별칭 사전으로 코드화
#      → R3/cache/src_acq.parquet (applicant_cd, rgst_no, ksic, date, reason) : 원출원인이 아닌 법인이 범위안 특허를 넘겨받은 사건
#  (c) 전용·통상실시권 설정(라이선스) — registry_full.elsLcnsList/noelsLcnsList → R3/cache/src_lic.parquet
import sys, pathlib, re, time, sqlite3
R3 = pathlib.Path(__file__).resolve().parent; T1 = R3.parent; sys.path.insert(0, str(T1))
import numpy as np, pandas as pd
from common import CACHE, B
t0 = time.time(); OUT = R3/'cache'
def ro(p): return sqlite3.connect((B/p).as_uri() + '?mode=ro', uri=True)

fm = pd.read_parquet(CACHE/'firm_meta.parquet'); F = set(fm.applicant_cd)
inscope = set(pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no']).rgst_no.astype(str))

# (a) 디자인·실용신안
if not (OUT/'src_design.parquet').exists():
    c = ro('수집파이프라인/business_rights.sqlite')
    for pre, nm in (('30', 'design'), ('20', 'um')):
        d = pd.read_sql(f"select applicant_cd, rgst_no, appl_date, rgst_date, title from right_list where substr(rgst_no,1,2)='{pre}'", c)
        d = d[d.applicant_cd.isin(F)]; d = d[~d.rgst_no.astype(str).isin(inscope)]
        d['title'] = d.title.fillna('').str.strip(); d = d[d.title.str.len() > 0]
        d['rgst_date'] = d.rgst_date.fillna('').astype(str); d['appl_date'] = d.appl_date.fillna('').astype(str)
        d = d[(d.rgst_date.str.len() == 8)]
        d['ay'] = pd.to_numeric(d.appl_date.str[:4], errors='coerce').fillna(0).astype(int)
        d = d.drop_duplicates(['applicant_cd', 'rgst_no'])
        d.to_parquet(OUT/f'src_{nm}.parquet', index=False)
        print(nm, len(d), 'firms', d.applicant_cd.nunique(), 'rgst range', d.rgst_date.min(), d.rgst_date.max(), round(time.time() - t0), flush=True)

# 법인 별칭 사전(prep_app.py와 같은 정규화)
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
norm = lambda s: STRIP.sub('', str(s)).lower()
firms = pd.read_parquet(CACHE/'firms.parquet'); firms = firms[firms.status == 'OK']
app = pd.read_parquet(CACHE/'ins_applicant.parquet')
al = pd.concat([firms[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                app[app.applicantCd.isin(firms.applicant_cd)][['applicantCd', 'applicantName']].rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})])
al['n'] = al.nm.map(norm); al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
amb = al.groupby('n').applicant_cd.nunique(); al = al[al.n.map(amb) == 1]
n2c = dict(zip(al.n, al.applicant_cd))

pat = pd.read_parquet(CACHE/'ins_pat.parquet', columns=['rgst_no', 'ksic', 'rgst_date']).drop_duplicates('rgst_no'); pat['rgst_no'] = pat.rgst_no.astype(str)
orig = app.groupby('rgst_no').applicantCd.agg(lambda s: set(s)).to_dict()

# (b) 권리 이전
if not (OUT/'src_acq.parquet').exists():
    c = ro('수집파이프라인/registry_full.sqlite')
    o = pd.read_sql("select rgst_no, seq, ownerName, ownerRgstCsDate, ownerRgstCsName, ownerRgstCsReason from owner where seq > 0", c)
    n_all = len(o)
    o['applicant_cd'] = o.ownerName.map(norm).map(n2c)
    o = o.dropna(subset=['applicant_cd']); o['rgst_no'] = o.rgst_no.astype(str)
    o = o[[a not in orig.get(r, set()) for r, a in zip(o.rgst_no, o.applicant_cd)]]      # 원출원인 자신(명칭 변경 등)은 제외
    o['date'] = o.ownerRgstCsDate.fillna('').astype(str); o = o[o.date.str.len() == 8]
    o = o.merge(pat[['rgst_no', 'ksic']], on='rgst_no')
    o = o.rename(columns={'ownerRgstCsName': 'cs', 'ownerRgstCsReason': 'reason'})[['applicant_cd', 'rgst_no', 'ksic', 'date', 'cs', 'reason']]
    o = o.sort_values('date').drop_duplicates(['applicant_cd', 'rgst_no'])
    o.to_parquet(OUT/'src_acq.parquet', index=False)
    print('acq owner rows seq>0', n_all, 'matched to firm & not original', len(o), 'firms', o.applicant_cd.nunique(), 'date range', o.date.min(), o.date.max(), round(time.time() - t0), flush=True)
    print(o.cs.value_counts().head(8).to_dict())

# (c) 실시권
if not (OUT/'src_lic.parquet').exists():
    c = ro('수집파이프라인/registry_full.sqlite')
    L = []
    for t, p in (('elsLcnsList', 'els'), ('noelsLcnsList', 'noels')):
        x = pd.read_sql(f"select rgst_no, {p}LcnsPsnNm nm, {p}LcnsStartDate sd from {t} where {p}LcnsPsnNm is not null", c); x['kind'] = p; L.append(x)
    x = pd.concat(L); n_all = len(x)
    x['applicant_cd'] = x.nm.map(norm).map(n2c); x = x.dropna(subset=['applicant_cd']); x['rgst_no'] = x.rgst_no.astype(str)
    x = x[[a not in orig.get(r, set()) for r, a in zip(x.rgst_no, x.applicant_cd)]]
    x['date'] = x.sd.fillna('').astype(str).str.replace('-', '').str.replace('.', '').str[:8]
    x = x.merge(pat[['rgst_no', 'ksic', 'rgst_date']], on='rgst_no')
    x['date'] = np.where(x.date.str.len() == 8, x.date, x.rgst_date.astype(str))   # 시작일 없으면 등록일(보수적이지 않음 → 사용 시 날짜 결측분 제외 옵션)
    x['has_date'] = x.sd.fillna('').str.len() >= 8
    x = x[['applicant_cd', 'rgst_no', 'ksic', 'date', 'kind', 'has_date']].drop_duplicates(['applicant_cd', 'rgst_no'])
    x.to_parquet(OUT/'src_lic.parquet', index=False)
    print('lic rows', n_all, 'matched', len(x), 'firms', x.applicant_cd.nunique(), 'has_date', int(x.has_date.sum()), round(time.time() - t0), flush=True)
print('done', round(time.time() - t0))
