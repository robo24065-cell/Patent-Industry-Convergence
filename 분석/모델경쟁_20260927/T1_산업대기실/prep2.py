# 2단계: 자사출원 필터 + 범위안/범위밖 분리 + 필터 정확도 점검(범위안 특허로 코드 기준 진실값 비교)
import pandas as pd, numpy as np, re, time
from common import *
t0 = time.time()
kr = pd.read_parquet(CACHE/'ins_pat.parquet')
app = pd.read_parquet(CACHE/'ins_applicant.parquet')
firms = pd.read_parquet(CACHE/'firms.parquet')
firms = firms[firms.status == 'OK']
rl = pd.read_parquet(CACHE/'rl_pat.parquet')
rl = rl[rl.pert == '권리자란'].copy()
rl['ay'] = pd.to_numeric(rl.appl_date.str[:4], errors='coerce')

STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
def norm(s):
    return STRIP.sub('', str(s)).lower()

# 별칭: 조회명 + 레지스트리에서 같은 applicantCd로 기록된 이름
al = pd.concat([firms[['applicant_cd','applicant_name']].rename(columns={'applicant_name':'nm'}),
                app[app.applicantCd.isin(firms.applicant_cd)][['applicantCd','applicantName']].rename(columns={'applicantCd':'applicant_cd','applicantName':'nm'})])
al['n'] = al.nm.map(norm)
al = al[al.n.str.len() >= 2].drop_duplicates(['applicant_cd','n'])
alias = al.groupby('applicant_cd').n.apply(list).to_dict()
print('alias built', len(alias), time.time()-t0, flush=True)

def is_self(cd, ainfo):
    a = norm(ainfo)
    for n in alias.get(cd, []):
        if len(n) >= 3 and n in a: return True
        if len(n) < 3 and any(norm(p) == n for p in str(ainfo).split(',')): return True
    return False
rl['self'] = [is_self(c, a) for c, a in zip(rl.applicant_cd.values, rl.ainfo.values)]
print('self-filter done', rl.self.mean().round(4), time.time()-t0, flush=True)

insset = set(kr.rgst_no)
rl['ins'] = rl.rgst_no.isin(insset)
# 필터 정확도: 범위안 특허에서 코드 진실값과 비교
truth = set(zip(app.rgst_no, app.applicantCd))
r_in = rl[rl.ins].copy()
r_in['truth'] = [(r, c) in truth for r, c in zip(r_in.rgst_no, r_in.applicant_cd)]
tp_ = (r_in.self & r_in.truth).sum(); prec = tp_/max(r_in.self.sum(),1); rec = tp_/max(r_in.truth.sum(),1)
print(f'[필터 점검 범위안] rows={len(r_in)} truth_self={r_in.truth.mean():.4f} name_self={r_in.self.mean():.4f} precision={prec:.4f} recall={rec:.4f}')
r_in2 = r_in[r_in.ay >= 2000]
tp2 = (r_in2.self & r_in2.truth).sum()
print(f'[필터 점검 범위안, 2000+] rows={len(r_in2)} precision={tp2/max(r_in2.self.sum(),1):.4f} recall={tp2/max(r_in2.truth.sum(),1):.4f}')

oos = rl[(~rl.ins) & (rl.rgst_date <= SNAP_CUT)][['applicant_cd','rgst_no','ay','appl_date','rgst_date','title','tp','self','ainfo']].copy()
print('out-of-scope rows (reg<=cut)', len(oos), 'unique', oos.rgst_no.nunique(), 'self', oos.self.sum(), 'self&ay>=2000', (oos.self & (oos.ay>=2000)).sum())
print('after-snapshot rows dropped (not in-scope, reg>cut)', ((~rl.ins) & (rl.rgst_date > SNAP_CUT)).sum())
oos.to_parquet(CACHE/'oos.parquet', index=False)

# 범위안: 원출원인 코드 기준 법인-특허
ins = app[app.applicantCd.isin(firms.applicant_cd)][['rgst_no','applicantCd']].rename(columns={'applicantCd':'applicant_cd'})
ins = ins.merge(kr[['rgst_no','ksic','ay','appl_date','rgst_date','title','mipc']], on='rgst_no').drop_duplicates(['rgst_no','applicant_cd'])
ins.to_parquet(CACHE/'ins_firm.parquet', index=False)
print('in-scope firm-patent rows', len(ins), 'firms', ins.applicant_cd.nunique())
# 법인 속성: 유형(국내/외국 등) 최빈값, 법인등록번호
ft = rl.groupby('applicant_cd').tp.agg(lambda x: x.value_counts().index[0]).rename('tp')
cr = rl.dropna(subset=['corp_rgstno']).groupby('applicant_cd').corp_rgstno.max()
fm = firms.set_index('applicant_cd').join(ft).join(cr).reset_index()
fm.to_parquet(CACHE/'firm_meta.parquet', index=False)
print(fm.tp.value_counts().head(8)); print('time', time.time()-t0)
