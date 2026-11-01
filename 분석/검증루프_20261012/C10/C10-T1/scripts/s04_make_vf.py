# -*- coding: utf-8 -*-
"""s04: S1·S2용 valid_from(정수 YYYYMMDD) 배열 — 동결 분류 확률 P_<kind>_index 행 순서 그대로.
 S1: 자사 행 = 등록일, 비자사 행 = 99999999(영구 제외)
 S2: 자사 행 = 등록일, 비자사 행 = max(등록일, 추정 이전일), 추정 불가 = 99999999
 추정 이전일: applicantInfo(원출원인 코드의 현재 이름) 각 이름 → 출원인코드(이름 사전, 모호하면 버림) A,
             등록원부 범위안 특허에서 A → 조회 법인 F 이전 기록 최초일(cache/acq_pairs.parquet).
 + 원점별 비자사 몫·S2 제외 몫 → cache/s04_vf_stats.json"""
import pathlib, re, json, time
import numpy as np, pandas as pd
HERE = pathlib.Path(__file__).resolve().parent.parent
B = pathlib.Path(r"C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전")
T1 = B/'분석/모델경쟁_20260927/T1_산업대기실'; CACHE = T1/'cache'; R3 = T1/'R3'
t0 = time.time()
STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|유한책임회사|재단법인|사단법인|학교법인|가부시키가이샤|가부시끼가이샤|가부시기가이샤|가부시키카이샤|코포레이션|인코포레이티드|리미티드|컴퍼니|[\s\.\,\-\(\)\'\"·&]")
def norm(s): return STRIP.sub('', str(s)).lower()
fl = pd.read_parquet(HERE/'cache/dsum_flags.parquet', columns=['applicant_cd', 'rgst_no', 'kind', 'ainfo', 'self', 'pert'])
fl = fl.drop_duplicates(['applicant_cd', 'rgst_no'])
# 이름 → 코드 사전(모든 조회 법인·개인 + 등록원부 원출원인 이름), 모호 이름 제거
firms = pd.read_parquet(CACHE/'firms.parquet')
app = pd.read_parquet(CACHE/'ins_applicant.parquet', columns=['applicantCd', 'applicantName'])
nm = pd.concat([firms[['applicant_cd', 'applicant_name']].rename(columns={'applicant_name': 'nm'}),
                app.rename(columns={'applicantCd': 'applicant_cd', 'applicantName': 'nm'})]).dropna()
nm['n'] = nm.nm.map(norm); nm = nm[nm.n.str.len() >= 2].drop_duplicates(['applicant_cd', 'n'])
amb = nm.groupby('n').applicant_cd.nunique(); nm = nm[nm.n.map(amb) == 1]; n2c = dict(zip(nm.n, nm.applicant_cd))
pp = pd.read_parquet(HERE/'cache/acq_pairs.parquet'); pd_ = {(a, f): t for a, f, t in zip(pp.orig_cd, pp.applicant_cd, pp.tdate_min)}
ns = fl[~fl.self].copy()
def est(f, ainfo):
    ds = []; mapped = False
    for p in str(ainfo).split(','):
        a = n2c.get(norm(p))
        if a is None: continue
        mapped = True
        if a == f: continue
        t = pd_.get((a, f))
        if t is not None and isinstance(t, str) and len(t) >= 8: ds.append(t[:8])
    return (min(ds) if ds else None), mapped
r = [est(f, a) for f, a in zip(ns.applicant_cd.values, ns.ainfo.values)]
ns['tdate'] = [x[0] for x in r]; ns['orig_mapped'] = [x[1] for x in r]
fl = fl.merge(ns[['applicant_cd', 'rgst_no', 'tdate', 'orig_mapped']], on=['applicant_cd', 'rgst_no'], how='left')
ST = {}
for kind in ('design', 'um'):
    x = fl[(fl.kind == kind) & (~fl.self)]
    ST[f'{kind}_nonself_rows'] = int(len(x)); ST[f'{kind}_nonself_orig_mapped'] = float(x.orig_mapped.mean()); ST[f'{kind}_nonself_dated'] = float(x.tdate.notna().mean())
    xd = x.dropna(subset=['tdate'])
    ST[f'{kind}_nonself_tyear_dist'] = {int(k): int(v) for k, v in xd.tdate.str[:4].astype(int).value_counts().sort_index().items()}
DIRS = {2019: R3/'cache_final'/'o2019'}; DIRS.update({c: R3/'cache'/f'o{c}' for c in (2008, 2009, 2010, 2011, 2012)})
CUTS = {2019: (2012, 2013, 2014, 2015, 2019)}; CUTS.update({c: tuple(range(c - 7, c - 3)) + (c,) for c in (2008, 2009, 2010, 2011, 2012)})
for c, FC in DIRS.items():
    for kind in ('design', 'um'):
        ix = pd.read_parquet(FC/f'P_{kind}_index.parquet'); P = np.load(FC/f'P_{kind}.npy').astype(np.float32)
        m = ix[['applicant_cd', 'rgst_no']].merge(fl[fl.kind == kind][['applicant_cd', 'rgst_no', 'self', 'tdate']], on=['applicant_cd', 'rgst_no'], how='left', validate='m:1')
        assert len(m) == len(ix) and m.self.notna().all(), (c, kind, int(m.self.isna().sum()))
        rg = ix.rgst_date.astype(str).str[:8].astype(np.int64).values; sf = m.self.values.astype(bool)
        td = pd.to_numeric(m.tdate, errors='coerce').fillna(99999999).astype(np.int64).values
        vf1 = np.where(sf, rg, 99999999).astype(np.int64)
        vf2 = np.where(sf, rg, np.maximum(rg, td)).astype(np.int64)
        np.save(HERE/'cache'/f'vf_S1_{kind}_o{c}.npy', vf1); np.save(HERE/'cache'/f'vf_S2_{kind}_o{c}.npy', vf2)
        mass = P.sum(1)  # 행당 1(확률 합) — 질량은 행 수와 같아서 목표 산업 질량 대신 max>0.2 행 몫도 기록
        hi = P.max(1) > 0.2
        for cut in CUTS[c]:
            ce = int(f'{cut}1231'); base = rg <= ce
            ST[f'o{c}_{kind}_cut{cut}'] = dict(rows=int(base.sum()), nonself_share=float((~sf[base]).mean()) if base.any() else None,
                                              s2_removed_share=float((vf2[base] > ce).mean()) if base.any() else None,
                                              s2_removed_share_of_nonself=float((vf2[base & ~sf] > ce).mean()) if (base & ~sf).any() else None,
                                              nonself_share_hi=float((~sf[base & hi]).mean()) if (base & hi).any() else None)
        print(c, kind, len(ix), round(time.time() - t0), flush=True)
json.dump(ST, open(HERE/'cache/s04_vf_stats.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in ST.items() if 'o2019' in k or 'nonself' in k and not isinstance(v, dict)}, ensure_ascii=False, indent=1))
