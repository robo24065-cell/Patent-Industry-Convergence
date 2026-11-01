# -*- coding: utf-8 -*-
"""verify_0: (1) 번호 형식·(2) 연결 정확도를 독립 코드로 재현·반박 시도.
- duckdb·이 폴더 cache 를 쓰지 않는다. pyarrow 로 _shared/meta 원본을 파일별로 읽고 파이썬 re 로 토큰을 분류한다.
- (1) 형식 분포: 모든 국가 family 의 US 토큰을 독자 정규식으로 분류, 코드 키('US'+7~8자리 full match)와 비교.
- (2) 정확도: 공유 r3ros_family_us_pairs 에서 다른 시드(7)로 비US family 쌍 국가별 3,000개를 뽑아
      '역방향': P 를 family 에 가진 US 레코드(자기 제외) 중 비US 토큰을 k 와 공유하는 것이 있나 / '우선일': 날짜 집합 공유 / 순열 기준선.
- 반박 시도: (a) 우선일만으로 정밀도, (b) 자기 토큰(KR 공개번호) 기반만으로 정밀도, (c) 초대형 패밀리(US 등록 토큰 >50) 제외 시 정밀도.
"""
import os, re, sys, json, glob, random, time, collections
sys.dont_write_bytecode = True
import pyarrow.parquet as pq, pandas as pd, numpy as np
BASE = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
MC = BASE + "/분석/모델경쟁_20260927"
OUT = BASE + "/분석/검증루프_20261012/C11/C11-T3/verify_0"
t = time.time()
REG = re.compile(r'^US(\d{7,8})$')
def cls(tok):
    d = tok[2:]
    if d.isdigit():
        L = len(d)
        if L == 7: return 'R7'
        if L == 8: return 'R8z' if d[0] == '0' else ('R8n' if d[:2] in ('10', '11', '12') else 'R8x')
        if L == 10: return 'P10'
        if L == 11: return 'P11'
        return f'N{L}'
    for p in ('RE', 'PP', 'D', 'H'):
        if d.startswith(p): return p
    return 'other'
# 표본 쌍
pairs = pd.read_parquet(MC + "/_shared/ext/r3ros_family_us_pairs.parquet")
fam = pairs[(pairs.src == 'family') & (pairs.cc != 'US')]
smp = fam.groupby('cc', group_keys=False).apply(lambda g: g.sample(min(3000, len(g)), random_state=7)).reset_index(drop=True)
Pset = set(smp.us_pn.astype(int))
perm = smp.copy(); perm['us_pn'] = smp.groupby('cc').us_pn.transform(lambda s: s.sample(frac=1, random_state=11).values)
Pset |= set(perm.us_pn.astype(int))
need_k = set(zip(smp.cc, smp.appl_no))
cnt = collections.Counter(); codekey = collections.Counter(); recs_k = {}; usP = collections.defaultdict(list); usrec = {}
for f in sorted(glob.glob(MC + "/_shared/meta/*.parquet")):
    tb = pq.read_table(f, columns=['cc', 'appl_no', 'family', 'prio_date', 'appl_date']).to_pandas()
    tb = tb[tb.family.notna()]
    for cc, an, fm, pdx, ad in zip(tb.cc.values, tb.appl_no.values, tb.family.values, tb.prio_date.values, tb.appl_date.values):
        toks = set(fm.split('|'))
        ust = [x for x in toks if x.startswith('US')]
        for x in ust:
            c = cls(x); cnt[c] += 1
            if REG.match(x): codekey[c] += 1
        dates = set(d for d in (pdx or '').split('|') if len(d) == 8) | ({ad} if ad and len(ad) == 8 else set())
        if (cc, an) in need_k: recs_k[(cc, an)] = (toks, dates)
        if cc == 'US':
            regs = set(int(REG.match(x).group(1)) for x in ust if REG.match(x))
            hit = regs & Pset
            if hit:
                usrec[an] = (toks, dates, len(regs))
                for p in hit: usP[p].append(an)
print('scan done', round(time.time() - t), flush=True)
def evaluate(df):
    out = []
    for cc, an, p in zip(df.cc.values, df.appl_no.values, df.us_pn.astype(int).values):
        if (cc, an) not in recs_k: out.append((cc, None, None, None, None)); continue
        kt, kd = recs_k[(cc, an)]; knon = {x for x in kt if not x.startswith('US') and len(x) > 3}
        us = [u for u in usP.get(p, []) if not (cc == 'US' and u == an)]
        if not us: out.append((cc, False, None, None, None)); continue
        fam_ok = any(knon & {x for x in usrec[u][0] if not x.startswith('US')} for u in us)
        date_ok = any(kd & usrec[u][1] for u in us)
        big = max(usrec[u][2] for u in us) > 50
        out.append((cc, True, fam_ok, date_ok, big))
    return pd.DataFrame(out, columns=['cc', 'evaluable', 'fam_ok', 'date_ok', 'big'])
A = evaluate(smp); B = evaluate(perm)
res = dict(format_counts=dict(cnt), codekey_counts=dict(codekey))
for nm, X in (('link', A), ('perm', B)):
    e = X[X.evaluable == True]
    res[nm] = dict(n=len(X), evaluable=float((X.evaluable == True).mean()), precision=float((e.fam_ok | e.date_ok).mean()),
                   fam_only=float(e.fam_ok.mean()), date_only=float(e.date_ok.mean()),
                   precision_excl_big=float((e[~e.big.astype(bool)].fam_ok | e[~e.big.astype(bool)].date_ok).mean()), big_share=float(e.big.astype(bool).mean()),
                   by_cc={cc: round(float((g.fam_ok | g.date_ok).mean()), 4) for cc, g in e.groupby('cc')})
json.dump(res, open(OUT + "/v0_out.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(res, ensure_ascii=False, indent=1)); print('secs', round(time.time() - t))
