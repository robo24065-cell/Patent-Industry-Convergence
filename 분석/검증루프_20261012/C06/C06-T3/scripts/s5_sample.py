# -*- coding: utf-8 -*-
"""C06-T3 s5: 복원 연결 정밀도 표본(M 시나리오 쌍 100 + U 전용 쌍 40) 근거표 생성 — 수기 판정용."""
import pandas as pd, numpy as np, sys, io, duckdb, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
SH = ROOT + r"\분석\모델경쟁_20260927\_shared\ext"
OUT = ROOT + r"\분석\검증루프_20261012\C06\C06-T3"; CA = OUT + r"\cache"
cand = pd.read_parquet(CA + r"\n6_cand.parquet"); p1 = pd.read_parquet(CA + r"\n6_p1_pairs.parquet")
firm = pd.read_parquet(CA + r"\n6_firm.parquet")
p1['window'] = p1.ad.between(20220101, 20231231); p1['known_le2021'] = p1.known <= 20211231
ok = cand[~cand.rl_other_only]
Mh = ok[ok.addr_sg_eq & ~ok.comp_same_sg].assign(path='P2h-M')
Uo = ok[~(ok.addr_sg_eq & ~ok.comp_same_sg)].assign(path='P2h-U only')
P1 = p1.assign(path='P1')
M = pd.concat([P1, Mh], ignore_index=True)
M = M[M.window | M.known_le2021]
sM = M.sample(100, random_state=0)
sU = Uo.sample(40, random_state=0)
s = pd.concat([sM.assign(sample='M100'), sU.assign(sample='U40')], ignore_index=True)
P = (SH + r"\ntis_project_raw.parquet").replace("\\", "/")
bzs = "','".join(s.bz.unique())
tt = duckdb.sql(f"""select regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, title_ko, research_agency from read_parquet('{P}')
     where regexp_replace(corp_rgst_no,'[^0-9]','','g') in ('{bzs}') and year::int between 2017 and 2021""").df()
tts = tt.groupby('bz').title_ko.apply(lambda x: ' / '.join(list(dict.fromkeys(x))[:3])).to_dict()
s = s.merge(firm[['bz', 'ntis_name', 'f_sido', 'f_sg', 'ldong', 'n_emp', 'grp']].rename(columns={'f_sido': 'fs', 'f_sg': 'fg', 'grp': 'g2', 'n_emp': 'emp'}), on='bz', how='left')
s['ntis_titles'] = s.bz.map(tts)
cols = ['sample', 'path', 'bz', 'cd', 'aname', 'n_same', 'ntis_name', 'names', 'addr', 'fs', 'fg', 'comp_addrs', 'rl_holders', 'ad', 'known', 'ksic', 'title', 'ntis_titles', 'emp']
s = s[[c for c in cols if c in s.columns]]
s.to_csv(OUT + r"\cache\precision_sample_raw.csv", index=False, encoding='utf-8-sig')
for i, r in s.iterrows():
    print(f"[{i}] {r['sample']} {r.path} | 코드상호={r.aname} (동명{r.n_same}) | NTIS상호={r.ntis_name} | 문서출원인={r.names} | 문서주소={str(r.addr)[:28]} | 기업주소={r.fs} {r.fg} | 경쟁주소={str(r.comp_addrs)[:40]} | 권리목록={r.rl_holders} | 출원={r.ad} ksic={r.ksic}")
    print(f"     발명={str(r.title)[:50]} || 과제={str(r.ntis_titles)[:110]}")
