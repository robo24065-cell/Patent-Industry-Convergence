# -*- coding: utf-8 -*-
"""verify_0 (b): N6 공란 복원 L/M/U를 별도 코드로 재계산.
s3/s4와 다른 점: 정규화 함수 별도 구현(문자 단위 필터), duckdb 조인, 시군구 대응표를 '등록원부 주소'가 아닌 '제공 KR 등록 문서 주소'로 만듦, AUC는 sklearn."""
import duckdb, pandas as pd, numpy as np, sys, io, json, re, sqlite3, unicodedata
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = ROOT + r"\분석\모델경쟁_20260927"; SH = MC + r"\_shared\ext"
OUT = ROOT + r"\분석\검증루프_20261012\C06\C06-T3"
o = {}
SUFS = ['주식회사', '유한책임회사', '유한회사', '합자회사', '합명회사', '농업회사법인', '영농조합법인', '재단법인', '사단법인', '(주)', '(유)', '(합)', '㈜', '（주）', '(株)', '株式会社']
SUF_RE = [re.compile(r'CO\.,?LTD\.?', re.I), re.compile(r'CORPORATION', re.I), re.compile(r'INC\.?', re.I)]
DROP = set(" ()[]·.,-&'\"/")
def norm2(s):
    s = str(s) if s is not None and s == s else ''
    for x in SUFS: s = s.replace(x, '')
    for r_ in SUF_RE: s = r_.sub('', s)
    s = s.upper()
    return ''.join(ch for ch in s if ch not in DROP and not ch.isspace())
nt = duckdb.sql(f"""select cast(year as int) y, regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, research_agency nm
   from read_parquet('{(SH + r"\ntis_project_raw.parquet").replace(chr(92), "/")}') where cast(year as int) between 2010 and 2021
   and perform_grp in ('중소기업','중견기업','대기업') and length(regexp_replace(corp_rgst_no,'[^0-9]','','g'))=10""").df()
pop = set(nt[nt.y >= 2019].bz)
st = pd.read_parquet(SH + r"\r3nts_applicant_status.parquet", columns=['applicantCd', 'bizno', 'aname'])
st['k'] = st.aname.map(norm2); vc = st.k.value_counts(); uniq = st[st.k.map(vc) == 1].set_index('k').applicantCd.to_dict()
mv = nt.groupby('bz').nm.agg(lambda s: s.value_counts().index[0]).to_dict()
by_bz = st.groupby('bizno').applicantCd.apply(list).to_dict()
L = []
for b in pop:
    if b in by_bz: L += [(b, c) for c in by_bz[b]]
    elif uniq.get(norm2(mv.get(b))): L.append((b, uniq[norm2(mv.get(b))]))
fl = pd.DataFrame(L, columns=['bz', 'cd'])
af = pd.read_parquet(MC + r"\_ideas_R3\cache\app_firm.parquet", columns=['applicantCd', 'appl_no', 'appl_date', 'known_date']).rename(columns={'applicantCd': 'cd'})
af = af[af.appl_date <= 20231231]
fa = duckdb.sql("select fl.bz, af.appl_no, af.appl_date, af.known_date from fl join af on fl.cd=af.cd").df()
kn = fa.groupby('bz').known_date.min()
risk = sorted(pop - set(kn[kn <= 20211231].index)); pos = set(fa[fa.appl_date >= 20220101].bz)
pr = pd.read_parquet(MC + r"\N6_정부RnD기업_첫특허전환\final_preds_r5.parquet"); pr['bz'] = pr.bz.astype(str).str.zfill(10)
o['official'] = dict(units_eq=set(risk) == set(pr.bz), y_eq=bool((pr.bz.isin(pos).astype(int) == pr.y).all()))
# CORP_APPLICANT 이름과 동명 수
ca = []
with open(ROOT + r"\공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt", encoding='utf-8') as f:
    next(f)
    for line in f:
        p = line.rstrip("\r\n").split("¶")
        if len(p) >= 2: ca.append((p[0].strip(), p[1].strip()))
ca = pd.DataFrame(ca, columns=['cd', 'nm']); ca['k'] = ca.nm.map(norm2); ca = ca[ca.k.str.len() >= 2]
nsame = ca.groupby('k').cd.nunique(); ca['ns'] = ca.k.map(nsame)
# 제공 KR
KP = ROOT + r"\공모전데이터\작업데이터\kr_patents.parquet"; NM = pq.ParquetFile(KP).schema_arrow.names
k = pq.read_table(KP, columns=[NM[i] for i in (0, 1, 3, 4, 6, 8, 9)]).to_pandas(); k.columns = ['appl_no', 'ad', 'rg', 'rd', 'pd', 'nm', 'addr']
k['appl_no'] = k.appl_no.astype(str); k['blank'] = k.rg.isna() | (k.rg.astype(str).str.strip() == '')
k['kd'] = pd.concat([pd.to_numeric(k[c].astype(str).str.replace(r'\D', '', regex=True).str[:8], errors='coerce') for c in ('pd', 'rd')], axis=1).min(axis=1)
k['k'] = k.nm.map(norm2)
def sido(t):
    t = t if isinstance(t, str) else ''
    for a, b in (('서울', '서울'), ('부산', '부산'), ('대구', '대구'), ('인천', '인천'), ('광주', '광주'), ('대전', '대전'), ('울산', '울산'), ('세종', '세종'), ('경기', '경기'), ('강원', '강원'),
                 ('충청북', '충북'), ('충북', '충북'), ('충청남', '충남'), ('충남', '충남'), ('전라북', '전북'), ('전북', '전북'), ('전라남', '전남'), ('전남', '전남'),
                 ('경상북', '경북'), ('경북', '경북'), ('경상남', '경남'), ('경남', '경남'), ('제주', '제주')):
        if t.startswith(a): return b
    return None
m_ = k.addr.fillna('').str.extract(r'^\s*(\S+)\s+(\S+)')
k['ds'] = m_[0].map(sido); k['dg'] = np.where(k.ds == '세종', None, m_[1])
# 대응표: 제공 KR 등록 문서(공란 아님) 주소 시군구 ↔ 그 문서 등록원부 출원인 코드 → NPS 연결 → NPS 2022-12 법정동 앞 5자리
con = sqlite3.connect('file:' + ROOT + r"\수집파이프라인\registry_full.sqlite?mode=ro", uri=True)
ra = pd.read_sql("select p.applNo appl_no, a.applicantCd cd from patent p join applicant a on a.rgst_no=p.rgst_no where a.seq=0", con)
raddr = pd.read_sql("select applicantCd cd, applicantAddr addr from applicant where substr(applicantCd,1,1)='1'", con).drop_duplicates('cd'); con.close()
lk = pd.read_parquet(SH + r"\firm_nps_applicant_link.parquet"); lk = lk[lk.nps_hit][['applicantCd', 'biz6', 'nname']].rename(columns={'applicantCd': 'cd'})
w = pd.read_parquet(SH + r"\firm_nps_wkpl_202212.parquet", columns=['name', 'biz6', 'ldong']); w['nn'] = w.name.map(norm2); w['biz6'] = w.biz6.astype(str); w['l5'] = w.ldong.astype(str).str[:5]
lw = lk.merge(w[['biz6', 'nn', 'l5']], left_on=['biz6', 'nname'], right_on=['biz6', 'nn']).drop_duplicates('cd')
x = ra.merge(lw[['cd', 'l5']], on='cd').merge(k[['appl_no', 'dg']], on='appl_no').dropna(subset=['dg'])
l5map = x.groupby('l5').dg.agg(lambda s: s.value_counts().index[0] if len(s) >= 3 else None).dropna().to_dict()
o['l5map_n'] = len(l5map)
SC = {'11': '서울', '26': '부산', '27': '대구', '28': '인천', '29': '광주', '30': '대전', '31': '울산', '36': '세종', '41': '경기', '42': '강원', '51': '강원', '43': '충북', '44': '충남', '45': '전북', '52': '전북', '46': '전남', '47': '경북', '48': '경남', '50': '제주'}
# 기업 주소
fn = fl.merge(ca[['cd', 'k']], on='cd', how='left')
names = {b: set(g.k.dropna()) | {norm2(mv.get(b))} for b, g in fn.groupby('bz')}
addr = {}
for ym in ('202212', '202112', '202312'):
    ww = w if ym == '202212' else pd.read_parquet(SH + rf"\firm_nps_wkpl_{ym}.parquet", columns=['name', 'biz6', 'ldong']).assign(
        nn=lambda d: d.name.map(norm2), biz6=lambda d: d.biz6.astype(str), l5=lambda d: d.ldong.astype(str).str[:5])
    ix = dict(zip(zip(ww.biz6, ww.nn), ww.l5))
    for b in risk:
        if b in addr: continue
        for n_ in names.get(b, {norm2(mv.get(b))}):
            v = ix.get((b[:6], n_))
            if v: addr[b] = v; break
fs = {b: (SC.get(v[:2]), l5map.get(v)) for b, v in addr.items()}
# 후보 쌍
fr_ = fl[fl.cd.isin(set(fl.cd))].merge(ca[['cd', 'k', 'ns']], on='cd'); fr_ = fr_[fr_.bz.isin(set(risk))]
kb = k[k.blank & ((k.ad >= 20220101) & (k.ad <= 20231231) | (k.kd <= 20211231))]
cd_ = fr_.merge(kb[['appl_no', 'ad', 'kd', 'k', 'ds', 'dg']], on='k')
have = set(zip(fa.bz, fa.appl_no)); cd_ = cd_[[(a, b) not in have for a, b in zip(cd_.bz, cd_.appl_no)]]
con = sqlite3.connect('file:' + ROOT + r"\수집파이프라인\business_rights.sqlite" + '?mode=ro', uri=True)
rl = pd.read_sql("select applicant_cd cd, appl_no from right_list where rgst_no like '10%'", con); con.close()
rl['appl_no'] = rl.appl_no.astype(str).str.replace(r'\D', '', regex=True)
rl = rl[rl.appl_no.isin(set(k[k.blank].appl_no))].drop_duplicates()
hold = rl.groupby('appl_no').cd.apply(set).to_dict()
cd_['other_only'] = [bool(hold.get(a)) and c not in hold.get(a, set()) for a, c in zip(cd_.appl_no, cd_.cd)]
# 경쟁 주소(같은 이름 다른 코드의 등록원부 주소)
rx = raddr.addr.fillna('').str.extract(r'^\s*(\S+)\s+(\S+)'); raddr['s'] = rx[0].map(sido); raddr['g'] = rx[1]
ca2 = ca.merge(raddr[['cd', 's', 'g']], on='cd', how='left')
comp = {}
for kk_, g in ca2.dropna(subset=['g']).groupby('k'):
    comp[kk_] = list(zip(g.cd, g.s, g.g))
def comp_same(row):
    return any((c != row.cd) and (s == row.ds) and (g == row.dg) for c, s, g in comp.get(row.k, []))
cd_['comp'] = cd_.apply(comp_same, axis=1)
cd_['fs'] = cd_.bz.map(lambda b: fs.get(b, (None, None))[0]); cd_['fg'] = cd_.bz.map(lambda b: fs.get(b, (None, None))[1])
cd_['sg_eq'] = cd_.ds.notna() & (cd_.ds == cd_.fs) & ((cd_.ds == '세종') | (cd_.dg == cd_.fg))
# P1
rp = fr_[['bz', 'cd']].merge(rl, on='cd').merge(k[['appl_no', 'ad', 'kd']], on='appl_no')
rp = rp[[(a, b) not in have for a, b in zip(rp.bz, rp.appl_no)]]
rp = rp[((rp.ad >= 20220101) & (rp.ad <= 20231231)) | (rp.kd <= 20211231)]
okc = cd_[~cd_.other_only]
SC_ = {'L': rp, 'M': pd.concat([rp, okc[okc.sg_eq & ~okc.comp]]), 'U': pd.concat([rp, okc])}
y0 = pr.y.values; base = (roc_auc_score(y0, pr.p_champ) - roc_auc_score(y0, pr.p_m0)) / (1 - roc_auc_score(y0, pr.p_m0))
for nm_, P in SC_.items():
    win = (P.ad >= 20220101) & (P.ad <= 20231231)
    flip = set(P[win].bz) & set(pr.bz[pr.y == 0]); drp = set(P[P.kd <= 20211231].bz)
    y = np.where(pr.bz.isin(flip), 1, y0); keep = ~pr.bz.isin(drp).values
    a1 = roc_auc_score(y[keep], pr.p_champ[keep]); a0 = roc_auc_score(y[keep], pr.p_m0[keep])
    o[nm_] = dict(pairs=len(P), flips=len(flip), drops=len(drp), skill=(a1 - a0) / (1 - a0), dskill=(a1 - a0) / (1 - a0) - base)
    o[nm_ + '_flipset'] = sorted(flip)
# s4 결과와 대조
s4 = json.load(open(OUT + r"\n6_rescore_blank.json", encoding='utf-8'))['scenarios']
bf = pd.read_csv(OUT + r"\n6_blank_full.csv", dtype={'bz': str})
for nm_ in ('L', 'M', 'U'):
    mine = set(bf[(bf[f'y_{nm_}'] == 1) & (bf.y == 0)].bz)
    o[nm_ + '_cmp'] = dict(s4_flips=s4[f'{nm_}|drop=True']['flips_0to1'], s4_drops=s4[f'{nm_}|drop=True']['drops'], s4_dskill=s4[f'{nm_}|drop=True']['dskill'],
                           flip_jaccard=len(mine & set(o[nm_ + '_flipset'])) / max(1, len(mine | set(o[nm_ + '_flipset']))),
                           only_s4=len(mine - set(o[nm_ + '_flipset'])), only_v0=len(set(o[nm_ + '_flipset']) - mine))
    del o[nm_ + '_flipset']
# M 시나리오 skill CI(부트스트랩 1,000, 기업 단위)
P = SC_['M']; win = (P.ad >= 20220101) & (P.ad <= 20231231)
flip = set(P[win].bz) & set(pr.bz[pr.y == 0]); drp = set(P[P.kd <= 20211231].bz)
y = np.where(pr.bz.isin(flip), 1, y0); keep = ~pr.bz.isin(drp).values
yy = y[keep]; pc = pr.p_champ.values[keep]; pm = pr.p_m0.values[keep]
rng = np.random.default_rng(99); v = []
for _ in range(1000):
    i = rng.integers(0, len(yy), len(yy)); a1 = roc_auc_score(yy[i], pc[i]); a0 = roc_auc_score(yy[i], pm[i]); v.append((a1 - a0) / (1 - a0))
o['M_skill_ci'] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
json.dump(o, open(OUT + r"\verify_0\v_n6_indep_out.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
