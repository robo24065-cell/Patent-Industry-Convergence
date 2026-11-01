# -*- coding: utf-8 -*-
"""C05-T3 s5: N6 재구성 불일치 원인 분해 + 재채점 + 공란 등록·2024 결손 영향.
트랙 코드는 import하지 않고, 트랙 규칙(n6_lib 확인 후)을 자기 코드로 '공식 규칙 변형'으로 다시 쓴다.
구성요소: [F] 기업->출원인코드 (자기: CORP_APPLICANT 전체·상호 2019–21 / 공식: r3nts_applicant_status·상호 2010–21)
          [A] 출원인코드->출원 (자기: registry_full+유일명 / 공식: 공용 app_firm.parquet)
          [K] known_date (자기: meta min(pub,rgst) / 공식: app_firm.known_date)
"""
import duckdb, pandas as pd, numpy as np, json, os, sys, io, re, math, sqlite3
from sklearn.metrics import roc_auc_score
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
B = os.path.join(ROOT, "분석"); SH = os.path.join(B, r"모델경쟁_20260927\_shared")
N6 = os.path.join(B, r"모델경쟁_20260927\N6_정부RnD기업_첫특허전환")
R3I = os.path.join(B, r"모델경쟁_20260927\_ideas_R3")
OUT = os.path.join(B, r"검증루프_20261012\C05\C05-T3"); CA = os.path.join(OUT, "cache")
c = duckdb.connect(); c.execute("SET memory_limit='3GB'; SET threads=2")
P = os.path.join(SH, "ext", "ntis_project_raw.parquet").replace("\\", "/")
M = os.path.join(SH, "meta", "*.parquet").replace("\\", "/")
o = {}
SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)


def norm_off(s):  # 공식 규칙 문장(정규화 상호)을 자기 코드로 — 트랙 정규식과 같은 접미어 목록
    s = SUF.sub("", str(s or "")).upper()
    return re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", s)


nt = c.sql(f"""select pjt_no, year::int y, regexp_replace(corp_rgst_no,'[^0-9]','','g') bz, research_agency nm
   from read_parquet('{P}') where year::int between 2008 and 2021 and perform_grp in ('중소기업','중견기업','대기업')""").df()
nt = nt[nt.bz.str.len() == 10]
pop = nt[nt.y.between(2019, 2021)].bz.unique()
o['pop'] = int(len(pop))


def mode_name(df):
    return df.groupby(['bz', 'nm']).size().reset_index(name='k').sort_values(['bz', 'k'], ascending=[True, False]).drop_duplicates('bz')


# 공식 규칙 기업 연결 (r3nts_applicant_status, 상호 2010–2021 최빈)
ns = pd.read_parquet(os.path.join(SH, "ext", "r3nts_applicant_status.parquet"), columns=["applicantCd", "bizno", "aname"])
b2a = ns.groupby("bizno").applicantCd.apply(list).to_dict()
ns["nn"] = ns.aname.map(norm_off); uq = ns.groupby("nn").applicantCd.agg(["first", "size"]); uq = uq[uq["size"] == 1]["first"].to_dict()
mn = mode_name(nt[nt.y >= 2010])
mn = mn[mn.bz.isin(pop)]
rows = []
for bz, nm in zip(mn.bz, mn.nm):
    cds = b2a.get(bz)
    if cds:
        rows += [(bz, x, 'bizno') for x in cds]
    else:
        x = uq.get(norm_off(nm))
        if x:
            rows.append((bz, x, 'name'))
fl_off = pd.DataFrame(rows, columns=['bz', 'cd', 'route'])
fl_me = pd.read_parquet(os.path.join(CA, 'n6_firm_link.parquet'))
o['firm_link_off'] = fl_off.groupby('route').bz.nunique().to_dict()
o['firm_link_me'] = fl_me.groupby('route').bz.nunique().to_dict()
o['firm_linked_off'] = int(fl_off.bz.nunique()); o['firm_linked_me'] = int(fl_me.bz.nunique())
# 공식 출원 연결 app_firm
af = pd.read_parquet(os.path.join(R3I, "cache", "app_firm.parquet"), columns=["applicantCd", "appl_no", "appl_date", "known_date", "src"])
af = af.rename(columns={'applicantCd': 'cd'})
af['appl_date'] = pd.to_numeric(af.appl_date, errors='coerce'); af['known_date'] = pd.to_numeric(af.known_date, errors='coerce')
# 자기 출원 연결
fa_me = pd.read_parquet(os.path.join(CA, 'n6_firm_apps.parquet'))
kr = c.sql(f"""select appl_no, any_value(appl_date) ad, any_value(pub_date) pd_, any_value(rgst_date) rd, any_value(rgst_no) rg
   from read_parquet('{M}') where cc='KR' group by 1""").df()
for col in ('ad', 'pd_', 'rd'):
    kr[col] = pd.to_numeric(kr[col], errors='coerce')
kr['known_me'] = kr[['pd_', 'rd']].min(axis=1)
apps_me = fa_me[['cd', 'appl_no']].drop_duplicates()
# 전체 코드->출원 (자기) 다시: firm_apps는 firm 연결 코드로 이미 제한 -> 공식 코드집합과 비교 위해 전체 필요
#  -> s4의 코드-출원 표를 재생성하지 않고, 두 기업연결 모두에 대해 쓰이는 코드만 필요하므로 firm_apps(자기 기업연결) + 공식 기업연결 코드의 자기 출원 재구성
con = sqlite3.connect('file:' + os.path.join(ROOT, r"수집파이프라인\registry_full.sqlite") + '?mode=ro', uri=True)
ra = pd.read_sql("select rgst_no, applicantCd cd from applicant", con).drop_duplicates(); con.close()
reg_set = set(ra.rgst_no)
a1 = kr[kr.rg.isin(reg_set)][['appl_no', 'rg']].merge(ra.rename(columns={'rgst_no': 'rg'}), on='rg')[['appl_no', 'cd']]
caf = []
with open(os.path.join(ROOT, r"공모전데이터\외부데이터\Corporate_20260720\CORP_APPLICANT.txt"), encoding="utf-8", errors="replace") as fh:
    next(fh)
    for line in fh:
        f = line.rstrip(chr(13) + chr(10)).split("¶")
        if len(f) >= 5:
            caf.append((f[0].strip(), f[1].strip()))
caf = pd.DataFrame(caf, columns=['cd', 'name'])


def norm_me(s):
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s)
    s = re.sub(r"\(주\)|㈜|주식회사|\(유\)|유한회사|\(사\)|사단법인|\(재\)|재단법인|\(합\)|합자회사|유한책임회사|농업회사법인|영농조합법인", "", s)
    s = re.sub(r"[\s\.\,\-\(\)\[\]&·'\"]", "", s).lower()
    return s or None


caf['nn'] = caf.name.map(norm_me)
cnt = caf.groupby('nn').cd.nunique(); nn2cd = caf[caf.nn.isin(cnt[cnt == 1].index)].drop_duplicates('nn')[['nn', 'cd']]
meta_app = c.sql(f"select appl_no, any_value(applicant) applicant from read_parquet('{M}') where cc='KR' group by 1").df()
rest = kr[~kr.rg.isin(reg_set)][['appl_no']].merge(meta_app, on='appl_no')
rest['nn'] = rest.applicant.map(norm_me)
a2 = rest.merge(nn2cd, on='nn')[['appl_no', 'cd']]
A_me = pd.concat([a1, a2]).drop_duplicates()
A_me = A_me.merge(kr[['appl_no', 'ad', 'known_me']], on='appl_no')
A_off = af.rename(columns={'appl_date': 'ad', 'known_date': 'known_off'})[['cd', 'appl_no', 'ad', 'known_off', 'src']]
A_off = A_off.merge(kr[['appl_no', 'known_me']], on='appl_no', how='left')
o['A_off_rows'] = int(len(A_off)); o['A_me_rows'] = int(len(A_me))
pa = A_off[['appl_no', 'cd']].drop_duplicates().merge(A_me[['appl_no', 'cd']].drop_duplicates(), how='outer', indicator=True)
o['A_pairs_cmp'] = pa._merge.value_counts().to_dict()
o['known_off_vs_me_diff_rows'] = int((A_off.known_off.round() != A_off.known_me.round()).sum())


def build(fl, A, kcol, tag):
    fa = fl[['bz', 'cd']].drop_duplicates().merge(A, on='cd')
    fa = fa[fa.ad <= 20231231]
    first = fa.groupby('bz')[kcol].min()
    risk = pd.Index(pop).difference(first[first <= 20211231].index)
    pos = set(fa[(fa.ad >= 20220101) & (fa.ad <= 20231231)].bz)
    d = pd.DataFrame({'bz': risk}); d['y'] = d.bz.isin(pos).astype(int)
    return d, fa


pr = pd.read_parquet(os.path.join(N6, "final_preds_r5.parquet")); pr['bz'] = pr.bz.astype(str).str.zfill(10)
pr = pr.rename(columns={'y': 'y_off'})
combos = {'OFF(F_off,A_off,K_off)': (fl_off, A_off, 'known_off'),
          'F_me,A_off,K_off': (fl_me, A_off, 'known_off'),
          'F_off,A_me,K_me': (fl_off, A_me, 'known_me'),
          'F_off,A_off,K_me': (fl_off, A_off, 'known_me'),
          'ME(F_me,A_me,K_me)': (fl_me, A_me, 'known_me')}
res = {}
store = {}


def score(y, p):
    return float(roc_auc_score(y, p))


def topk(y, p, frac):
    k = max(1, int(round(frac * len(y))))
    return float(np.asarray(y)[np.argsort(-np.asarray(p), kind='stable')[:k]].mean())


for k, (fl, A, kc) in combos.items():
    d, fa = build(fl, A, kc, k)
    store[k] = (d, fa)
    m = pr.merge(d, on='bz', how='outer', indicator=True)
    both = m[m._merge == 'both']
    r = {'n': int(len(d)), 'pos': int(d.y.sum()), 'left_only(official only)': int((m._merge == 'left_only').sum()),
         'right_only(rebuild only)': int((m._merge == 'right_only').sum()), 'jaccard': float(len(both) / len(m)),
         'y_agree_common': float((both.y_off == both.y).mean()), 'pos_diff_pct': float((d.y.sum() - pr.y_off.sum()) / pr.y_off.sum())}
    # 재채점 1: 공통 단위, 재구성 y
    a = score(both.y, both.p_champ); b = score(both.y, both.p_m0)
    r['common_auc'] = a; r['common_auc_M0'] = b; r['common_skill'] = (a - b) / (1 - b); r['common_top1'] = topk(both.y.values, both.p_champ.values, 0.01)
    # 재채점 2: 공식 단위 전체에 재구성 y(위험집합 밖으로 빠진 단위도 y_any 계산)
    pos_all = set(fa[(fa.ad >= 20220101) & (fa.ad <= 20231231)].bz)
    yy = pr.bz.isin(pos_all).astype(int)
    a = score(yy, pr.p_champ); b = score(yy, pr.p_m0)
    r['offunits_pos'] = int(yy.sum()); r['offunits_auc'] = a; r['offunits_auc_M0'] = b; r['offunits_skill'] = (a - b) / (1 - b)
    r['offunits_top1'] = topk(yy.values, pr.p_champ.values, 0.01); r['offunits_y_agree'] = float((yy == pr.y_off).mean())
    res[k] = r
o['combos'] = res
# 공식 원본 값
a = score(pr.y_off, pr.p_champ); b = score(pr.y_off, pr.p_m0)
o['official_from_preds'] = {'auc': a, 'auc_M0': b, 'skill': (a - b) / (1 - b), 'top1': topk(pr.y_off.values, pr.p_champ.values, 0.01),
                            'top5': topk(pr.y_off.values, pr.p_champ.values, 0.05), 'top10': topk(pr.y_off.values, pr.p_champ.values, 0.10),
                            'top1_M0': topk(pr.y_off.values, pr.p_m0.values, 0.01), 'base': float(pr.y_off.mean()), 'n': int(len(pr)), 'pos': int(pr.y_off.sum())}
# 원인 사례: ME vs OFF 불일치 단위
dO, faO = store['OFF(F_off,A_off,K_off)']; dM, faM = store['ME(F_me,A_me,K_me)']
mm = dO.merge(dM, on='bz', how='outer', suffixes=('_off', '_me'), indicator=True)
lo = mm[mm._merge == 'left_only'].bz; ro = mm[mm._merge == 'right_only'].bz
flo = set(fl_off.bz); flm = set(fl_me.bz)
o['diag_left_only'] = {'n': int(len(lo)), 'linked_me': int(lo.isin(flm).sum()), 'linked_off': int(lo.isin(flo).sum()),
                       'me_link_route_name': int(lo.isin(set(fl_me[fl_me.route == 'name'].bz)).sum())}
o['diag_right_only'] = {'n': int(len(ro)), 'linked_me': int(ro.isin(flm).sum()), 'linked_off': int(ro.isin(flo).sum())}
# 공식 재현 여부 (OFF 변형이 공식 preds와 같은가)
o['off_variant_reproduces'] = bool(res['OFF(F_off,A_off,K_off)']['jaccard'] == 1.0 and res['OFF(F_off,A_off,K_off)']['y_agree_common'] == 1.0)
# ---- 공란 등록 영향: 제공 KR에서 rgst_no 공란이나 실제 등록된 출원 — 여기서는 '등록원부 경로가 없어 이름 경로로만 잡힌 결과창 출원' 비중
fa_off = store['OFF(F_off,A_off,K_off)'][1]
win = fa_off[(fa_off.ad >= 20220101) & (fa_off.ad <= 20231231)]
o['window_apps_off_src'] = win.groupby('src').appl_no.nunique().to_dict()
o['window_apps_off_rgst_present'] = int(win.merge(kr[['appl_no', 'rg', 'rd']], on='appl_no').rg.notna().sum())
# 공란 등록 후보: 결과창 출원 중 meta rgst_no 공란 & 2023-11 이후 등록일 가능 -> 공식 app_firm에서 'name' 경로. 기업 단위로, 양성이 name 경로에만 의존하는 수
posf = win.groupby('bz').src.apply(lambda s: set(s))
o['pos_firms_off'] = int(len(posf)); o['pos_firms_name_only'] = int(sum(1 for s in posf if s == {'name'}))
# 2024+ 레코드가 y/X에 쓰이는지
o['apps_used_after_2023'] = int((fa_off.ad > 20231231).sum())
o['risk_known_after_2021_but_appl_le_2021'] = int(((fa_off.ad <= 20211231) & (fa_off.known_off > 20211231)).sum())
# 2023-12 출원 결과창 비중 vs 2021-12 (개발 비교)
o['window_dec2023_share'] = float(((win.ad >= 20231201) & (win.ad <= 20231231)).mean())
json.dump(o, open(os.path.join(CA, 's5_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps(o, ensure_ascii=False, indent=1, default=str))
