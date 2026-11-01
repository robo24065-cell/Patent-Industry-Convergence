# C07-T2 s3: 보류 ERRATA 독립 재검(원 작업자 스크립트 import 없음, 원천에서 다시 계산). sqlite mode=ro. 출력은 C07-T2\cache.
import os, sys, json, re, sqlite3, time
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.stdout.reconfigure(encoding='utf-8')
ROOT = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = ROOT + r"\분석\모델경쟁_20260927"; SH = MC + r"\_shared\ext"
OUT = ROOT + r"\분석\검증루프_20261012\C07\C07-T2"
part = sys.argv[1] if len(sys.argv) > 1 else 'all'
t0 = time.time()
def auc(y, p):
    y = np.asarray(y).astype(int); r = pd.Series(np.asarray(p, float)).rank(method='average').values
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
sk = lambda a, b: (a - b) / (1 - b)
def dump(name, o):
    json.dump(o, open(OUT + rf"\cache\s3_{name}.json", 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
    print(name, json.dumps(o, ensure_ascii=False, default=float)[:3000]); print('t=%.0fs' % (time.time() - t0))

# ---------- A. E-C06T3-03 N7 보안과제 ----------
if part in ('A', 'all'):
    o = {}
    n = pd.read_parquet(SH + r"\ntis_project_raw.parquet", columns=['year', 'secret_flag', 'perform_grp', 'corp_rgst_no'])
    n['digits'] = n.corp_rgst_no.fillna('').str.replace(r'\D', '', regex=True).str.len()
    n['sec'] = n.secret_flag.fillna(False).astype(bool)
    o['secret_flag_vs_grp_agree'] = float((n.sec == (n.perform_grp == '보안과제')).mean())
    o['secret_rows'] = int(n.sec.sum()); o['secret_with_digit_bizno'] = int((n.sec & (n.digits > 0)).sum())
    g = n.groupby('year').agg(rows=('sec', 'size'), sec=('sec', 'sum'))
    g['share'] = g.sec / g.rows
    o['share_by_year'] = {int(k): [int(r.sec), int(r.rows), round(r.share, 5)] for k, r in g.loc[2015:2024].iterrows()}
    def win(a, b):
        s = g.loc[a:b]; return round(s.sec.sum() / s.rows.sum(), 5)
    o['share_2023_24'] = win(2023, 2024); o['share_dev_window_2018_22'] = win(2018, 2022)
    o['share_2015_21'] = win(2015, 2021); o['share_2018_21'] = win(2018, 2021)
    o['dev_years_min_max'] = [round(g.loc[2018:2022].share.min(), 5), round(g.loc[2018:2022].share.max(), 5)]
    # 민감도: 누락 양성 m곳을 현재 음성에서 골라 1로 바꿈(무작위 / 챔피언 점수 비례)
    pr = pd.read_parquet(MC + r"\N7_특허기업_정부RnD첫수행\final_test_pred_r5.parquet")
    y = pr.y.values.astype(int); pc = pr.p_champ.values; pm = pr.p_M0.values.astype(float)
    base = sk(auc(y, pc), auc(y, pm)); o['n'] = len(y); o['pos'] = int(y.sum()); o['base_skill'] = base
    neg = np.where(y == 0)[0]; rng = np.random.default_rng(20261013)
    for miss in (0.049, 0.074):
        m = int(round(miss * y.sum() / (1 - miss)))
        res = {}
        for mode in ('random', 'score_prop'):
            ds = []
            for _ in range(200):
                if mode == 'random': pick = rng.choice(neg, m, replace=False)
                else:
                    w = pc[neg] / pc[neg].sum(); pick = rng.choice(neg, m, replace=False, p=w)
                yy = y.copy(); yy[pick] = 1
                ds.append(sk(auc(yy, pc), auc(yy, pm)) - base)
            res[mode] = [float(np.mean(ds)), float(np.percentile(ds, 5)), float(np.percentile(ds, 95))]
        o[f'miss_{miss}'] = {'added': m, **res}
    dump('A_n7_secret', o)

# ---------- B. E-C05T3-03 N9 보고 지연 신호 ----------
if part in ('B', 'all'):
    o = {}
    lk = pd.read_parquet(SH + r"\ntis_patent_link.parquet", columns=['cc', 'ProjectID', 'appl_year', 'ksic', 'appl_no'])
    o['link_cc'] = lk.cc.value_counts().to_dict(); lk = lk[lk.cc == 'KR']
    fp = pd.read_parquet(MC + r"\N9_정부RnD과제_특허성과예측\final_preds_r6.parquet")
    dv = pd.read_parquet(MC + r"\N9_정부RnD과제_특허성과예측\cache\panel_dev.parquet", columns=['pjt_no', 'year'])
    sets = {int(t): set(g.pjt_no.astype(str)) for t, g in dv.groupby('year')}
    sets[2022] = set(fp.pjt_no.astype(str))
    raw = pd.read_parquet(SH + r"\ntis_project_raw.parquet", columns=['pjt_no', 'year', 'secret_flag'])
    raw = raw[~raw.secret_flag.fillna(False).astype(bool)]
    rs = {int(t): set(g.pjt_no.astype(str)) for t, g in raw.groupby('year')}
    o['panel_vs_raw_nonsecret'] = {t: [len(sets[t]), len(rs.get(t, ())), len(sets[t] & rs.get(t, set()))] for t in sorted(sets)}
    sets[2021] = rs[2021]; o['set_2021_source'] = 'ntis_project_raw 비보안(패널 미보존)'
    lk['ProjectID'] = lk.ProjectID.astype(str)
    rows = {}
    for t, S in sorted(sets.items()):
        l = lk[lk.ProjectID.isin(S)]
        a0 = set(l.loc[l.appl_year == t, 'ProjectID']); a1 = set(l.loc[l.appl_year == t + 1, 'ProjectID'])
        rows[t] = {'n': len(S), 'lag0': len(a0), 'lag1only': len(a1 - a0), 'pos': len(a0 | a1), 'ratio_lag1only_lag0': round(len(a1 - a0) / max(len(a0), 1), 4)}
    o['by_t'] = rows
    # 공식 y 재현
    l = lk[lk.ProjectID.isin(sets[2022])]
    posset = set(l.loc[l.appl_year.isin([2022, 2023]), 'ProjectID'])
    fp['y_re'] = fp.pjt_no.astype(str).isin(posset).astype(int)
    o['y_reproduced_share'] = float((fp.y_re == fp.y).mean()); o['y_pos_official'] = int(fp.y.sum()); o['y_pos_re'] = int(fp.y_re.sum())
    a0 = set(l.loc[l.appl_year == 2022, 'ProjectID']); a1only = set(l.loc[l.appl_year == 2023, 'ProjectID']) - a0
    m2021 = np.mean([rows[2020]['ratio_lag1only_lag0'], rows[2021]['ratio_lag1only_lag0']])
    m1419 = np.mean([rows[t]['ratio_lag1only_lag0'] for t in range(2014, 2020)])
    o['mean_ratio_2020_21'] = m2021; o['mean_ratio_2014_19'] = m1419
    n_add = int(round(m2021 * len(a0))) - len(a1only); o['n_add_upper'] = n_add
    y = fp.y.values.astype(int); pc = fp.p_champ.values.astype(float); pb = fp.p_M0_L2.values.astype(float)
    base = sk(auc(y, pc), auc(y, pb)); o['base_skill'] = base
    neg = np.where(y == 0)[0]; rng = np.random.default_rng(777)
    # 모사: 관측 '이듬해 전용 양성'의 챔피언 점수 분위(20칸) 분포대로 음성에서 뽑음
    q = pd.Series(pc).rank(pct=True).values; bins = np.minimum((q * 20).astype(int), 19)
    is_a1 = fp.pjt_no.astype(str).isin(a1only).values
    target = np.bincount(bins[is_a1], minlength=20) / is_a1.sum()
    negbins = bins[neg]; cnt = np.bincount(negbins, minlength=20)
    w_mimic = target[negbins] / np.maximum(cnt[negbins], 1); w_mimic = w_mimic / w_mimic.sum()
    sims = {}
    for mode in ('random', 'p_M0_L2_prop', 'p_champ_prop', 'mimic_lag1only'):
        ss = []
        for _ in range(200):
            if mode == 'random': pick = rng.choice(neg, n_add, replace=False)
            elif mode == 'p_M0_L2_prop': pick = rng.choice(neg, n_add, replace=False, p=pb[neg] / pb[neg].sum())
            elif mode == 'p_champ_prop': pick = rng.choice(neg, n_add, replace=False, p=pc[neg] / pc[neg].sum())
            else: pick = rng.choice(neg, n_add, replace=False, p=w_mimic)
            yy = y.copy(); yy[pick] = 1; ss.append(sk(auc(yy, pc), auc(yy, pb)))
        sims[mode] = [float(np.percentile(ss, 2.5)), float(np.mean(ss)), float(np.percentile(ss, 97.5))]
    o['sims_skill'] = sims; o['skill_ci_official'] = [0.3623, 0.4195]
    # 출원연도별 연결률(제공 KR 대비 NTIS 연결 출원 비율, 2014–21 평균 대비)
    kp = ROOT + r"\공모전데이터\작업데이터\kr_patents.parquet"; nm = pq.ParquetFile(kp).schema_arrow.names
    k = pq.read_table(kp, columns=[nm[0], nm[1]]).to_pandas(); k.columns = ['appl_no', 'ad']
    k['ay'] = pd.to_numeric(k.ad, errors='coerce') // 10000; k['appl_no'] = k.appl_no.astype(str)
    linked = set(lk.appl_no.astype(str).str.replace(r'\D', '', regex=True))
    k['lk'] = k.appl_no.str.replace(r'\D', '', regex=True).isin(linked)
    r = k[k.ay.between(2014, 2024)].groupby('ay').lk.mean()
    o['linkrate_rel_2014_21'] = {int(a): round(v / r.loc[2014:2021].mean(), 3) for a, v in r.items()}
    dump('B_n9_lag', o)

# ---------- C. E-C05T2-03 국민연금 연결률(출원기간별) ----------
if part in ('C', 'all'):
    o = {}
    con = sqlite3.connect('file:' + ROOT + r"\수집파이프라인\registry_full.sqlite?mode=ro", uri=True)
    p = pd.read_sql("select rgst_no, applDate from patent", con)
    a = pd.read_sql("select rgst_no, seq, applicantCd from applicant", con); con.close()
    p['ay'] = pd.to_numeric(p.applDate.astype(str).str[:4], errors='coerce')
    lk = pd.read_parquet(SH + r"\firm_nps_applicant_link.parquet", columns=['applicantCd', 'nps_hit'])
    nps = set(lk.loc[lk.nps_hit, 'applicantCd'].astype(str)); biz = set(lk.applicantCd.astype(str))
    a['applicantCd'] = a.applicantCd.astype(str)
    a['nps'] = a.applicantCd.isin(nps); a['biz'] = a.applicantCd.isin(biz); a['dom'] = a.applicantCd.str[0].isin(['1', '2'])
    a = a.sort_values(['rgst_no', 'seq'])
    first = a.groupby('rgst_no').first()
    anyf = a.groupby('rgst_no')[['nps', 'biz', 'dom']].any()
    p = p.set_index('rgst_no')
    p = p.join(first[['nps', 'biz', 'dom']].add_prefix('f_')).join(anyf.add_prefix('a_'))
    p = p.fillna(False)
    o['n_patents'] = len(p)
    o['overall'] = {k: round(float(p[k].mean()), 4) for k in ['f_nps', 'a_nps', 'f_biz', 'a_biz']}
    per = {}
    for lo, hi in ((2005, 2012), (2013, 2016), (2017, 2022)):
        s = p[p.ay.between(lo, hi)]
        dom = s[s.a_dom]
        per[f'{lo}-{hi}'] = {'n': len(s), 'first_applicant_nps': round(float(s.f_nps.mean()), 4), 'any_applicant_nps': round(float(s.a_nps.mean()), 4),
                             'dom_corp_share': round(float(s.a_dom.mean()), 4), 'nps_among_domcorp_patents': round(float(dom.a_nps.mean()), 4),
                             'biz_among_domcorp_patents': round(float(dom.a_biz.mean()), 4), 'nps_among_biz_patents': round(float(s[s.a_biz].a_nps.mean()), 4)}
    o['by_period'] = per
    fr = pd.read_csv(SH + r"\firm_nps_link_rates.csv", encoding='utf-8-sig'); o['file_rows'] = fr.values.tolist()
    dump('C_nps_rates', o)

# ---------- D. E-C06T1-06 DART 12,322사 상장 비율 ----------
if part in ('D', 'all'):
    o = {}
    D = ROOT + r"\공모전데이터\외부데이터\DART"
    con = sqlite3.connect('file:' + D + r"\dart_company.sqlite?mode=ro", uri=True)
    c = pd.read_sql("select corp_code, corp_cls, status from company", con); con.close()
    o['n'] = len(c); o['cls'] = c.corp_cls.value_counts().to_dict()
    o['listed_YKN_share'] = round(float(c.corp_cls.isin(['Y', 'K', 'N']).mean()), 4)
    o['listed_YK_share'] = round(float(c.corp_cls.isin(['Y', 'K']).mean()), 4)
    t = pd.read_csv(D + r"\dart_targets.csv", dtype=str, encoding='utf-8-sig')
    o['targets_n'] = len(t); o['targets_cols'] = list(t.columns)
    o['targets_eq_company'] = bool(set(t.corp_code) == set(c.corp_code))
    o['targets_stockcode_share'] = round(float(t.stock_code.fillna('').str.strip().ne('').mean()), 4)
    o['targets_n_pat_min'] = int(pd.to_numeric(t.n_pat).min())
    cc = pd.read_csv(D + r"\dart_corpcode.csv", dtype=str, encoding='utf-8-sig')
    o['corpcode_all_n'] = len(cc); o['corpcode_all_stock_share'] = round(float(cc.stock_code.fillna('').str.strip().ne('').mean()), 4)
    dump('D_dart', o)

# ---------- E. E-C06T2-02 그림 1 '공개 최강 최대 58.8%' ----------
if part in ('E', 'all'):
    o = {}
    L = pd.read_parquet(MC + r"\_shared_report\label_origin_R10\confirm_R12\cache\labels.parquet")
    o['split_counts'] = L.split.value_counts().to_dict()
    cands = [c for c in L.columns if c not in ('appl_no', 'ay', 'split', 'PROV', 'code', 'PKR')]
    agree = {}
    for sp in L.split.unique():
        s = L[L.split == sp]
        agree[sp] = {c: round(float((s[c].astype(str) == s.PROV.astype(str)).mean()), 4) for c in cands}
    o['agree'] = agree
    for sp, d in agree.items():
        best = max(d, key=d.get); o[f'max_{sp}'] = [best, d[best]]
    o['ay_by_split'] = {sp: [int(L[L.split == sp].ay.min()), int(L[L.split == sp].ay.max()), int((L.split == sp).sum())] for sp in L.split.unique()}
    dev = pd.read_csv(MC + r"\_shared_report\label_origin_R10\confirm_R12\res_dev.csv", encoding='utf-8-sig')
    o['best_by_criterion'] = dict(zip(dev['기준'], dev['BEST']))
    dump('E_fig1', o)

# ---------- F. E-C05T2-04 N4 폐업 양성 업태 ----------
if part in ('F', 'all'):
    o = {}
    pr = pd.read_parquet(MC + r"\N4_특허행태_기업폐업조기경보\final_test_pred_r4.parquet", columns=['applicantCd', 'bizno', 'y'])
    o['n'] = len(pr); o['pos'] = int(pr.y.sum())
    lk = pd.read_parquet(SH + r"\firm_nps_applicant_link.parquet")
    lk = lk[lk.nps_hit][['applicantCd', 'biz6', 'nname']]
    SUF = re.compile(r"주식회사|유한책임회사|유한회사|합자회사|합명회사|농업회사법인|영농조합법인|재단법인|사단법인|\(주\)|\(유\)|\(합\)|㈜|（주）|\(株\)|株式会社|CO\.,?LTD\.?|CORPORATION|INC\.?", re.I)
    nrm = lambda s: re.sub(r"[\s\(\)\[\]·\.,\-&'\"/]", "", SUF.sub("", str(s or "")).upper())
    def grp(code):
        c = str(code or '')[:2]
        if not c.isdigit(): return '미상'
        v = int(c)
        if v == 99: return '미상'
        if 1 <= v <= 5: return '농림어'
        if 10 <= v <= 14: return '광업'
        if 15 <= v <= 37: return '제조'
        if 40 <= v <= 41: return '전기가스수도'
        if 45 <= v <= 46: return '건설'
        if 50 <= v <= 51: return '도매'
        if v == 52: return '소매'
        if v == 55: return '음식숙박'
        if 60 <= v <= 64: return '운수통신'
        if v == 70: return '부동산'
        return '서비스기타'
    ind = {}
    for ym in ['202312', '202212', '202112', '202412', '202012', '201912', '201812', '201712', '201612', '201512', '202512', '202608']:  # 기준일(2023말) 우선, 없으면 이전·이후 보충
        w = pd.read_parquet(SH + rf"\firm_nps_wkpl_{ym}.parquet", columns=['name', 'biz6', 'indcd', 'indnm', 'n_emp'])
        w['nname'] = w.name.map(nrm)
        m = lk.merge(w, on=['biz6', 'nname'])
        m = m.sort_values('n_emp', ascending=False).drop_duplicates('applicantCd')
        for cd, ic in zip(m.applicantCd, m.indcd):
            ind.setdefault(cd, ic)
        if ym == '202312':
            sample = w[['indcd', 'indnm']].drop_duplicates().assign(p2=lambda d: d.indcd.astype(str).str[:2]).groupby('p2').indnm.first().to_dict()
            o['indcd_prefix_examples'] = {k: sample[k] for k in sorted(sample)[:60]}
    pr['g'] = pr.applicantCd.map(lambda c: grp(ind.get(c)))
    pos = pr[pr.y == 1]
    o['pos_unknown_share'] = round(float((pos.g == '미상').mean()), 4)
    kn = pos[pos.g != '미상']
    o['pos_dist_known'] = (kn.g.value_counts(normalize=True).round(4)).to_dict()
    o['pop_dist_known'] = (pr[pr.g != '미상'].g.value_counts(normalize=True).round(4)).to_dict()
    o['pop_unknown_share'] = round(float((pr.g == '미상').mean()), 4)
    # 공식 KOSIS 9.8.15 (C05-T2가 화면 표에서 옮긴 CSV; 내용 점검 후 사용)
    kf = ROOT + r"\분석\검증루프_20261012\C05\C05-T2\download\kosis_DT_133N_9815A_closures_by_type_industry.csv"
    lines = [l for l in open(kf, encoding='utf-8-sig') if not l.startswith('#')]
    from io import StringIO
    K = pd.read_csv(StringIO(''.join(lines)), sep=';')
    o['kosis_biz_values'] = K.biz.unique().tolist(); o['kosis_ind_values'] = K.ind.unique().tolist()
    corp = K[K.biz.astype(str).str.contains('법인')]
    for c in ['2024', '2025']: corp[c] = pd.to_numeric(corp[c].astype(str).str.replace(',', ''), errors='coerce')
    tot = corp[corp.ind.astype(str).str.contains('합계')][['2024', '2025']].sum().sum()
    man = corp[corp.ind.astype(str).str.contains('제조')][['2024', '2025']].sum().sum()
    o['kosis_corp_2024_25_total'] = float(tot); o['kosis_corp_2024_25_manuf'] = float(man); o['kosis_manuf_share'] = round(float(man / tot), 4)
    dump('F_n4_industry', o)
