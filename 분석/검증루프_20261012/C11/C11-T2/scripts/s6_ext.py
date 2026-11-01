# -*- coding: utf-8 -*-
"""R9 s06: (2) 외부 결과 정합에 TXT 셀 정의 추가 + (3) 산업 통계 연결 수. R8 NONCIRC_R8/s05_external.py 설계 A(50개 5자리) 그대로.
 신호(산업 × t0): cell_real / cell_pred / bridge_real / bridge_pred / pat_mom
   제공 = R7 pub_full(P50), 연계표 = R7 conc_task 값을 같은 코드 KSIC에 공유, TXT = T50(txt50_task, TXT_R9)
   bridge = 라벨 무관 M_none(NONCIRC_R8) mg 부상을 각 체계 패널 가중치로 산업에 옮김(제공 = R3 cache_pub, 연계표 = R7 cache_conc_pub, TXT = cache_txt_pub)
 외부: KOSIS 광업제조업조사 5자리(10차) 출하액·종사자수·사업체수, 수출·수입(HS6). 성장·Spearman·산업 부트스트랩 1,000회 = R8 s05.
 설계 B(27코드)는 TXT = 연계표(정의상 같은 코드 배정)라 계산하지 않는다.
 보조: (가) BH q(제공−TXT 25개), (나) 연계표 그룹 안(34 KSIC, 원점마다 코드 평균을 뺀 값) Pearson — 제공 대 TXT, 그룹(11) 군집 부트스트랩."""
import os, json, sys, time
# C11-T2 사본: 계산 코드·seed·B는 R9 s06 그대로. 바꾼 것: TXT 예측·TXT 패널·txt_assign 경로(팔별), 출력 경로, 단계 분할(9분 제한).
# 사용: python s6_ext.py <arm> main <pair_idx,...> | within | link   (pair_idx: 0=prov-txt, 1=txt-conc, 2=prov-conc)
sys.dont_write_bytecode = True
import numpy as np, pandas as pd
from scipy.stats import rankdata
BASE0 = "C:/Users/PC/Downloads/특허-산업 융합데이터 활용 아이디어 공모전"
T3 = BASE0 + "/분석/모델경쟁_20260927/T3_기술셀부상예측"; WD = BASE0 + "/분석/검증루프_20261012/C11/C11-T2"
E7 = os.path.join(T3, 'ESSENTIAL_R7'); R8D = os.path.join(T3, 'NONCIRC_R8')
ARM = sys.argv[1]; PART = sys.argv[2]; T_START = time.time()
TXT_PREDS = (T3 + '/TXT_R9/results/txt50_task_preds.parquet') if ARM == 'stored' else f'{WD}/res/runs/txt50_task__{ARM}_preds.parquet'
TXT_PANEL = (T3 + '/TXT_R9/cache_txt_pub/panel_ccy.parquet') if ARM == 'stored' else f'{WD}/cache/txt_{ARM}/panel_ccy.parquet'
TXT_ASSIGN = (T3 + '/TXT_R9/txt_assign.parquet') if ARM == 'stored' else f'{WD}/cache/txt_assign_{ARM}.parquet'
OUTP = f'{WD}/res/s6'; os.makedirs(OUTP, exist_ok=True)
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"; SHX = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
ORIG = list(range(2010, 2019)); NB = 1000
km = pd.read_csv(os.path.join(E7, 'conc_ksic_map.csv'), dtype=str).set_index('ksic').code
K50 = sorted(km.index); CODES = sorted(km.unique())
MULTI = sorted([k for k in K50 if (km == km[k]).sum() > 1])

# ---------- 외부 지표 (R8 s05 그대로) ----------
ko = pd.read_csv(os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
ko = ko[(ko.기준 == "10차") & ko.항목코드.isin(["T01", "T02", "T04"])]
ext = {}
for code, nm in (("T04", "출하액"), ("T02", "종사자수"), ("T01", "사업체수")):
    z = ko[ko.항목코드 == code]
    ext[nm] = z[z.마스킹 == 0].pivot_table(index="세세분류코드", columns="연도", values="값", aggfunc="sum").reindex(index=K50)
mp = pd.read_csv(os.path.join(BASE, r"분석\fusion\ksic_hs6_map.csv"), dtype=str).drop_duplicates(["k5", "hs6"])
mp["w"] = 1.0 / mp.groupby("hs6").k5.transform("nunique")
ex = pd.read_csv(os.path.join(SHX, "trade_kr_export_hs6_uv_2000_2021.csv"), dtype={"hs6": str}, usecols=["year", "hs6", "export_usd"])
im = pd.read_csv(os.path.join(BASE, r"분석\fusion\hs6_import_yearly.csv"), dtype={"hs6": str}); im = im[im.partner == "WLD"].rename(columns={"value": "import_usd"})
for nm, d, col in (("수출액", ex, "export_usd"), ("수입액", im, "import_usd")):
    z = d[d.year <= 2021].merge(mp, on="hs6"); z["v"] = z[col] * z.w
    ext[nm] = z.pivot_table(index="k5", columns="year", values="v", aggfunc="sum").reindex(index=K50)
for nm in ext: assert max(ext[nm].columns) <= 2021


def growth(Yd, t0):
    f = [y for y in range(t0 + 1, t0 + 4) if y in Yd.columns]; p = [y for y in range(t0 - 2, t0 + 1) if y in Yd.columns]
    return np.log(Yd[f].mean(axis=1, skipna=True)) - np.log(Yd[p].mean(axis=1, skipna=True))


G = {nm: pd.DataFrame({t0: growth(Yd, t0) for t0 in ORIG}) for nm, Yd in ext.items()}


# ---------- 신호 ----------
def cell_sig(preds):
    d = preds[preds.t0.isin(ORIG)].copy(); d['r'] = d.groupby('t0').p.rank(pct=True)
    return d.groupby(['ksic', 't0']).yb.mean().unstack(), d.groupby(['ksic', 't0']).r.mean().unstack()


def panel_tot(fn):
    df = pd.read_parquet(fn, columns=['ksic', 'mg', 'cc', 'year', 'n']); df['kr'] = np.where(df.cc == 'KR', df.n, 0.0)
    return df.groupby(['ksic', 'mg', 'year'])[['n', 'kr']].sum().reset_index()


mgp = pd.read_parquet(os.path.join(R8D, 'results', 'mg_ctx-none_preds.parquet')); mgp = mgp[mgp.t0.isin(ORIG)].copy()
mgp['r'] = mgp.groupby('t0').p.rank(pct=True)


def bridge_sig(pan):
    real, pred, mom = {}, {}, {}
    for t0 in ORIG:
        w = pan[(pan.year >= t0 - 2) & (pan.year <= t0)].groupby(['ksic', 'mg']).n.sum().reset_index()
        w['s'] = w.n / w.groupby('ksic').n.transform('sum')
        m = w.merge(mgp[mgp.t0 == t0][['mg', 'yb', 'r']], on='mg'); den = m.groupby('ksic').s.sum()
        real[t0] = (m.s * m.yb).groupby(m.ksic).sum() / den; pred[t0] = (m.s * m.r).groupby(m.ksic).sum() / den
        k0 = pan[(pan.year >= t0 - 2) & (pan.year <= t0)].groupby('ksic').kr.sum(); k1 = pan[(pan.year >= t0 - 5) & (pan.year <= t0 - 3)].groupby('ksic').kr.sum()
        mom[t0] = np.log((k0 + 1) / (k1.reindex(k0.index).fillna(0) + 1))
    return pd.DataFrame(real), pd.DataFrame(pred), pd.DataFrame(mom)


expand = lambda X: X.reindex(km.reindex(K50).values).set_axis(K50)
P50 = pd.read_parquet(os.path.join(E7, 'results', 'pub_full_preds.parquet')); C27r7 = pd.read_parquet(os.path.join(E7, 'results', 'pub_conc_task_preds.parquet'))
T50 = pd.read_parquet(TXT_PREDS)
S = {}
S[('prov', 'cell_real')], S[('prov', 'cell_pred')] = cell_sig(P50)
cr, cp = cell_sig(C27r7); S[('conc', 'cell_real')], S[('conc', 'cell_pred')] = expand(cr), expand(cp)
S[('txt', 'cell_real')], S[('txt', 'cell_pred')] = cell_sig(T50)
for sysn, fn in (('prov', os.path.join(T3, 'R3', 'cache_pub', 'panel_ccy.parquet')), ('conc', os.path.join(E7, 'cache_conc_pub', 'panel_ccy.parquet')),
                 ('txt', TXT_PANEL)):
    br = bridge_sig(panel_tot(fn))
    if sysn == 'conc': br = [expand(x) for x in br]
    S[(sysn, 'bridge_real')], S[(sysn, 'bridge_pred')], S[(sysn, 'pat_mom')] = br
for k in S: S[k] = S[k].reindex(index=K50, columns=ORIG)


def rho_cols(A, Bm):
    out = np.full(A.shape[1], np.nan)
    for j in range(A.shape[1]):
        a = A[:, j]; b = Bm[:, j]; ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 8: continue
        ra = rankdata(a[ok]); rb = rankdata(b[ok])
        if ra.std() == 0 or rb.std() == 0: continue
        out[j] = np.corrcoef(ra, rb)[0, 1]
    return out


SIGS = ('cell_real', 'cell_pred', 'bridge_real', 'bridge_pred', 'pat_mom')
PAIRS = (('prov', 'txt'), ('txt', 'conc'), ('prov', 'conc'))
if PART == 'main':
    units = np.array(K50); rows = []; SEL = [int(x) for x in sys.argv[3].split(',')]
    for pi_, (A, Bn) in enumerate(PAIRS):
        if pi_ not in SEL: continue
        rng = np.random.default_rng(7 + pi_)
        for nm in ext:
            g = G[nm]; valid = units[g.reindex(units).notna().any(axis=1).values]; gv = g.loc[valid, ORIG].values.astype(float)
            for sg in SIGS:
                sa = S[(A, sg)].loc[valid, ORIG].values.astype(float); sb = S[(Bn, sg)].loc[valid, ORIG].values.astype(float)
                ra, rb = rho_cols(sa, gv), rho_cols(sb, gv); bs = np.empty(NB)
                for b in range(NB):
                    ii = rng.integers(0, len(valid), len(valid)); bs[b] = np.nanmean(rho_cols(sa[ii], gv[ii]) - rho_cols(sb[ii], gv[ii]))
                d = ra - rb; lo, hi = np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5); w = int(np.nansum(d > 0)); n = int(np.sum(~np.isnan(d)))
                pboot = float(min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean())))
                verdict = 'A 우위' if (np.nanmean(d) > 0 and lo > 0 and w >= 6) else ('B 우위' if (np.nanmean(d) < 0 and hi < 0 and (n - w) >= 6) else '같다')
                rows.append(dict(비교=f'{A}-{Bn}', 외부지표=nm, 신호=sg, n산업=len(valid), A_rho=round(np.nanmean(ra), 4), B_rho=round(np.nanmean(rb), 4),
                                 차이=round(np.nanmean(d), 4), CI_lo=round(lo, 4), CI_hi=round(hi, 4), A우세_원점=w, 원점=n, p_boot=round(pboot, 4), 판정=verdict))
            print(A, Bn, nm, 'done', round(time.time() - T_START), flush=True)
    O = pd.DataFrame(rows)
    # BH (비교쌍별 25개)
    O['q_BH'] = np.nan
    for cmp_, g in O.groupby('비교'):
        p = g.p_boot.values; o = np.argsort(p); m = len(p); q = np.empty(m); q[o] = np.minimum.accumulate((p[o] * m / np.arange(1, m + 1))[::-1])[::-1]
        O.loc[g.index, 'q_BH'] = np.minimum(q, 1).round(4)
    O.to_csv(f"{OUTP}/s6_external__{ARM}__p{'_'.join(map(str, SEL))}.csv", index=False, encoding='utf-8-sig')
    print(O.to_string()); sys.exit(0)

summ = pd.DataFrame()

# ---------- (나) 연계표 그룹 안 상관(34 KSIC) ----------
def within(sig, nm):
    g = G[nm].loc[MULTI, ORIG]; s = sig.loc[MULTI, ORIG]; code = km.reindex(MULTI).values
    ok = g.notna() & s.notna()
    gd = g.where(ok); sd = s.where(ok)
    gd = gd - gd.groupby(code).transform('mean'); sd = sd - sd.groupby(code).transform('mean')
    return gd, sd, code


def wcorr(G_, S_):
    """G_, S_: (n, 9) 그룹 평균을 뺀 배열. 열마다 Pearson."""
    r = np.full(G_.shape[1], np.nan)
    for j in range(G_.shape[1]):
        a = S_[:, j]; b = G_[:, j]; ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() >= 8 and a[ok].std() > 0 and b[ok].std() > 0: r[j] = np.corrcoef(a[ok], b[ok])[0, 1]
    return r


if PART == 'within':
    wrows = []; rng = np.random.default_rng(11); codes_m = sorted(set(km.reindex(MULTI)))
    for nm in ext:
        for sg in ('cell_real', 'cell_pred', 'bridge_real'):
            gp, sp, code = within(S[('prov', sg)], nm); gt, st, _ = within(S[('txt', sg)], nm)
            gp, sp, gt, st = [x.values.astype(float) for x in (gp, sp, gt, st)]
            rows_by_code = {c: np.where(code == c)[0] for c in codes_m}
            rp = wcorr(gp, sp); rt = wcorr(gt, st); bs = np.empty(NB)
            for b in range(NB):
                keep = np.concatenate([rows_by_code[c] for c in rng.choice(codes_m, len(codes_m))])
                bs[b] = np.nanmean(wcorr(gp[keep], sp[keep]) - wcorr(gt[keep], st[keep]))
            d = rp - rt
            wrows.append(dict(외부지표=nm, 신호=sg, 제공_r=round(np.nanmean(rp), 4), TXT_r=round(np.nanmean(rt), 4), 차이=round(np.nanmean(d), 4),
                              CI_lo=round(np.nanpercentile(bs, 2.5), 4), CI_hi=round(np.nanpercentile(bs, 97.5), 4), 제공우세_원점=int(np.nansum(d > 0)), 원점=int(np.sum(~np.isnan(d)))))
    W = pd.DataFrame(wrows); W.to_csv(f'{OUTP}/s6_within__{ARM}.csv', index=False, encoding='utf-8-sig'); print(W.to_string(), round(time.time() - T_START)); sys.exit(0)

# ---------- (3) 연결 수 ----------
ta = pd.read_parquet(TXT_ASSIGN, columns=['cc', 'ksic_txt', 'ksic_prov', 'method'])
docs_txt = ta.ksic_txt.value_counts().reindex(K50).fillna(0); kr_txt = ta[ta.cc == 'KR'].ksic_txt.value_counts().reindex(K50).fillna(0)
kr_text = ta[(ta.cc == 'KR') & (ta.method == 'text')].ksic_txt.value_counts().reindex(K50).fillna(0)
docs_prov = pd.read_parquet(os.path.join(T3, 'R3', 'cache_pub', 'base_rows.parquet'), columns=['ksic']).ksic.value_counts().reindex(K50).fillna(0)
el_t = T50[T50.t0.isin(ORIG)].groupby(['ksic', 't0']).size().unstack().reindex(index=K50, columns=ORIG).fillna(0)
el_p = P50[P50.t0.isin(ORIG)].groupby(['ksic', 't0']).size().unstack().reindex(index=K50, columns=ORIG).fillna(0)
kos = ext['출하액'].notna().any(axis=1) & ext['종사자수'].notna().any(axis=1); hs = ext['수출액'].notna().any(axis=1)
uniq = (km.value_counts() == 1); n_unique5 = int(km.map(uniq).sum())


def link(has_docs):
    return dict(KOSIS5=int((has_docs & kos).sum()), HS6=int((has_docs & hs).sum()))


LK = dict(
    txt_docs_ge1=link(docs_txt > 0), txt_kr_text_docs_ge1=link((kr_text > 0) | (~km.reindex(K50).isin(km[MULTI].unique()).values)),
    txt_eligible_cell_every_origin=link((el_t > 0).all(axis=1)), txt_eligible_cell_mean_ge1=link(el_t.mean(axis=1) >= 1),
    prov_docs_ge1=link(docs_prov > 0), prov_eligible_cell_every_origin=link((el_p > 0).all(axis=1)),
    concordance_unique_5digit=n_unique5,
    txt_vs_prov_docs_ratio=dict(median=float((docs_txt / docs_prov).median()), min=float((docs_txt / docs_prov).min()), max=float((docs_txt / docs_prov).max()),
                                n_ratio_lt_0_5=int(((docs_txt / docs_prov) < 0.5).sum()), n_ratio_gt_2=int(((docs_txt / docs_prov) > 2).sum())),
    mean_eligible_cells_per_origin=dict(txt=float(el_t.sum().mean()), prov=float(el_p.sum().mean())),
    method_share=ta.method.value_counts(normalize=True).round(4).to_dict(),
    method_share_multi34=ta[ta.ksic_txt.isin(MULTI)].method.value_counts(normalize=True).round(4).to_dict(),
    per_ksic=pd.DataFrame(dict(docs_txt=docs_txt, kr_txt=kr_txt, kr_text=kr_text, docs_prov=docs_prov, elig_txt_mean=el_t.mean(axis=1).round(1),
                               elig_prov_mean=el_p.mean(axis=1).round(1), kosis=kos, hs6=hs)).reset_index().rename(columns={'index': 'ksic'}).to_dict('records'))
json.dump(dict(link=LK), open(f'{OUTP}/s6_meta__{ARM}.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=float)
print(json.dumps({k: v for k, v in LK.items() if k != 'per_ksic'}, ensure_ascii=False, indent=1))
