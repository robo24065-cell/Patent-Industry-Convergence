# -*- coding: utf-8 -*-
"""R8 s05: (3) 외부 결과 정합 — 셀/mg 부상 신호를 산업으로 모아 이후 산업 외부지표(KOSIS 광업제조업조사 5자리·수출입) 성장과 맞춰 본다.
개발 원점 t0 = 2010–2018. 외부 성장 g = log(평균 Y[t0+1..t0+3]) − log(평균 Y[t0−2..t0]) (있는 연도만; KOSIS 10차 2008–2020, 2010 없음; 무역 ≤2021).
신호(산업 × t0):
  cell_real / cell_pred : 그 산업 셀의 부상 실현율 / 모형 예측(원점 내 백분위) 평균
       설계 A(50 KSIC): 제공 = R7 동결 재현 50셀(pub_full_preds), 연계표 = R7 conc_task 27코드 값을 같은 코드의 KSIC에 공유
       설계 B(27코드): 제공 = P27 과제(pcode_feat-pcode), 연계표 = C27 과제(conc_feat-conc)
  bridge_real / bridge_pred : **라벨 무관** mg 부상(실현 yb, 라벨 무관 모형 M_none 예측)을 산업으로 옮김. 가중 s_g(mg) = 산업 g의 t0−2..t0 5개국 문서 중 mg 비중.
       제공 = 제공 KSIC 패널(설계 B는 P27 패널), 연계표 = 연계표 패널. 결과·예측은 양쪽이 똑같고 '다리'만 다르다.
  pat_mom : 산업 KR 특허 3년 모멘텀(참고)
지표: 원점별 산업 간 Spearman → 9원점 평균. 차이 CI = 산업 복원추출 1,000회(원점 공통 표본)."""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); T3 = os.path.dirname(HERE); E7 = os.path.join(T3, 'ESSENTIAL_R7')
BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"; SHX = os.path.join(BASE, r"분석\모델경쟁_20260927\_shared\ext")
ORIG = list(range(2010, 2019)); Y0, Y1 = 2000, 2021; NB = 1000; rng = np.random.default_rng(7)
km = pd.read_csv(os.path.join(E7, 'conc_ksic_map.csv'), dtype=str).set_index('ksic').code
K50 = sorted(km.index); CODES = sorted(km.unique())

# ---------- 외부 지표 ----------
ko = pd.read_csv(os.path.join(BASE, r"공모전데이터\작업데이터\kosis_광업제조업_50개산업.csv"), dtype={"세세분류코드": str})
ko = ko[(ko.기준 == "10차") & ko.항목코드.isin(["T01", "T02", "T04"])]
ext, linkinfo = {}, {}
for code, nm in (("T04", "출하액"), ("T02", "종사자수"), ("T01", "사업체수")):
    z = ko[ko.항목코드 == code]
    linkinfo[nm] = dict(n_ksic=int(z.세세분류코드.nunique()), masked_rows=int((z.마스킹 != 0).sum()))
    ext[nm] = z[z.마스킹 == 0].pivot_table(index="세세분류코드", columns="연도", values="값", aggfunc="sum").reindex(index=K50)
mp = pd.read_csv(os.path.join(BASE, r"분석\fusion\ksic_hs6_map.csv"), dtype=str).drop_duplicates(["k5", "hs6"])
mp["w"] = 1.0 / mp.groupby("hs6").k5.transform("nunique")
ex = pd.read_csv(os.path.join(SHX, "trade_kr_export_hs6_uv_2000_2021.csv"), dtype={"hs6": str}, usecols=["year", "hs6", "export_usd"])
im = pd.read_csv(os.path.join(BASE, r"분석\fusion\hs6_import_yearly.csv"), dtype={"hs6": str}); im = im[im.partner == "WLD"].rename(columns={"value": "import_usd"})
for nm, d, col in (("수출액", ex, "export_usd"), ("수입액", im, "import_usd")):
    z = d[d.year <= 2021].merge(mp, on="hs6"); z["v"] = z[col] * z.w
    ext[nm] = z.pivot_table(index="k5", columns="year", values="v", aggfunc="sum").reindex(index=K50)
    linkinfo[nm] = dict(n_ksic=int(ext[nm].notna().any(axis=1).sum()))
for nm in ext: assert max(ext[nm].columns) <= 2021


def growth(Yd, t0):
    f = [y for y in range(t0 + 1, t0 + 4) if y in Yd.columns]; p = [y for y in range(t0 - 2, t0 + 1) if y in Yd.columns]
    return np.log(Yd[f].mean(axis=1, skipna=True)) - np.log(Yd[p].mean(axis=1, skipna=True))


def to_code(Yd):   # 코드 합: 모든 연도에 값이 있는 KSIC만 합산(구성 변화 방지)
    full = Yd.dropna(how='any'); return full.groupby(km.reindex(full.index).values).sum()


G = {('A', nm): pd.DataFrame({t0: growth(Yd, t0) for t0 in ORIG}) for nm, Yd in ext.items()}
G.update({('B', nm): pd.DataFrame({t0: growth(to_code(Yd), t0) for t0 in ORIG}).reindex(CODES) for nm, Yd in ext.items()})


# ---------- 신호 ----------
def cell_sig(preds, key):
    d = preds[preds.t0.isin(ORIG)].copy(); d['r'] = d.groupby('t0').p.rank(pct=True); d['g'] = d.ksic if key is None else d.ksic.map(key)
    real = d.groupby(['g', 't0']).yb.mean().unstack(); pred = d.groupby(['g', 't0']).r.mean().unstack()
    return real, pred


def panel_tot(fn):
    df = pd.read_parquet(fn, columns=['ksic', 'mg', 'cc', 'year', 'n'])
    df['kr'] = np.where(df.cc == 'KR', df.n, 0.0)
    return df.groupby(['ksic', 'mg', 'year'])[['n', 'kr']].sum().reset_index()


mgp = pd.read_parquet(os.path.join(HERE, 'results', 'mg_ctx-none_preds.parquet')); mgp = mgp[mgp.t0.isin(ORIG)].copy()
mgp['r'] = mgp.groupby('t0').p.rank(pct=True)


def bridge_sig(pan):
    real, pred, mom = {}, {}, {}
    for t0 in ORIG:
        w = pan[(pan.year >= t0 - 2) & (pan.year <= t0)].groupby(['ksic', 'mg']).n.sum().reset_index()
        w['s'] = w.n / w.groupby('ksic').n.transform('sum')
        m = w.merge(mgp[mgp.t0 == t0][['mg', 'yb', 'r']], on='mg')   # 적격 mg 만
        den = m.groupby('ksic').s.sum()
        real[t0] = (m.s * m.yb).groupby(m.ksic).sum() / den; pred[t0] = (m.s * m.r).groupby(m.ksic).sum() / den
        k0 = pan[(pan.year >= t0 - 2) & (pan.year <= t0)].groupby('ksic').kr.sum(); k1 = pan[(pan.year >= t0 - 5) & (pan.year <= t0 - 3)].groupby('ksic').kr.sum()
        mom[t0] = np.log((k0 + 1) / (k1.reindex(k0.index).fillna(0) + 1))
    return pd.DataFrame(real), pd.DataFrame(pred), pd.DataFrame(mom)


P50 = pd.read_parquet(os.path.join(E7, 'results', 'pub_full_preds.parquet')); C27r7 = pd.read_parquet(os.path.join(E7, 'results', 'pub_conc_task_preds.parquet'))
P27 = pd.read_parquet(os.path.join(HERE, 'results', 'pcode_feat-pcode_preds.parquet')); C27 = pd.read_parquet(os.path.join(HERE, 'results', 'conc_feat-conc_preds.parquet'))
pan_prov = panel_tot(os.path.join(T3, 'R3', 'cache_pub', 'panel_ccy.parquet'))
pan_pcode = panel_tot(os.path.join(HERE, 'cache_pcode_pub', 'panel_ccy.parquet'))
pan_conc = panel_tot(os.path.join(E7, 'cache_conc_pub', 'panel_ccy.parquet'))
S = {}
S[('A', 'prov', 'cell_real')], S[('A', 'prov', 'cell_pred')] = cell_sig(P50, None)
cr, cp = cell_sig(C27r7, None)
expand = lambda X: X.reindex(km.reindex(K50).values).set_axis(K50)   # 코드 값 → 같은 코드의 KSIC 에 공유
S[('A', 'conc', 'cell_real')], S[('A', 'conc', 'cell_pred')] = expand(cr), expand(cp)
br = bridge_sig(pan_prov); S[('A', 'prov', 'bridge_real')], S[('A', 'prov', 'bridge_pred')], S[('A', 'prov', 'pat_mom')] = br
bc = bridge_sig(pan_conc); S[('A', 'conc', 'bridge_real')], S[('A', 'conc', 'bridge_pred')], S[('A', 'conc', 'pat_mom')] = [expand(x) for x in bc]
S[('B', 'prov', 'cell_real')], S[('B', 'prov', 'cell_pred')] = cell_sig(P27, None)
S[('B', 'conc', 'cell_real')], S[('B', 'conc', 'cell_pred')] = cell_sig(C27, None)
bp = bridge_sig(pan_pcode); S[('B', 'prov', 'bridge_real')], S[('B', 'prov', 'bridge_pred')], S[('B', 'prov', 'pat_mom')] = bp
S[('B', 'conc', 'bridge_real')], S[('B', 'conc', 'bridge_pred')], S[('B', 'conc', 'pat_mom')] = bc
for k in S: S[k] = S[k].reindex(index=K50 if k[0] == 'A' else CODES, columns=ORIG)


def rho_cols(A, Bm):
    """A, Bm: (n, 9) 배열. 열마다 결측 제외 Spearman. 반환 길이 9."""
    out = np.full(A.shape[1], np.nan)
    for j in range(A.shape[1]):
        a = A[:, j]; b = Bm[:, j]; ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 8: continue
        ra = rankdata(a[ok]); rb = rankdata(b[ok])
        if ra.std() == 0 or rb.std() == 0: continue
        out[j] = np.corrcoef(ra, rb)[0, 1]
    return out


rows = []
for design in ('A', 'B'):
    units = np.array(K50 if design == 'A' else CODES)
    for nm in ext:
        g = G[(design, nm)]
        valid = units[g.reindex(units).notna().any(axis=1).values]
        gv = g.loc[valid, ORIG].values.astype(float)
        for sg in ('cell_real', 'cell_pred', 'bridge_real', 'bridge_pred', 'pat_mom'):
            sp = S[(design, 'prov', sg)].loc[valid, ORIG].values.astype(float); sc = S[(design, 'conc', sg)].loc[valid, ORIG].values.astype(float)
            rp, rc = rho_cols(sp, gv), rho_cols(sc, gv)
            bs = np.empty(NB)
            for b in range(NB):
                ii = rng.integers(0, len(valid), len(valid))
                bs[b] = np.nanmean(rho_cols(sp[ii], gv[ii]) - rho_cols(sc[ii], gv[ii]))
            d = rp - rc
            rows.append(dict(설계=design, 외부지표=nm, 신호=sg, n산업=len(valid), 제공_rho=round(np.nanmean(rp), 4), 연계표_rho=round(np.nanmean(rc), 4),
                             차이=round(np.nanmean(d), 4), CI_lo=round(np.nanpercentile(bs, 2.5), 4), CI_hi=round(np.nanpercentile(bs, 97.5), 4),
                             제공우세_원점=int(np.nansum(d > 0)), 원점=int(np.sum(~np.isnan(d)))))
        print(design, nm, 'done', flush=True)
O = pd.DataFrame(rows); O.to_csv(os.path.join(HERE, 'external_alignment.csv'), index=False, encoding='utf-8-sig')
print(O.to_string())
summ = O.groupby(['설계', '신호']).apply(lambda g: pd.Series(dict(비교=len(g), 제공우세=int((g.차이 > 0).sum()), CI위=int((g.CI_lo > 0).sum()), CI아래=int((g.CI_hi < 0).sum())))).reset_index()
print(summ.to_string())
# 연결 기록: 5자리 직접 연결 가능 수
uniq = (km.value_counts() == 1); n_unique5 = int(km.map(uniq).sum())
link = dict(provided_label_KOSIS5_link=f"{linkinfo['출하액']['n_ksic']}/50", provided_label_HS6_link=f"{linkinfo['수출액']['n_ksic']}/50",
            concordance_unique_5digit=f"{n_unique5}/50", concordance_codes=len(CODES), kosis_masked_rows=linkinfo, note='연계표 코드는 4자리 수준(27개)이라 5자리 KOSIS·HS6 연결은 코드 안 KSIC 가 하나인 경우에만 정해진다')
json.dump(dict(link=link, summary=summ.to_dict('records')), open(os.path.join(HERE, 'external_alignment_meta.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps(link, ensure_ascii=False))
