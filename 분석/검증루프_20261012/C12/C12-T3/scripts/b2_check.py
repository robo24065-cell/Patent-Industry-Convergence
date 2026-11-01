# -*- coding: utf-8 -*-
"""(B1·B2) 받은 표 점검(data_check (a)–(d)), 단절 크기, 시나리오 피처(S0 재계산·S1·S1a·S2) 생성.
입력(읽기만): 작업 원천 CSV(PI_CSV), download/old9·new10(이번 바퀴 KOSIS), C07 연계표, C09 v0_D_final(T3 KSIC 목록·S0 피처 대조).
출력: cache/b_pi_long.parquet, cache/b_feats.parquet, cache/b2_check.json, cache/b2_break_by_code.csv"""
import os, sys, json, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *
from kjson import loadj
import numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
OUT = {}
# ---------- 원천 ----------
csv = pd.read_csv(PI_CSV, dtype={'산업코드': str}); csv['값'] = pd.to_numeric(csv.값, errors='coerce')
csv = csv[csv.항목코드.isin(['T10', 'T11'])].rename(columns={'항목코드': 'it', '연월': 'ym', '산업코드': 'code', '값': 'v_csv'})[['it', 'ym', 'code', 'v_csv']]
def load_dl(sub):
    rows = []
    for fp in sorted(glob.glob(os.path.join(DL, sub, '*.json'))):
        for e in loadj(fp):
            rows.append(dict(it=e['ITM_ID'], ym=int(e['PRD_DE']), code=e['C2'], name=e['C2_NM'], v=pd.to_numeric(e['DT'], errors='coerce'), lst=e.get('LST_CHN_DE'), unit=e.get('UNIT_NM')))
    return pd.DataFrame(rows)
old = load_dl('old9'); new = load_dl('new10')
OUT['old_rows'] = len(old); OUT['new_rows'] = len(new)
OUT['old_dup'] = int(old.duplicated(['it', 'ym', 'code']).sum()); OUT['new_dup'] = int(new.duplicated(['it', 'ym', 'code']).sum())
OUT['old_unit'] = old.unit.unique().tolist(); OUT['new_unit'] = new.unit.unique().tolist()
OUT['old_ym'] = [int(old.ym.min()), int(old.ym.max())]; OUT['new_ym'] = [int(new.ym.min()), int(new.ym.max())]
OUT['old_lst_by_year'] = old.assign(y=old.ym // 100).groupby('y').lst.agg(lambda s: sorted(set(s))).to_dict()
OUT['new_lst_by_year'] = new.assign(y=new.ym // 100).groupby('y').lst.agg(lambda s: sorted(set(s))).to_dict()
# (a) 신표 다운로드 = 작업 CSV ?
a = new.merge(csv, on=['it', 'ym', 'code'], how='outer', indicator=True)
OUT['a_new_vs_csv_merge'] = a._merge.value_counts().to_dict()
b = a[a._merge == 'both']; OUT['a_new_vs_csv_maxabs'] = float(np.nanmax(np.abs(b.v - b.v_csv))); OUT['a_new_vs_csv_n'] = int(len(b))
OUT['a_new_vs_csv_mismatch_n'] = int((np.abs(b.v - b.v_csv) > 1e-9).sum())
# 총지수 2010 연평균: 구표 = 100(2010=100), 신표 2020 연평균 = 100(2020=100) 점검
def ann(df, code, it, y): s = df[(df.code == code) & (df.it == it) & (df.ym // 100 == y)].v; return float(s.mean()) if len(s) == 12 else None
OUT['a_old_total_2010_mean'] = ann(old, '0', 'T10', 2010); OUT['a_old_C_2010_mean'] = ann(old, 'C', 'T10', 2010)
OUT['a_new_total_2020_mean'] = ann(new, '0', 'T10', 2020) if new.ym.max() >= 202012 else None
# 공표 수치 대조(보도자료 수준): 전국 광공업(A) 생산 연간 증감률 구표 vs 신표, 2011–2017
def yoy(df, code):
    s = df[(df.code == code) & (df.it == 'T10')].assign(y=lambda d: d.ym // 100).groupby('y').v.agg(['mean', 'count'])
    s = s[s['count'] == 12]['mean']; return (s / s.shift(1) - 1) * 100
yo = pd.DataFrame(dict(old=yoy(old, 'A'), new=yoy(new, 'A'))).loc[2006:2017].round(2)
OUT['a_yoy_A_old_new'] = yo.reset_index().to_dict('records')
# (b) 결측·코드 범위
OUT['b_old_codes'] = int(old.code.nunique()); OUT['b_new_codes'] = int(new.code.nunique())
oc = set(old.code); nc_ = set(new.code)
OUT['b_codes_only_old'] = sorted(oc - nc_); OUT['b_codes_only_new'] = sorted(nc_ - oc)
cnt_old = old.groupby(['code', 'it']).ym.nunique(); OUT['b_old_incomplete'] = cnt_old[cnt_old < cnt_old.max()].reset_index().to_dict('records')
cnt_new = new.groupby(['code', 'it']).ym.nunique(); OUT['b_new_incomplete'] = cnt_new[cnt_new < cnt_new.max()].reset_index().to_dict('records')
OUT['b_old_nan'] = int(old.v.isna().sum()); OUT['b_new_nan'] = int(new.v.isna().sum())
# ---------- T3 코드 ----------
Dk = pd.read_parquet(os.path.join(C09V0, 'v0_D_final.parquet'), columns=['ksic', 't0', 'pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']).drop_duplicates(['ksic', 't0'])
ks_list = sorted(Dk.ksic.unique())
PI = csv.rename(columns={'v_csv': 'v'})
def code10_of(ks):
    c3 = 'C' + ks[:3]; c2 = 'C' + ks[:2]
    return c3 if ((PI.code == c3) & (PI.it == 'T10') & (PI.ym // 100 == 2010)).any() else c2
c10 = {ks: code10_of(ks) for ks in ks_list}
# (c) 9↔10 대응: 연계표 KSIC 10차(신)↔9차(구) 5자리
L = pd.read_csv(os.path.join(C07DL, 'ksic_link', 'ksic_ksco_link_2025Q2.csv'), encoding='cp949', dtype=str)
L = L[(L.분류종류 == 'KSIC') & (L.분류차순정보 == '10')]
OUT['c_link_ranges'] = L.분류범위.value_counts().to_dict()
L5 = L[L.신분류코드.str.len() == 5]
def old9_of(ks):
    olds = sorted(set(L5[L5.신분류코드 == ks].구분류코드))
    g3 = sorted(set(o[:3] for o in olds)); g2 = sorted(set(o[:2] for o in olds))
    return olds, g3, g2
mp = []
for ks in ks_list:
    olds, g3, g2 = old9_of(ks); code10 = c10[ks]
    lvl = len(code10) - 1  # 3 or 2 digits
    if lvl == 3:
        cand = ['C' + g for g in g3]
    else:
        cand = ['C' + g for g in g2]
    cand_in = [c for c in cand if c in oc]
    # 10차 3자리 code10 이 포함하는 10차 5자리 전체가 9차에서 어디서 왔는지(그룹 수준 대응 유형)
    sub10 = L5[L5.신분류코드.str[:lvl] == code10[1:]]
    from_g = sorted(set(sub10.구분류코드.str[:lvl]))
    # 변경 기록 3: 피처가 3(2)자리 지수이므로 대응도 그룹 수준으로 — 10차 그룹이 온 9차 그룹(from_g) 중 같은 번호가 있으면 그것(주 대응), 없고 하나뿐이면 그것, 아니면 NaN
    own = code10[1:]
    base9 = own if own in from_g else (from_g[0] if len(from_g) == 1 else None)
    old_g_cover = L5[L5.구분류코드.str[:lvl] == base9] if base9 else L5.iloc[0:0]
    to_g = sorted(set(old_g_cover.신분류코드.str[:lvl]))
    typ = ('1:1' if (from_g == [own] and to_g == [own]) else ('같은번호·분할/통합' if own in from_g else '다른번호'))
    code9 = ('C' + base9) if (base9 and ('C' + base9) in oc) else None
    code9_ksic5 = cand_in[0] if len(cand_in) == 1 else None
    mp.append(dict(ksic=ks, code10=code10, olds5=';'.join(olds), cand9=';'.join(cand), code9=code9, code9_ksic5=code9_ksic5, group_from9=';'.join(from_g), group_to10=';'.join(to_g), type=typ))
MP = pd.DataFrame(mp); OUT['c_map_types_ksic'] = MP.type.value_counts().to_dict(); OUT['c_map_code9_missing'] = int(MP.code9.isna().sum())
OUT['c_map_by_code10'] = MP.drop_duplicates('code10')[['code10', 'cand9', 'code9', 'group_from9', 'group_to10', 'type']].to_dict('records')
OUT['c_n_ksic_by_code10'] = MP.code10.value_counts().to_dict()
# ---------- 월→창 연평균 ----------
def win(df, maxym=202111):
    d = df[df.ym <= maxym].copy(); d['y'] = d.ym // 100; d['m'] = d.ym % 100; d['wy'] = np.where(d.m == 12, d.y + 1, d.y)
    g = d.groupby(['code', 'it', 'wy']).v.agg(['mean', 'count']); return g[g['count'] == 12]['mean']
Wn = win(PI); Wo = win(old.rename(columns={'v': 'v'}))
# (B2) 단절 크기: 창 연도 성장률 log(A_wy/A_wy-1), 신(code10) − 구(code9)
rows = []
for r in MP.drop_duplicates('code10').itertuples():
    for it in ['T10', 'T11']:
        for wy in range(2007, 2018):
            gn = np.log(Wn.get((r.code10, it, wy), np.nan) / Wn.get((r.code10, it, wy - 1), np.nan))
            go_same = np.log(Wo.get((r.code10, it, wy), np.nan) / Wo.get((r.code10, it, wy - 1), np.nan)) if r.code10 in oc else np.nan
            go = np.log(Wo.get((r.code9, it, wy), np.nan) / Wo.get((r.code9, it, wy - 1), np.nan)) if r.code9 else np.nan
            rows.append(dict(code10=r.code10, code9=r.code9, type=r.type, it=it, wy=wy, g_new=gn, g_old9=go, g_old_samenum=go_same, gap=gn - go))
BR = pd.DataFrame(rows); BR.to_csv(os.path.join(CACHE, 'b2_break_by_code.csv'), index=False, encoding='utf-8-sig')
def summ(x): x = x.dropna(); return dict(n=int(len(x)), mean=float(x.mean()), mad=float(x.abs().mean()), p90=float(x.abs().quantile(0.9)) if len(x) else None, max=float(x.abs().max()) if len(x) else None)
OUT['B2_diff_pre2015'] = summ(BR[(BR.wy <= 2014)].gap); OUT['B2_diff_wy2015'] = summ(BR[BR.wy == 2015].gap)
OUT['B2_diff_wy2016_2017'] = summ(BR[BR.wy >= 2016].gap)
OUT['B2_diff_by_wy'] = {int(w): summ(g['gap']) for w, g in BR.groupby('wy')}
OUT['B2_diff_wy2015_by_type'] = {t: summ(g['gap']) for t, g in BR[BR.wy == 2015].groupby('type')}
OUT['B2_diff_wy2016_17_by_type'] = {t: summ(g['gap']) for t, g in BR[BR.wy >= 2016].groupby('type')}
# 신표 2014 이전 = 구표 재기준? (같은 번호 코드, 월별 비율의 변동계수)
rat = PI[PI.ym <= 201412].merge(old[old.ym <= 201412][['it', 'ym', 'code', 'v']].rename(columns={'v': 'vo'}), on=['it', 'ym', 'code'])
rat['r'] = rat.v / rat.vo
cv = rat.groupby(['code', 'it']).r.agg(lambda s: float(s.std() / s.mean()))
OUT['B2_pre2015_ratio_cv'] = dict(n=int(len(cv)), median=float(cv.median()), p90=float(cv.quantile(0.9)), max=float(cv.max()), n_cv_lt_1e3=int((cv < 1e-3).sum()))
OUT['B2_pre2015_ratio_cv_T3codes'] = {c: float(cv.get((c, 'T10'), np.nan)) for c in sorted(set(MP.code10))}
# 겹침 2015–2017 월별 비율 변동(신/구)
rat2 = PI[(PI.ym >= 201501) & (PI.ym <= 201712)].merge(old[(old.ym >= 201501)][['it', 'ym', 'code', 'v']].rename(columns={'v': 'vo'}), on=['it', 'ym', 'code'])
rat2['r'] = rat2.v / rat2.vo; cv2 = rat2.groupby(['code', 'it']).r.agg(lambda s: float(s.std() / s.mean()))
OUT['B2_overlap_ratio_cv_T3codes'] = {c: float(cv2.get((c, 'T10'), np.nan)) for c in sorted(set(MP.code10))}
# ---------- 피처 ----------
lg = lambda a_, b_: np.log(a_ / b_) if (a_ == a_ and b_ == b_ and a_ > 0 and b_ > 0) else np.nan
def feats(Wsrc, code, t0, override=None):
    def p(it, y):
        if override is not None:
            v = override(code, it, y)
            if v is not None: return v
        return Wsrc.get((code, it, y), np.nan)
    return dict(pi_prod_g2=lg((p('T10', t0) + p('T10', t0 - 1)) / 2, (p('T10', t0 - 2) + p('T10', t0 - 3)) / 2),
                pi_prod_g1=lg(p('T10', t0), p('T10', t0 - 1)),
                pi_ship_g2=lg((p('T11', t0) + p('T11', t0 - 1)) / 2, (p('T11', t0 - 2) + p('T11', t0 - 3)) / 2))
T0S = list(range(2005, 2019)) + [2021]
# S2 t0=2018: 신표 wy2015 창의 2014-12 값을 구표 전년동월비로 연쇄(신 2015-12 × 구 2014-12/구 2015-12)
def wy2015_chained(code10, code9, it):
    d = PI[(PI.code == code10) & (PI.it == it)].set_index('ym').v
    if code9 is None or not ((old.code == code9) & (old.it == it)).any(): return np.nan
    o = old[(old.code == code9) & (old.it == it)].set_index('ym').v
    dec14 = d.get(201512, np.nan) * o.get(201412, np.nan) / o.get(201512, np.nan)
    vals = [dec14] + [d.get(201500 + mm, np.nan) for mm in range(1, 12)]
    return float(np.mean(vals)) if all(v == v for v in vals) else np.nan
rows = []
for r in MP.itertuples():
    for t0 in T0S:
        f0 = feats(Wn, r.code10, t0)
        # S1 엄격: 2014-12 이전 달과 2015-01 이후 달을 함께 쓰는 값 NaN. 창 wy=2015 가 걸침(12월 2014 + 1–11월 2015).
        # g1 은 창 t0, t0-1 → 2015 ∈ {t0, t0-1} ⇔ t0 ∈ {2015, 2016}; g2 는 창 t0..t0-3 → t0 ∈ 2015..2018. 또 창 경계가 아닌 혼합(창 2014 와 2016 의 비)도
        # 2015 를 사이에 두므로 2014 이전 창과 2015 이후 창을 함께 쓰는 경우 = g1 t0∈{2015,2016}, g2 t0∈2015..2018(t0-3 ≤ 2014 < t0 포함) 로 같다.
        f1 = dict(f0)
        if t0 in (2015, 2016): f1['pi_prod_g1'] = np.nan
        if 2015 <= t0 <= 2018: f1['pi_prod_g2'] = np.nan; f1['pi_ship_g2'] = np.nan
        f1a = dict(f0)
        if t0 == 2016: f1a['pi_prod_g1'] = np.nan
        if 2016 <= t0 <= 2018: f1a['pi_prod_g2'] = np.nan; f1a['pi_ship_g2'] = np.nan
        f2 = dict(f0)
        if 2015 <= t0 <= 2017:
            if r.code9:
                fo = feats(Wo, r.code9, t0)
                if t0 in (2015, 2016): f2['pi_prod_g1'] = fo['pi_prod_g1']
                f2['pi_prod_g2'] = fo['pi_prod_g2']; f2['pi_ship_g2'] = fo['pi_ship_g2']
            else:
                if t0 in (2015, 2016): f2['pi_prod_g1'] = np.nan
                f2['pi_prod_g2'] = np.nan; f2['pi_ship_g2'] = np.nan
        elif t0 == 2018:
            ov = lambda code, it, y: (wy2015_chained(r.code10, r.code9, it) if y == 2015 else None)
            fc = feats(Wn, r.code10, t0, ov); f2['pi_prod_g2'] = fc['pi_prod_g2']; f2['pi_ship_g2'] = fc['pi_ship_g2']
        rec = dict(ksic=r.ksic, t0=t0)
        for tag, f in [('S0', f0), ('S1', f1), ('S1a', f1a), ('S2', f2)]:
            for k, v in f.items(): rec[f'{k}__{tag}'] = v
        rows.append(rec)
F = pd.DataFrame(rows)
# B0: S0 재계산 = v0_D_final ?
chk = Dk.merge(F, on=['ksic', 't0'], how='left')
OUT['B0_pi_recalc_maxdiff'] = {k: float(np.nanmax(np.abs(chk[k].astype(float) - chk[k + '__S0']))) for k in ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']}
OUT['B0_pi_nan_agree'] = {k: bool((chk[k].isna() == chk[k + '__S0'].isna()).all()) for k in ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']}
# S0 대 S2 값 차(걸치는 t0)
for k in ['pi_prod_g2', 'pi_prod_g1', 'pi_ship_g2']:
    dd = F[k + '__S2'] - F[k + '__S0']
    OUT[f'S2_minus_S0_{k}'] = {int(t): summ(dd[F.t0 == t]) for t in range(2015, 2019)}
    OUT[f'S2_nan_{k}'] = {int(t): int(F[(F.t0 == t)][k + '__S2'].isna().sum()) for t in range(2015, 2019)}
F.to_parquet(os.path.join(CACHE, 'b_feats.parquet'), index=False); MP.to_csv(os.path.join(CACHE, 'b2_map_ksic.csv'), index=False, encoding='utf-8-sig')
pd.concat([PI.assign(src='csv'), old[['it', 'ym', 'code', 'v']].assign(src='old9'), new[['it', 'ym', 'code', 'v']].assign(src='new10dl')]).to_parquet(os.path.join(CACHE, 'b_pi_long.parquet'), index=False)
json.dump(OUT, open(os.path.join(CACHE, 'b2_check.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else str(o))
for k, v in OUT.items():
    s = json.dumps(v, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, 'item') else str(o)); print(k, s[:900])
