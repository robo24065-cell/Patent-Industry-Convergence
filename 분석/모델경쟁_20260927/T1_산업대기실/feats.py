# 확장 피처: 노출 집계 변형, 시간 모멘텀, 관련산업 확산, 산업 모멘텀, 공동출원 파트너
import pandas as pd, numpy as np
from lib import *

def agg_feats(df, ctx, P, pre='t'):
    """노출 집계 변형: 평균/최대/상위3평균/로그합/순위/엔트로피/최근성 가중/최근3년"""
    op = ctx['op']; fi = ctx['fi']; nF = len(ctx['firms']); cut = ctx['cut']
    rows = fi[op.applicant_cd].values; Po = P[op.row.values]; ay = op.ay.values
    f, k = df.fidx.values, df.k.values
    T = np.zeros((nF, 50), np.float32); np.add.at(T, rows, Po)
    n = np.bincount(rows, minlength=nF).astype(np.float32)
    Tm = T / np.maximum(n, 1)[:, None]
    df[f'{pre}_mean'] = Tm[f, k]
    df[f'{pre}_lsum'] = np.log1p(T[f, k])
    MX = np.zeros((nF, 50), np.float32); np.maximum.at(MX, rows, Po)
    df[f'{pre}_max'] = MX[f, k]
    H = np.zeros((nF, 50), np.float32); m = Po.max(1) > 0.5
    np.add.at(H, (rows[m], Po[m].argmax(1)), 1)
    df[f'{pre}_hi'] = np.log1p(H[f, k])
    H2 = np.zeros((nF, 50), np.float32); np.add.at(H2, rows, (Po > 0.2).astype(np.float32))
    df[f'{pre}_n20'] = np.log1p(H2[f, k])
    # 상위3 평균: 법인별로 k열 상위 3개 확률 평균 (정렬 기반)
    order = np.argsort(rows, kind='stable'); r_s = rows[order]; P_s = Po[order]
    starts = np.searchsorted(r_s, np.arange(nF)); ends = np.searchsorted(r_s, np.arange(nF), side='right')
    TOP3 = np.zeros((nF, 50), np.float32)
    for i in np.nonzero(ends > starts)[0]:
        blk = P_s[starts[i]:ends[i]]
        if len(blk) <= 3: TOP3[i] = blk.mean(0)
        else: TOP3[i] = np.sort(blk, axis=0)[-3:].mean(0)
    df[f'{pre}_top3'] = TOP3[f, k]
    # 법인 내 순위(해당 k가 법인 노출 중 몇 위인지) 와 법인 노출 엔트로피
    rk = (-Tm).argsort(1).argsort(1)
    df[f'{pre}_rank'] = rk[f, k].astype(np.float32)
    pn = Tm / np.maximum(Tm.sum(1, keepdims=True), 1e-9)
    ent = -(pn * np.log(np.maximum(pn, 1e-12))).sum(1)
    df[f'{pre}_ent'] = ent[f]
    # 최근성 가중(반감기 3년), 최근 3년 평균, 모멘텀
    w = np.power(0.5, (cut - ay) / 3.0).astype(np.float32)
    TW = np.zeros((nF, 50), np.float32); np.add.at(TW, rows, Po * w[:, None]); W = np.bincount(rows, weights=w, minlength=nF)
    df[f'{pre}_rw'] = (TW / np.maximum(W, 1e-9)[:, None])[f, k]
    rec = ay > cut - 3
    TR = np.zeros((nF, 50), np.float32); np.add.at(TR, rows[rec], Po[rec]); nr = np.bincount(rows[rec], minlength=nF).astype(np.float32)
    TE = T - TR; ne = n - nr
    rmean = TR / np.maximum(nr, 1)[:, None]; emean = TE / np.maximum(ne, 1)[:, None]
    df[f'{pre}_rec3'] = rmean[f, k]
    df[f'{pre}_mom'] = np.where((nr[f] > 0) & (ne[f] > 0), rmean[f, k] - emean[f, k], 0.0)
    df[f'{pre}_lrec_n'] = np.log1p(nr[f])
    df[f'{pre}_grow'] = np.log1p(nr[f]) - np.log1p(ne[f] * 3.0 / max(cut - 2000 - 2, 1))
    # 관련산업 확산: 법인 노출을 산업 근접도로 전파
    prox = ctx['prox']
    NB = Tm @ prox / np.maximum(prox.sum(0), 1e-9)
    df[f'{pre}_nbr'] = NB[f, k]
    return df

def industry_mom(df, ctx):
    """산업 수준 모멘텀(기준시점까지 알 수 있는 범위안 출원 추세): 최근3년/이전3년 출원 성장, 최근 신규진입 법인 수"""
    D = load(); ins = D['ins']; kidx = D['kidx']; cut = ctx['cut']
    if ctx['cap'] is not None: ins = ins[ins.vis <= ctx['cap']]
    a = ins[(ins.ay > cut - 3) & (ins.ay <= cut)].ksic.map(kidx).value_counts().reindex(range(50), fill_value=0).values
    b = ins[(ins.ay > cut - 6) & (ins.ay <= cut - 3)].ksic.map(kidx).value_counts().reindex(range(50), fill_value=0).values
    df['k_grow'] = (np.log1p(a) - np.log1p(b))[df.k.values]
    # 최근 3년 동안 그 산업에 처음 들어온 법인 수(모든 법인 기준)
    first = ins[ins.ay <= cut].groupby(['applicant_cd', 'ksic']).ay.min().reset_index()
    ne = first[first.ay > cut - 3].ksic.map(kidx).value_counts().reindex(range(50), fill_value=0).values
    df['k_newent'] = np.log1p(ne)[df.k.values]
    return df

def partner_feats(df, ctx):
    """공동출원 파트너 노출: 범위밖·범위안 자사출원의 공동출원인(이름→코드) 중 k 보유 파트너 비율"""
    D = load(); ins = D['ins']; cut = ctx['cut']; kidx = D['kidx']
    if ctx['cap'] is not None: ins = ins[ins.vis <= ctx['cap']]
    fi = ctx['fi']; nF = len(ctx['firms'])
    # 범위안: 같은 특허에 공동출원한 법인 쌍
    ip = ins[ins.ay <= cut][['rgst_no', 'applicant_cd']]
    pairs = ip.merge(ip, on='rgst_no'); pairs = pairs[pairs.applicant_cd_x != pairs.applicant_cd_y]
    # 범위밖: ainfo 공동출원인 이름을 별칭 사전으로 코드화
    op = ctx['op']; op2 = op[op.ainfo.str.contains(',', na=False)][['applicant_cd', 'ainfo']]
    names = pd.read_parquet(CACHE/'firms.parquet')[['applicant_cd', 'applicant_name']]
    import re
    STRIP = re.compile(r"주식회사|\(주\)|㈜|유한회사|재단법인|사단법인|학교법인|[\s\.\,\-\(\)]")
    nm2cd = dict(zip(names.applicant_name.map(lambda s: STRIP.sub('', str(s))), names.applicant_cd))
    rows = []
    for cd, ai in zip(op2.applicant_cd.values, op2.ainfo.values):
        for p in str(ai).split(','):
            c2 = nm2cd.get(STRIP.sub('', p))
            if c2 is not None and c2 != cd: rows.append((cd, c2))
    pp = pd.DataFrame(rows, columns=['applicant_cd_x', 'applicant_cd_y'])
    allp = pd.concat([pairs[['applicant_cd_x', 'applicant_cd_y']], pp]).drop_duplicates()
    allp = allp[allp.applicant_cd_x.isin(fi.index)]
    # 파트너의 산업 보유(기준시점까지 범위안)
    held = ins[ins.ay <= cut].drop_duplicates(['applicant_cd', 'ksic'])
    hp = allp.merge(held[['applicant_cd', 'ksic']], left_on='applicant_cd_y', right_on='applicant_cd')
    PK = np.zeros((nF, 50), np.float32)
    np.add.at(PK, (fi[hp.applicant_cd_x].values, hp.ksic.map(kidx).values), 1)
    npart = allp.groupby('applicant_cd_x').size().reindex(ctx['firms'], fill_value=0).values.astype(np.float32)
    f, k = df.fidx.values, df.k.values
    df['p_n'] = np.log1p(npart[f]); df['p_k'] = np.log1p(PK[f, k]); df['p_share'] = PK[f, k] / np.maximum(npart[f], 1)
    return df
