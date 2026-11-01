# C12-T4 피처 엔진: 나라 하나를 메모리에 올려 원점별 위험집합·결과·피처를 만든다.
import time, json
import numpy as np, pandas as pd
import scipy.sparse as sp
import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib12 import *

SCL = None


def rownorm_dense(M):
    s = M.sum(1, keepdims=True); s[s == 0] = 1
    return M / s


class Country:
    def __init__(self, cc):
        global SCL
        self.cc = cc
        SCL = json.load(open(C + r'\sc_list.json'))
        self.SCL = np.array(SCL); self.S = len(SCL)
        si = {s: i for i, s in enumerate(SCL)}
        d = pd.read_parquet(C + rf'\docs_{cc}.parquet')
        assert d.pub.max() < CUT
        assert (d.did.values == np.arange(len(d))).all()
        self.docs = d
        self.d_pub = d.pub.values.astype(np.int64)
        self.d_ad = d.ad.values
        self.d_sc = d.sc.map(si).fillna(-1).astype(np.int32).values
        self.d_k = d.k.values.astype(np.int16)
        self.d_year = (self.d_pub // 10000).astype(np.int32)
        a = pd.read_parquet(C + rf'\app_{cc}.parquet')
        keys = pd.Index(a.key.unique())
        self.keys = keys
        aid = keys.get_indexer(a.key).astype(np.int64)
        o = np.lexsort((a.did.values, aid))
        self.m_aid = aid[o]; self.m_did = a.did.values[o].astype(np.int64)
        self.m_pub = self.d_pub[self.m_did]; self.m_sc = self.d_sc[self.m_did]
        ak = a[['key', 'akey']].iloc[o]
        akk = ak.groupby('key').akey.agg(lambda x: x.value_counts().index[0])
        self.akey_of = akk.reindex(keys).values
        self.nF = len(keys)
        ds = pd.read_parquet(C + rf'\dsc_{cc}.parquet')
        self.x_did = ds.did.values.astype(np.int64)
        self.x_sc = ds.sc2.map(si).astype(np.int32).values
        kr = pd.read_parquet(C + r'\docs_KR.parquet')
        self.kr = kr
        self.agg5 = pd.read_parquet(C + r'\agg5.parquet')
        cls = np.array([s[:3] for s in SCL]); sect = np.array([s[0] for s in SCL])
        self.cls_id = pd.factorize(cls)[0]; self.sect_id = pd.factorize(sect)[0]
        self.labels = {'prov': self.d_k.astype(np.int32)}

    # ---------- 기본 집계 ----------
    def mwin(self, lo, hi):
        return (self.m_pub >= lo) & (self.m_pub < hi)

    def dwin(self, lo, hi):
        return (self.d_pub >= lo) & (self.d_pub < hi)

    def active(self, t0):
        w3 = self.mwin(addy(t0, -3), t0)
        n3 = np.bincount(self.m_aid[w3], minlength=self.nF)
        w1 = self.mwin(addy(t0, -1), t0)
        n1 = np.bincount(self.m_aid[w1], minlength=self.nF)
        return np.where((n3 >= 10) & (n1 >= 1))[0]

    def cand(self, t0):
        w = self.dwin(addy(t0, -5), t0) & (self.d_sc >= 0)
        n = np.bincount(self.d_sc[w], minlength=self.S)
        return np.where(n >= 30)[0], n

    def firm_sc_counts(self, ids, lo, hi):
        """ids(전체 기업 인덱스 배열) × S 건수(dense float32)."""
        pos = pd.Index(ids)
        w = self.mwin(lo, hi) & (self.m_sc >= 0)
        r = pos.get_indexer(self.m_aid[w]); ok = r >= 0
        M = np.zeros((len(ids), self.S), np.float32)
        np.add.at(M, (r[ok], self.m_sc[w][ok]), 1)
        return M

    def all_firm_sc(self, lo, hi):
        w = self.mwin(lo, hi) & (self.m_sc >= 0)
        Ms = sp.csr_matrix((np.ones(w.sum(), np.float32), (self.m_aid[w], self.m_sc[w])), shape=(self.nF, self.S))
        Ms.sum_duplicates()
        return Ms

    def risk_event(self, t0, ids, cand, wlo=None, whi=None):
        H10 = self.firm_sc_counts(ids, addy(t0, -10), t0)
        risk = np.zeros((len(ids), self.S), bool)
        risk[:, cand] = True
        risk &= (H10 == 0)
        ev = None
        if wlo is not None:
            Nw = self.firm_sc_counts(ids, wlo, whi)
            ev = risk & (Nw > 0)
        return H10, risk, ev

    # ---------- 공개 배정 라벨 ----------
    def set_pub_labels(self, t0):
        K = json.load(open(C + r'\ksic_list.json')); K2I = {k: i for i, k in enumerate(K)}
        tabs = build_pub_tables(self.kr, t0)
        dd = self.docs[['sc', 'mg', 'sg']]
        kd = self.kr[['sc', 'mg', 'sg']]
        ag = self.agg5[['sc', 'mg', 'sg']]
        self.kr_lab = {'prov': self.kr.k.values.astype(np.int32)}
        self.ag_lab = {'prov': self.agg5.k.values.astype(np.int32)}
        for a, rule in [('free', 'H-sg-3'), ('hsc', 'H-sc-1')]:
            self.labels[a] = apply_pub(dd, tabs, rule).map(K2I).fillna(-1).astype(np.int32).values
            self.kr_lab[a] = apply_pub(kd, tabs, rule).map(K2I).fillna(-1).astype(np.int32).values
            self.ag_lab[a] = apply_pub(ag, tabs, rule).map(K2I).fillna(-1).astype(np.int32).values

    # ---------- 추세 ----------
    @staticmethod
    def kg_from(years, labs, w, Y):
        ok = labs >= 0
        y = years[ok]; l = labs[ok]; w = w[ok]
        r = (y >= Y - 3) & (y <= Y - 1); o = (y >= Y - 6) & (y <= Y - 4)
        n1 = np.bincount(l[r], weights=w[r], minlength=NK); n0 = np.bincount(l[o], weights=w[o], minlength=NK)
        return (np.log((n1 + 1) / (n1.sum() + NK)) - np.log((n0 + 1) / (n0.sum() + NK))).astype(np.float32)

    def trends(self, Y, lab_c, sysname):
        """kg_c, kg_kr, kg_5. lab_c = 이 나라 문서 라벨 배열(귀무에서 교체 가능)."""
        okc = self.d_pub < ymd(Y)
        kg_c = self.kg_from(self.d_year[okc], lab_c[okc], np.ones(okc.sum()), Y)
        krp = self.kr.pub.values < ymd(Y)
        kl = self.kr_lab[sysname]
        kg_kr = self.kg_from((self.kr.pub.values[krp] // 10000).astype(np.int32), kl[krp], np.ones(krp.sum()), Y)
        ag = self.agg5; al = self.ag_lab[sysname]
        oth = (ag.cc.values != self.cc) & (ag.y.values < Y)
        yy = np.concatenate([ag.y.values[oth].astype(np.int32), self.d_year[okc]])
        ll = np.concatenate([al[oth], lab_c[okc]])
        ww = np.concatenate([ag.n.values[oth].astype(np.float64), np.ones(okc.sum())])
        kg_5 = self.kg_from(yy, ll, ww, Y)
        return kg_c, kg_kr, kg_5

    # ---------- 맥락 피처 ----------
    def context(self, t0, lab, ff, ss, Hd_all, Hn_all, kgs, M_override=None, Q_override=None):
        """lab: 이 나라 문서 라벨 배열. ff: 전체 기업 인덱스(행), ss: 서브클래스(행)."""
        lo5 = addy(t0, -5)
        wm = self.mwin(lo5, t0)
        la = lab[self.m_did[wm]]; ok = la >= 0
        if M_override is None:
            M = np.zeros((self.nF, NK), np.float32)
            np.add.at(M, (self.m_aid[wm][ok], la[ok]), 1)
            M = rownorm_dense(M)
        else:
            M = M_override
        if Q_override is None:
            wd = self.dwin(lo5, t0) & (self.d_sc >= 0) & (lab >= 0)
            Q = np.zeros((self.S, NK), np.float32)
            np.add.at(Q, (self.d_sc[wd], lab[wd]), 1)
            Q = rownorm_dense(Q)
        else:
            Q = Q_override
        F = {}
        Mf = M[ff]; Qs = Q[ss]
        dot = (Mf * Qs).sum(1)
        F['dot'] = dot
        F['cos'] = dot / np.maximum(np.linalg.norm(Mf, axis=1) * np.linalg.norm(Qs, axis=1), 1e-12)
        mden = M.sum(0); mden[mden == 0] = 1
        A = np.asarray((Hd_all.T @ M)).T / mden[:, None]    # K × S
        En = np.asarray((Hn_all.T @ M)).T / mden[:, None]
        F['adopt'] = (Mf * A[:, ss].T).sum(1)
        F['entry'] = (Mf * En[:, ss].T).sum(1)
        hasm = Mf.sum(1) > 0; main = Mf.argmax(1)
        F['qmain'] = np.where(hasm, Q[ss, main], 0)
        for nm, kg in zip(['c', 'kr', '5'], kgs):
            F['mom_' + nm] = np.where(hasm, kg[main], 0)
            F['sind_' + nm] = Qs @ kg
        return {k: v.astype(np.float32) for k, v in F.items()}, M, Q

    # ---------- 원점 하나 ----------
    def count(self, Y):
        t0 = ymd(Y)
        ids = self.active(t0)
        cand, scn = self.cand(t0)
        H10, risk, ev = self.risk_event(t0, ids, cand)
        return dict(cc=self.cc, Y=Y, n_active=int(len(ids)), n_cand=int(len(cand)), n_risk=int(risk.sum()))

    def gen_origin(self, Y, mode='dev', store_neg=300000, chunk_firms=None):
        """mode dev: 결과·표집, lock: 결과 없이 모든 위험 행(청크 생성기), count: 위험 크기만."""
        t0 = ymd(Y)
        ids = self.active(t0)
        cand, scn = self.cand(t0)
        if mode == 'dev':
            wlo, whi = t0, ymd(Y, 6, 30) + 1
            assert whi <= CUT, '잠금 창은 만들지 않는다'
            H10, risk, ev = self.risk_event(t0, ids, cand, wlo, whi)
        else:
            H10, risk, ev = self.risk_event(t0, ids, cand)
        info = dict(cc=self.cc, Y=Y, n_active=int(len(ids)), n_cand=int(len(cand)), n_risk=int(risk.sum()))
        if mode == 'dev':
            npos = int(ev.sum()); nneg = info['n_risk'] - npos
            r = min(1.0, store_neg / max(nneg, 1))
            rng = np.random.default_rng(SEED + Y + (0 if self.cc == 'CN' else 100))
            U = rng.random(risk.shape)
            keep = ev | (risk & ~ev & (U < r))
            fi, si_ = np.where(keep)
            yv = ev[fi, si_]
            wv = np.where(yv, 1.0, 1.0 / r)
            info.update(n_pos=npos, n_pos_firms=int(ev.any(1).sum()), n_neg=nneg, r=r, n_rows=int(len(fi)))
            # P3(b): 양성 문서 공개 < 출원
            Nw_mask = self.mwin(wlo, whi) & (self.m_sc >= 0)
            pos_idx = pd.Index(ids)
            rr = pos_idx.get_indexer(self.m_aid[Nw_mask]); okr = rr >= 0
            dd = self.m_did[Nw_mask][okr]; scc = self.m_sc[Nw_mask][okr]; rr = rr[okr]
            isp = ev[rr, scc]
            pd_ = np.unique(dd[isp])
            adv = self.d_ad[pd_]
            info.update(pos_docs=int(len(pd_)), pos_docs_pub_lt_ad=int(np.nansum(self.d_pub[pd_] < adv)),
                        pos_docs_ad_missing=int(np.isnan(adv.astype(float)).sum()))
            yield info, self.features(Y, t0, ids, cand, scn, H10, fi, si_, yv, wv)
        else:
            fi_all, si_all = np.where(risk)
            n = len(fi_all); step = chunk_firms or 2500
            firms_u = np.unique(fi_all)
            for c0 in range(0, len(firms_u), step):
                fs = firms_u[c0:c0 + step]
                sel = (fi_all >= fs[0]) & (fi_all <= fs[-1])
                yield info, self.features(Y, t0, ids, cand, scn, H10, fi_all[sel], si_all[sel], np.zeros(sel.sum(), bool), np.ones(sel.sum()))

    def features(self, Y, t0, ids, cand, scn, H10, fi, si_, yv, wv):
        cache = getattr(self, '_fc', None)
        if cache is None or cache['Y'] != Y:
            self._fc = self.origin_mats(Y, t0, ids, cand, scn, H10)
        Z = self._fc
        ff = ids[fi]
        D = {'cc': np.repeat(self.cc, len(fi)), 'aid': ff.astype(np.int64), 'key': self.keys.values[ff], 's': si_.astype(np.int32),
             'Y': np.repeat(Y, len(fi)).astype(np.int16), 'y': yv.astype(np.int8), 'w': wv.astype(np.float32)}
        for k in ['f_log3', 'f_log1', 'f_gr', 'f_nsc', 'f_age', 'f_missc']:
            D[k] = Z[k][fi]
        D['is_cn'] = np.full(len(fi), np.float32(self.cc == 'CN'))
        for k in ['s_n5', 's_grow', 's_nfirm', 's_newrate', 'pi_rsc']:
            D[k] = Z[k][si_]
        D['pi_rfirm'] = Z['pi_rfirm'][fi]
        Csec = Z['Csec']
        D['p_sec'] = np.log1p(np.asarray(Csec[fi, si_]).ravel()).astype(np.float32)
        Wd = Z['W']; R = Z['R']
        D['p_relmean'] = (Wd[fi] * R[:, si_].T).sum(1)
        relmax = np.zeros(len(fi), np.float32)
        Hb = H10 > 0
        order = np.argsort(fi, kind='stable'); fis = fi[order]
        bnd = np.flatnonzero(np.diff(fis)) + 1
        for grp in np.split(order, bnd):
            f = fi[grp[0]]
            h = np.where(Hb[f])[0]
            if len(h):
                relmax[grp] = R[np.ix_(h, si_[grp])].max(0)
        D['p_relmax'] = relmax
        D['p_cls'] = Z['Wc'][fi, self.cls_id[si_]]
        D['p_sect'] = Z['Ws'][fi, self.sect_id[si_]]
        D['p_cf'] = (Wd[fi] * Z['Aipc'][:, si_].T).sum(1)
        fa = ff
        for sysn in SYS:
            F = self.ctx_for(sysn, Y, t0, fa, si_)
            for k, v in F.items():
                D[f'{sysn}_{k}'] = v
        df = pd.DataFrame(D)
        for c in df.columns:
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        return df

    def ctx_for(self, sysn, Y, t0, fa, si_):
        Z = self._fc
        if sysn == 'scagg':
            F, _, _ = self.context(t0, self.labels['prov'], fa, si_, Z['Hd_all'], Z['Hn_all'], Z['kg']['prov'],
                                   M_override=Z['Magg'], Q_override=Z['Q']['prov'])
        else:
            F, _, _ = self.context(t0, self.labels[sysn], fa, si_, Z['Hd_all'], Z['Hn_all'], Z['kg'][sysn],
                                   M_override=Z['M'][sysn], Q_override=Z['Q'][sysn])
        return F

    def origin_mats(self, Y, t0, ids, cand, scn, H10):
        Z = {'Y': Y}
        nA = len(ids)
        # 기업 활동
        def nd(lo, hi):
            w = self.mwin(lo, hi)
            return np.bincount(self.m_aid[w], minlength=self.nF)[ids].astype(np.float32)
        n3 = nd(addy(t0, -3), t0); n1 = nd(addy(t0, -1), t0)
        Z['f_log3'] = np.log1p(n3); Z['f_log1'] = np.log1p(n1)
        Z['f_gr'] = n1 / np.maximum((n3 - n1) / 2, 1)
        Z['f_nsc'] = (H10 > 0).sum(1).astype(np.float32)
        w = self.m_pub < t0
        fmin = pd.Series(self.m_pub[w]).groupby(self.m_aid[w]).min()
        Z['f_age'] = ((t0 - fmin.reindex(ids).values) / 10000.0).astype(np.float32)
        w3 = self.mwin(addy(t0, -3), t0)
        miss = np.bincount(self.m_aid[w3 & (self.m_sc < 0)], minlength=self.nF)[ids]
        Z['f_missc'] = (miss / np.maximum(n3, 1)).astype(np.float32)
        # 서브클래스
        def sn(lo, hi):
            wd = self.dwin(lo, hi) & (self.d_sc >= 0)
            return np.bincount(self.d_sc[wd], minlength=self.S).astype(np.float32)
        Z['s_n5'] = np.log1p(scn.astype(np.float32))
        Z['s_grow'] = np.log1p(sn(addy(t0, -2), t0)) - np.log1p(sn(addy(t0, -4), addy(t0, -2)))
        hold = H10 > 0
        nh = hold.sum(0)
        Z['s_nfirm'] = np.log1p(nh).astype(np.float32)
        # 첫 사용 연도(10년 창) 최근 3년 비율
        wl = self.mwin(addy(t0, -10), t0) & (self.m_sc >= 0)
        pos = pd.Index(ids)
        r = pos.get_indexer(self.m_aid[wl]); ok = r >= 0
        fu = pd.DataFrame({'r': r[ok], 's': self.m_sc[wl][ok], 'p': self.m_pub[wl][ok]}).groupby(['r', 's']).p.min().reset_index()
        rec = fu[fu.p >= addy(t0, -3)]
        nrec = np.bincount(rec.s.values, minlength=self.S)
        Z['s_newrate'] = (nrec / np.maximum(nh, 1)).astype(np.float32)
        # 보조 IPC 사용(10년)
        dl = np.zeros(len(self.d_pub), bool); dl[self.m_did[wl]] = True
        xm = dl[self.x_did]
        xd = self.x_did[xm]; xs = self.x_sc[xm]
        nonmain = xs != self.d_sc[xd]
        xd = xd[nonmain]; xs = xs[nonmain]
        # 문서→활성 기업 펼치기
        mw = self.mwin(addy(t0, -10), t0)
        mr = pos.get_indexer(self.m_aid[mw]); mok = mr >= 0
        A2 = pd.DataFrame({'did': self.m_did[mw][mok], 'r': mr[mok]})
        X2 = pd.DataFrame({'did': xd, 's': xs}).merge(A2, on='did')
        Csec = sp.csr_matrix((np.ones(len(X2), np.float32), (X2.r.values, X2.s.values)), shape=(nA, self.S))
        Csec.sum_duplicates(); Z['Csec'] = Csec
        # 동시출현 R (5년, ipc_all)
        d5 = self.dwin(addy(t0, -5), t0)
        x5 = d5[self.x_did]
        pi, _ = pd.factorize(self.x_did[x5])
        X = sp.csr_matrix((np.ones(x5.sum(), np.float32), (pi, self.x_sc[x5])), shape=(pi.max() + 1, self.S))
        X.data[:] = 1
        CO = (X.T @ X).toarray(); dg = np.sqrt(np.diag(CO)); dg[dg == 0] = 1
        R = (CO / dg[:, None] / dg[None, :]).astype(np.float32); np.fill_diagonal(R, 0)
        Z['R'] = R
        Wd = rownorm_dense(H10.copy())
        Z['W'] = Wd
        ncl = self.cls_id.max() + 1; nse = self.sect_id.max() + 1
        Wc = np.zeros((nA, ncl), np.float32); Ws = np.zeros((nA, nse), np.float32)
        for j in range(self.S):
            Wc[:, self.cls_id[j]] += Wd[:, j]; Ws[:, self.sect_id[j]] += Wd[:, j]
        Z['Wc'] = Wc; Z['Ws'] = Ws
        # 협업필터(전체 기업, 10년)
        Hc = self.all_firm_sc(addy(t0, -10), t0)
        rs = np.asarray(Hc.sum(1)).ravel(); rs[rs == 0] = 1
        Wall = sp.diags(1 / rs) @ Hc
        Hb_all = Hc.copy(); Hb_all.data[:] = 1
        den = np.asarray(Wall.sum(0)).ravel(); den[den == 0] = 1
        Aipc = (Wall.T @ Hb_all).toarray() / den[:, None]; np.fill_diagonal(Aipc, 0)
        Z['Aipc'] = Aipc.astype(np.float32)
        Z['Hd_all'] = Hb_all.tocsr()
        # 최근 3년 첫 사용(전체 기업)
        wl_all = self.mwin(addy(t0, -10), t0) & (self.m_sc >= 0)
        fa = pd.DataFrame({'a': self.m_aid[wl_all], 's': self.m_sc[wl_all], 'p': self.m_pub[wl_all]}).groupby(['a', 's']).p.min().reset_index()
        fa = fa[fa.p >= addy(t0, -3)]
        Z['Hn_all'] = sp.csr_matrix((np.ones(len(fa), np.float32), (fa.a.values, fa.s.values)), shape=(self.nF, self.S))
        # 지속 지수(직전 12개월)
        tp = addy(t0, -1)
        idp = self.active(tp)
        candp, _ = self.cand(tp)
        _, riskp, evp = self.risk_event(tp, idp, candp, tp, t0)
        rf = np.where(riskp.sum(1) > 0, evp.sum(1) / np.maximum(riskp.sum(1), 1), np.nan)
        Z['pi_rfirm'] = pd.Series(rf, index=idp).reindex(ids).values.astype(np.float32)
        rsc = np.where(riskp.sum(0) > 0, evp.sum(0) / np.maximum(riskp.sum(0), 1), np.nan)
        Z['pi_rsc'] = rsc.astype(np.float32)
        # 맥락 행렬(체계별)
        self.set_pub_labels(t0)
        Z['M'] = {}; Z['Q'] = {}; Z['kg'] = {}; Z['cover'] = {}
        for sysn in ['prov', 'free', 'hsc']:
            lab = self.labels[sysn]
            Z['kg'][sysn] = self.trends(Y, lab, sysn)
            _, M, Q = self.context(t0, lab, np.array([0]), np.array([0]), Z['Hd_all'], Z['Hn_all'], Z['kg'][sysn])
            Z['M'][sysn] = M; Z['Q'][sysn] = Q
            d5l = self.dwin(addy(t0, -5), t0)
            Z['cover'][sysn] = float((lab[d5l] >= 0).mean())
        # 서브클래스 집계 포트폴리오(5년 메인 서브클래스 몫 × Q_prov)
        H5 = self.all_firm_sc(addy(t0, -5), t0)
        rs5 = np.asarray(H5.sum(1)).ravel(); rs5[rs5 == 0] = 1
        Z['Magg'] = np.asarray((sp.diags(1 / rs5) @ H5) @ Z['Q']['prov']).astype(np.float32)
        Z['kg']['scagg'] = Z['kg']['prov']
        return Z
