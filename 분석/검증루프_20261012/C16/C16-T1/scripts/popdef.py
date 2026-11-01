# -*- coding: utf-8 -*-
"""가시 기준 모집단 정의(PREREG §1·§2). 트랙 코드 import 없음. pop(vcol) → dict
 Foff/Fvis: 법인 집합, held_off/held_vis: (법인,k) 범위안 등록 보유(출원≤2019; 가시 여부), prior_vis/prior_all: 이름매칭 선행출원 (법인,k)
 Call: 가시 법인 중 공식 보유·가시 미보유 쌍(가시 선행 포함), C: 그중 가시 선행 없는 쌍(채점 대상)"""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from c16common import *


def counts(ip, op, vcol):
    i = ip if vcol is None else ip[ip[vcol] <= CAPV]; o = op if vcol is None else op[op[vcol] <= CAPV]
    return pd.concat([i.drop_duplicates(['applicant_cd', 'rgst_no']).applicant_cd, o.drop_duplicates(['applicant_cd', 'rgst_no']).applicant_cd]).value_counts()


def pop(vcol='v_pub', ins=None, oos=None, a=None):
    ins = load_ins() if ins is None else ins; oos = load_oos() if oos is None else oos; a = load_app() if a is None else a
    K, ki = kidx()
    ip = ins[ins.ay <= CUT]; op = oos[oos.ay <= CUT]
    coff = counts(ip, op, None); Foff = set(coff[coff >= 3].index)
    cv = counts(ip, op, vcol); Fvis = set(cv[cv >= 3].index)
    ipo = ip[ip.applicant_cd.isin(Foff)].assign(k=lambda d: d.ksic.map(ki))
    held_off = set(zip(ipo.applicant_cd, ipo.k)); ipv = ipo[ipo[vcol] <= CAPV]; held_vis = set(zip(ipv.applicant_cd, ipv.k))
    apr = a[a.ay <= CUT]; prior_all = set(zip(apr.applicant_cd, apr.k)); av = apr[apr.v <= CAPV]; prior_vis = set(zip(av.applicant_cd, av.k))
    Call = sorted((f, k) for (f, k) in held_off - held_vis if f in Fvis)
    C = [(f, k) for (f, k) in Call if (f, k) not in prior_vis]
    # 가시 신규진입자: 가시 범위안 등록특허 0건 법인
    vis_any = set(ipv.applicant_cd)
    return dict(Foff=Foff, Fvis=Fvis, held_off=held_off, held_vis=held_vis, prior_all=prior_all, prior_vis=prior_vis, Call=Call, C=C, vis_any_firms=vis_any, cnt_off=coff, cnt_vis=cv)
