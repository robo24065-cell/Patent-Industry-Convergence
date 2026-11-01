# -*- coding: utf-8 -*-
"""3라운드 A단계(재시도) 추가 외부데이터 연계율·시점정합 점검: Reliance on Science(rs_), 국세청 폐업(nt_), DART 재무(학습 가능성 점검).
출력: ext_link_A2.csv — 원점×기준일×데이터별 평가행 연계율, 연결/비연결 3년 포기율, 학습 프레임(분류형·위험모형)에서 값이 있는 비율."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r3lib, r3ext, numpy as np, pandas as pd
R = r3lib.R
EXT = r3ext.EXT
# DART: 출원인코드 → 사업자번호(국세청 조회표) → DART 재무(avail_date = 사업보고서 공시 추정일)
st = pd.read_parquet(os.path.join(EXT, 'r3nts_applicant_status.parquet'), columns=['applicantCd', 'bizno'])
fin = pd.read_parquet(os.path.join(EXT, 'firm_dart_fin_panel.parquet'), columns=['bizr_no', 'fy', 'avail_date'])
fin = fin[fin.bizr_no.notna()]
first_av = fin.groupby('bizr_no').avail_date.min()
st = st.assign(av=st.bizno.map(first_av))
a_av = st.groupby('applicantCd').av.min()
rows = []
for o, _, Ls in R.ORIGINS:
    d = R._read_frame(o, 'ev.parquet', ['pid', 'L', 'ksic', 'acd', 'y', 'next_annl'])
    d = r3ext.add(d, ['rs_sci', 'rs_lncit', 'rs_has', 'nt_closed', 'nt_cl1y'])
    av = pd.to_datetime(a_av.reindex(d.acd.astype(str).values).values)
    for L in Ls:
        m = (d.L.astype(str).str[:10] == L).values
        e = d[m]; Lt = pd.Timestamp(L)
        for nm, lk, pos in [('RoS: US 패밀리 등록(발행<L) 보유', e.rs_has == 1, e.rs_has == 1),
                            ('RoS: 그 US 특허가 논문 인용', e.rs_sci == 1, e.rs_sci == 1),
                            ('국세청: 사업자번호 연결(L>=2016만 값)', e.nt_closed.notna(), e.nt_closed == 1),
                            ('국세청: 직전 1년 폐업', e.nt_closed.notna(), e.nt_cl1y == 1),
                            ('DART 재무: 공시일<L 재무 보유', pd.Series(av[m] < Lt, index=e.index), pd.Series(av[m] < Lt, index=e.index))]:
            rows.append(dict(origin=o, L=L, data=nm, n=len(e), linked=int(lk.sum()), rate=round(float(lk.mean()), 4),
                             ksic_linked=int(e.loc[lk, 'ksic'].nunique()), pos_n=int(pos.sum()),
                             y_pos=round(float(e.y[pos].mean()), 4) if pos.any() else np.nan,
                             y_other=round(float(e.y[~pos].mean()), 4)))
    # 학습 프레임에서 값이 있는 비율(학습 가능성)
    for stem in ('frame_cls', 'frame_haz'):
        f = R._read_frame(o, stem + '.parquet', ['pid', 'L', 'acd', 'ksic', 'next_annl'])
        Lmax = str(f.L.astype(str).max())[:10]
        fav = pd.to_datetime(a_av.reindex(f.acd.astype(str).values).values)
        rows.append(dict(origin=o, L=f'학습:{stem}(마지막 기준일 {Lmax})', data='DART 재무: 공시일<L 재무 보유',
                         n=len(f), linked=int((fav < pd.to_datetime(f.L.astype(str).str[:10]).values).sum())))
        rows.append(dict(origin=o, L=f'학습:{stem}(마지막 기준일 {Lmax})', data='국세청: L>=2016 관측 가능 행',
                         n=len(f), linked=int((pd.to_datetime(f.L.astype(str).str[:10]) >= '2016-01-01').sum())))
    print(o, 'done', R.peak_mem_gb(), flush=True)
out = pd.DataFrame(rows)
out.to_csv(os.path.join(r3lib.R3, 'ext_link_A2.csv'), index=False, encoding='utf-8-sig')
print(out.to_string())
