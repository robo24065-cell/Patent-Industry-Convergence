# 감사 프로브 1: 스냅샷성 필드 점검 (홀드아웃 결과 조회 없음)
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, t4lib as T
D = T.load(); pt, ow, pay, rt = D['pt'], D['ow'], D['pay'], D['rt']
print('dspst top:'); print(pt.dspst.value_counts().head(25).to_string())
print('owner od NaN by seq==0?', ow.od.isna().sum(), ow[ow.od.isna()].seq.value_counts().head(5).to_dict())
print('owner rows seq0 od vs rd: seq0 od NaN share', ow[ow.seq==0].od.isna().mean())
print('ownerRgstCsName top'); print(ow.ownerRgstCsName.value_counts().head(10).to_string())
# 존속기간 연장 이벤트
print('right names containing 연장:', rt[rt.rgstCsName.str.contains('연장',na=False)].rgstCsName.value_counts().to_dict())
# init seq0 pay after rd?
p0 = pay[pay.seq==0]
lag = (p0.pd.values - pt.rd.values[p0.pid.values]).astype('timedelta64[D]').astype(float)
print('seq0 pay - rd days quantiles', np.nanpercentile(lag,[1,50,99,99.9]), 'share>30d', np.mean(lag>30))
# 데이터 마감 확인(비결과 사건만)
nonout = rt[~rt.rgstCsName.isin(['소멸등록'])]
print('max cd of non-lapse right events', nonout.cd.max(), 'max pay pd', pay.pd.max(), 'max rd', pt.rd.max())
