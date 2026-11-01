# [2단계 감사 수정 F3] 권리자 정적 속성 재정의 → cache/owner_attr2.parquet
# 감사 발견: s25 의 대기업집단 목록에 검증기(L2014·15) 양성 상위 양도인인 '제일모직'(495건)·'노루'(노루홀딩스 283건)와
#  학습기 양도인 '매그나칩'·'하이닉스' 등이 사후적으로 추가돼 있었다(v22 가설 문구에도 '제일모직→삼성SDI, 노루홀딩스→노루페인트' 명시).
#  → 검증 결과를 보고 만든 피처(검증 누수). 또 권리자명은 2026 현재명칭이라 '홀딩스' 문자열은 L 이후 지주사 전환을 반영할 수 있다.
# 수정: 사전에 정한 표준 목록(공정위 상호출자제한 기업집단 상위권 그룹명, 매각 사례와 무관하게 고정)만 사용하고 holding 플래그는 삭제.
import sys, pandas as pd, numpy as np
sys.stdout.reconfigure(encoding='utf-8')
from common import *
app = pd.read_parquet('cache/app.parquet', columns=['applicantName', 'applicantNatl'])
own = pd.read_parquet('cache/own.parquet', columns=['ownerName', 'ownerNatl'])
a = pd.concat([app.rename(columns={'applicantName': 'raw', 'applicantNatl': 'natl'}), own.rename(columns={'ownerName': 'raw', 'ownerNatl': 'natl'})])
a['nm'] = a.raw.map(norm)
natl = a.dropna(subset=['natl']).groupby('nm').natl.agg(lambda s: s.value_counts().index[0])
raw = a.groupby('nm').raw.first()
o = pd.DataFrame({'raw': raw}); o['natl'] = natl.reindex(o.index)
NG = {'대한민국': 1, '일본': 2, '미국': 3, '독일': 4, '프랑스': 4, '영국': 4, '네덜란드': 4, '스위스': 4, '스웨덴': 4, '핀란드': 4, '이탈리아': 4,
      '벨기에': 4, '오스트리아': 4, '덴마크': 4, '중국': 5, '대만': 6}
o['natl_grp'] = o.natl.map(NG).fillna(pd.Series(np.where(o.natl.isna(), 0, 7), index=o.index)).astype(int)
# 표준 대기업집단 그룹명(상위권, 사전 고정)
GROUPS = ['삼성', '현대', '에스케이', 'SK', '엘지', 'LG', '롯데', '포스코', '지에스', 'GS', '한화', '케이티', 'KT', '한진', '두산', '씨제이', 'CJ',
          '엘에스', 'LS', '금호', '대림', '부영', '동부', '신세계', '효성', '대우', '영풍', '에쓰오일', '오씨아이', 'OCI', '코오롱', '케이씨씨', 'KCC',
          '한국타이어', '대성', '동국', '세아', '태광', '미래에셋', '한라', '이랜드', '한솔', '동양']
o['chaebol'] = o.raw.fillna('').apply(lambda s: int(any(s.replace('주식회사', '').replace('(주)', '').strip().startswith(g) for g in GROUPS)))
o['ip_entity'] = o.raw.fillna('').str.contains('라이센싱|라이선싱|licensing|아이피|IP매니지먼트|인텔렉추얼|intellectual|특허관리', case=False).astype(int)
o = o.reset_index().rename(columns={'index': 'nm'})
o[['nm', 'natl_grp', 'chaebol', 'ip_entity']].to_parquet('cache/owner_attr2.parquet', index=False)
print(len(o), o[['natl_grp', 'chaebol', 'ip_entity']].apply(lambda s: s.value_counts().to_dict()).to_dict())
