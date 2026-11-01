# 권리자 정적 속성(이름 기반, 시점 무관): 국적군, 대기업집단 소속, 지주회사, IP관리 전문법인 → cache/owner_attr.parquet
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
GROUPS = ['삼성', '엘지', 'LG', '현대', '기아', '에스케이', 'SK', '롯데', '한화', '두산', '코오롱', '효성', '씨제이', 'CJ', '포스코', '지에스', 'GS', '한진',
          '금호', '동부', '대림', '동양', '태광', '이랜드', '노루', '아모레', '오씨아이', 'OCI', '한솔', '신세계', '영풍', '대상', '농심', '세아', '한라', '만도',
          '케이씨씨', 'KCC', '웅진', '동국', '한국타이어', '넥센', '엘에스', 'LS', '대우', '하이닉스', '매그나칩', '제일모직', '에이치디씨', 'HDC']
o['chaebol'] = o.raw.fillna('').apply(lambda s: int(any(s.replace('주식회사', '').replace('(주)', '').strip().startswith(g) for g in GROUPS)))
o['holding'] = o.raw.fillna('').str.contains('홀딩스|홀딩|지주|Holdings', case=False).astype(int)
o['ip_entity'] = o.raw.fillna('').str.contains('라이센싱|라이선싱|licensing|아이피|IP매니지먼트|인텔렉추얼|intellectual|특허관리', case=False).astype(int)
o = o.reset_index().rename(columns={'index': 'nm'})
o[['nm', 'natl_grp', 'chaebol', 'holding', 'ip_entity']].to_parquet('cache/owner_attr.parquet', index=False)
print(len(o), o[['natl_grp', 'chaebol', 'holding', 'ip_entity']].apply(lambda s: s.value_counts().to_dict()).to_dict())
