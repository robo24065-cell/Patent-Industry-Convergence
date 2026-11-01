# 반박 검증 r3: 연결 경로를 시점 정합하게 바꿨을 때 P1(개발 고유 사건) 민감도. 개발(공개<2025)만.
#  V0 = 원래(C02 docs: 등록원부 코드[2026 스냅샷 등록 여부 의존] + app_pub 미등록)
#  V1 = app_pub 코드만(등록 여부와 무관, 잠금창과 같은 경로)
#  V2 = as-of: 등록원부 코드는 등록일 <= 창 끝인 문서만, 그 밖에는 app_pub 코드
import sys
sys.path.insert(0, r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C03\C03-T4\scripts')
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from libC import build, DEV, B, C2
kp = pd.read_parquet(B + r'\공모전데이터\작업데이터\kr_patents.parquet', columns=['출원번호', '공개일자', '세세분류코드'])
kp.columns = ['appl_no', 'p', 'ksic']; kp = kp[kp.p.fillna('').str.len() == 8]; kp = kp[kp.p < '20250101'].copy(); kp['ksic'] = kp.ksic.astype(str)
assert kp.p.max() < '20250101'
docs = pd.read_parquet(C2 + r'\docs.parquet', columns=['cd', 'appl_no', 'reg'])
docs = docs[docs.cd.str[0] == '1']
ap = pd.read_parquet(B + r'\분석\모델경쟁_20260927\T1_산업대기실\cache\app_pub.parquet', columns=['applicant_cd', 'appl_no', 'reg'])
ap = ap.rename(columns={'applicant_cd': 'cd'}); ap = ap[ap.cd.fillna('').str[0] == '1']
reg = pd.read_parquet(C2 + r'\app_reg.parquet', columns=['cd', 'appl_no', 'rgstDate']).dropna(subset=['cd'])
reg = reg[reg.cd.str[0] == '1']
def run(dd, name, regdate=None):
    out = []; tot = 0
    for t0 in DEV:
        x = dd
        if regdate is not None:
            we = f'{t0+1}0630'
            x = pd.concat([regdate[regdate.rgstDate.fillna('99999999') <= we][['cd', 'appl_no']], ap[['cd', 'appl_no']]]).drop_duplicates()
        x = x[['cd', 'appl_no']].drop_duplicates().merge(kp, on='appl_no')
        st, pos, r = build(t0, x)
        out.append(st['pos']); tot += st['pos']
    print(name, 'per-origin', out, 'total', tot, flush=True)
    return tot
run(docs, 'V0 원래')
run(ap, 'V1 app_pub만')
run(None, 'V2 as-of(등록일<=창끝만 등록원부 코드)', regdate=reg)
