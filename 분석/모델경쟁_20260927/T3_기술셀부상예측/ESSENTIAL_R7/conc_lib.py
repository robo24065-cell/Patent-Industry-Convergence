# -*- coding: utf-8 -*-
"""R7 필수성 검증 — 공식 IPC-KSIC 연계표(2013, KSIC 9차) 배정 라이브러리 (T3·N5 공용 사본).
읽기 전용 입력:
  - 공모전데이터/외부데이터/20131002_IPC-KSIC/20131002_IPC-KSIC.xlsx  ('전체연계표': 산업코드 × IPC 서브그룹 × 공출현 건수)
  - _shared/ext/r7_cell_ksic_ksco_revision_link_20260430.csv  (국가데이터처 산업·직업분류 신구 연계표, KSIC 10차→9차 5자리)
배정 규칙(사전 고정):
  1) 레코드의 메인 IPC(없으면 ipc_all 첫 코드)를 서브그룹으로 정규화(공백 제거, 그룹 앞자리 0 제거, 서브그룹 끝 0 제거 후 최소 2자리)
  2) 연계표 서브그룹 일치 → 그 서브그룹의 최다 건수 산업코드 (L1)
     없으면 메인그룹 합산 최다 (L2), 없으면 서브클래스 합산 최다 (L3), 없으면 미배정
  3) 동률은 산업코드 사전순 첫 번째
제공 KSIC 10차 5자리 → 연계표 코드: 공식 10→9차 연계(첫 번째 대응) → 9차 코드 앞 4자리 'C'+4 가 연계표에 있으면 그것,
  아니면 'C'+앞3+'0', 아니면 'C'+앞2+'00'. 9차 291x는 2918(사무용 기계)만 C2918, 나머지 C2919.
"""
import os, re
import numpy as np, pandas as pd

BASE = r"C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전"
MC = os.path.join(BASE, r"분석\모델경쟁_20260927")
XL = os.path.join(BASE, r"공모전데이터\외부데이터\20131002_IPC-KSIC\20131002_IPC-KSIC.xlsx")
REV = os.path.join(MC, r"_shared\ext\r7_cell_ksic_ksco_revision_link_20260430.csv")
META = os.path.join(MC, r"_shared\meta").replace("\\", "/") + "/*.parquet"

KS50 = ('15220 20202 20411 20423 20491 21220 23992 24191 24219 25914 25922 25924 25932 26111 26112 26121 26129 26211 26212 26223 '
        '26224 26291 26322 26323 26422 26429 26519 26529 27199 27211 27212 27301 28111 28122 28511 28902 29131 29132 29133 29150 '
        '29162 29210 29229 29241 29280 30393 31202 31312 33301 33309').split()


def norm_sg(s):
    """IPC 문자열 → (서브클래스, 그룹, 서브그룹) 정규화 문자열 'H01L21/336'. 실패 시 None."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return None
    t = str(s).upper().replace(' ', '')
    m = re.match(r'^([A-H][0-9]{2}[A-Z])([0-9]{1,4})/([0-9]+)$', t)
    if m:
        sc, g, sub = m.group(1), int(m.group(2)), m.group(3)
    else:
        m = re.match(r'^([A-H][0-9]{2}[A-Z])([0-9]{3,6})$', t)
        if not m:
            return None
        sc, g, sub = m.group(1), int(m.group(2)[:-2]), m.group(2)[-2:]
    sub = sub.rstrip('0')
    sub = (sub + '00')[:max(2, len(sub))]
    return f'{sc}{g}/{sub}'


# duckdb SQL 매크로(위 norm_sg 와 같은 규칙) — 대용량 레코드용
SQL_MACRO = r"""
create or replace macro r7_t(s) as upper(replace(coalesce(s,''),' ',''));
create or replace macro r7_sg(s) as (case
  when regexp_matches(r7_t(s),'^[A-H][0-9]{2}[A-Z][0-9]{1,4}/[0-9]+$') then
     regexp_extract(r7_t(s),'^([A-H][0-9]{2}[A-Z])',1) || cast(cast(regexp_extract(r7_t(s),'^[A-H][0-9]{2}[A-Z]([0-9]{1,4})/',1) as int) as varchar) || '/' ||
     left(rtrim(regexp_extract(r7_t(s),'/([0-9]+)$',1),'0') || '00', greatest(2, length(rtrim(regexp_extract(r7_t(s),'/([0-9]+)$',1),'0'))))
  when regexp_matches(r7_t(s),'^[A-H][0-9]{2}[A-Z][0-9]{3,6}$') then
     left(r7_t(s),4) || cast(cast(substr(r7_t(s),5,length(r7_t(s))-6) as int) as varchar) || '/' ||
     left(rtrim(right(r7_t(s),2),'0') || '00', greatest(2, length(rtrim(right(r7_t(s),2),'0'))))
  else null end);
"""


def load_concordance():
    cc = pd.read_excel(XL, sheet_name=1, header=0)
    cc.columns = ['code', 'ipc', 'n']
    cc = cc.dropna()
    cc['code'] = cc.code.astype(str).str.strip()
    cc['sg'] = cc.ipc.map(norm_sg)
    bad = cc.sg.isna().mean()
    cc = cc.dropna(subset=['sg'])
    cc['mg'] = cc.sg.str.split('/').str[0]
    cc['sc'] = cc.sg.str[:4]
    return cc, float(bad)


def argmax_table(cc, key):
    g = cc.groupby([key, 'code']).n.sum().reset_index()
    tot = g.groupby(key).n.transform('sum')
    g['p'] = g.n / tot
    g = g.sort_values([key, 'n', 'code'], ascending=[True, False, True]).drop_duplicates(key)
    return g[[key, 'code', 'p']].rename(columns={'code': f'code_{key}', 'p': f'p_{key}'})


def ksic10_to_code(codes):
    rev = pd.read_csv(REV, encoding='cp949', dtype=str)
    rev = rev[(rev.분류종류 == 'KSIC') & (rev.분류차순정보 == '10') & (rev.분류범위 == '5')]
    out = {}
    for k in KS50:
        r = rev[rev.신분류코드 == k]
        k9 = r.구분류코드.iloc[0] if len(r) else k
        if k9.startswith('291'):
            c = 'C2918' if k9.startswith('2918') else 'C2919'
        else:
            c = next((x for x in ('C' + k9[:4], 'C' + k9[:3] + '0', 'C' + k9[:2] + '00') if x in codes), None)
        out[k] = dict(ksic9=k9, ksic9_all=';'.join(r.구분류코드) if len(r) else k, code=c)
    return out
