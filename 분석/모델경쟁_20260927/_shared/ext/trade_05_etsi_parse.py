# -*- coding: utf-8 -*-
"""ETSI IPR 데이터베이스 공개 추출본(ISLD-export, docbox.etsi.org/IPR/Open, 로그인 불필요) → 특허 단위 요약.
행 = (선언 × 특허 × 표준). 특허(국가, 출원번호, 공개번호)별 최초 선언일·선언기업 수·표준 수·세대 플래그로 축약."""
import zipfile, io, pandas as pd
EXT = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\모델경쟁_20260927\_shared\ext'
cols = ['IPRD_ID','IPRD_SIGNATURE_DATE','Reflected_Date','COMP_LEGAL_NAME','Standard','Patent_Type','PATT_APPLICATION_NUMBER','PUBL_NUMBER',
        'PBPA_PRIORITY_NUMBERS','Country_Of_Registration','PBPA_APP_DATE','DIPG_PATF_ID','Original_Application_Number',
        'Original_Publication_Number','Normalized_Patent','2G','3G','4G','5G','Ess_To_Standard']
z = zipfile.ZipFile(EXT + r'\trade_etsi_ISLD-export.zip')
parts = []; n = 0
for ch in pd.read_csv(z.open('ISLD-export.csv'), sep=';', usecols=cols, dtype=str, chunksize=400_000, encoding='utf-8', encoding_errors='replace'):
    n += len(ch)
    ch['cc'] = ch.Country_Of_Registration.str[:2]
    ch['sig'] = pd.to_datetime(ch.IPRD_SIGNATURE_DATE.str[:10], errors='coerce')
    ch['refl'] = pd.to_datetime(ch.Reflected_Date.str[:10], errors='coerce')
    for g in ['2G','3G','4G','5G']: ch[g] = (ch[g].fillna('') != '').astype('int8')
    g = ch.groupby(['cc','PATT_APPLICATION_NUMBER','PUBL_NUMBER'], dropna=False).agg(
        first_decl=('sig','min'), last_decl=('sig','max'), first_refl=('refl','min'), n_rows=('sig','size'), company=('COMP_LEGAL_NAME','first'),
        n_std=('Standard','nunique'), family_id=('DIPG_PATF_ID','first'), app_date=('PBPA_APP_DATE','first'),
        prio=('PBPA_PRIORITY_NUMBERS','first'), orig_app=('Original_Application_Number','first'), orig_pub=('Original_Publication_Number','first'),
        norm=('Normalized_Patent','first'), g2=('2G','max'), g3=('3G','max'), g4=('4G','max'), g5=('5G','max')).reset_index()
    parts.append(g); print('rows', n, flush=True)
d = pd.concat(parts)
d = d.groupby(['cc','PATT_APPLICATION_NUMBER','PUBL_NUMBER'], dropna=False).agg(
    first_decl=('first_decl','min'), last_decl=('last_decl','max'), first_refl=('first_refl','min'), n_rows=('n_rows','sum'), company=('company','first'),
    n_std=('n_std','sum'), family_id=('family_id','first'), app_date=('app_date','first'), prio=('prio','first'),
    orig_app=('orig_app','first'), orig_pub=('orig_pub','first'), norm=('norm','first'),
    g2=('g2','max'), g3=('g3','max'), g4=('g4','max'), g5=('g5','max')).reset_index()
d.to_parquet(EXT + r'\trade_etsi_sep_patents.parquet', index=False)
print('총 행', n, '고유 특허', len(d)); print(d.cc.value_counts().head(10))
print(pd.crosstab(d.first_decl.dt.year, d.first_refl.dt.year.clip(upper=2012)).to_string())
