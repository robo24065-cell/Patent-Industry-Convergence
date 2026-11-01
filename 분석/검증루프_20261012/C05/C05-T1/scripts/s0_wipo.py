# C05-T1 s0: WIPO IPC 개정 대응표·표제 목록 파싱과 검증(D-a, D-b, D-d). python -I 로 실행, 인자 = C05-T1 폴더.
import sys, os, zipfile, hashlib, re, json, glob
import xml.etree.ElementTree as ET
import pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
W = sys.argv[1]; D = os.path.join(W, 'download'); O = os.path.join(W, 'wipo'); os.makedirs(O, exist_ok=True)
VERS = ['20210101', '20220101', '20230101', '20240101', '20250101', '20260101']
NS = '{http://www.wipo.int/classifications/ipc/masterfiles}'
rep = {}
# --- md5 무결성: md5 목록의 xml/zip 값과 대조
md5 = {}
for v in VERS:
    lines = open(os.path.join(D, f'{v}md5.txt'), encoding='utf-8', errors='replace').read().splitlines()
    md5[v] = dict(head=lines[0] if lines else '', tab={l.split()[1]: l.split()[0] for l in lines[1:] if len(l.split()) == 2})
chk = []
for v in VERS:
    z = zipfile.ZipFile(os.path.join(D, f'ipc_concordancelist_{v}.zip')); b = z.read(f'ipc_concordancelist_{v}.xml')
    h = hashlib.md5(b).hexdigest(); exp = md5[v]['tab'].get(f'ipc_concordancelist_{v}.xml')
    hz = hashlib.md5(open(os.path.join(D, f'EN_ipc_title_list_{v}.zip'), 'rb').read()).hexdigest(); expz = md5[v]['tab'].get(f'EN_ipc_title_list_{v}.zip')
    chk.append(dict(version=v, md5_list_time=md5[v]['head'], conc_xml_md5_ok=(h == exp), conc_md5_in_list=exp is not None,
                    title_zip_md5_ok=(hz == expz), title_md5_in_list=expz is not None))
rep['md5'] = chk
# --- 대응표
rows = []; attrs = {}
for v in VERS:
    b = zipfile.ZipFile(os.path.join(D, f'ipc_concordancelist_{v}.zip')).read(f'ipc_concordancelist_{v}.xml')
    root = ET.fromstring(b); attrs[v] = dict(root.attrib)
    for c in root.iter(NS + 'concordance'):
        tos = list(c.iter(NS + 'concordance-to'))
        if not tos:
            rows.append(dict(version=v, from_sym=c.get('from-symbol'), from_mod=c.get('modification'), project=c.get('revision-project'),
                             to_sym=None, to_mod=None, default=None))
        for t in tos:
            rows.append(dict(version=v, from_sym=c.get('from-symbol'), from_mod=c.get('modification'), project=c.get('revision-project'),
                             to_sym=t.get('to-symbol'), to_mod=t.get('modification'), default=t.get('default-reclassification')))
cc = pd.DataFrame(rows)
rep['conc_attrs'] = {v: {k.split('}')[-1]: x for k, x in a.items() if 'schemaLocation' not in k} for v, a in attrs.items()}
SYM = re.compile(r'^[A-H]\d\d[A-Z]\d{4}\d{6}$')
def mg_of(s):
    if not isinstance(s, str) or not SYM.match(s): return None
    return s[:4] + ' ' + str(int(s[4:8]))
cc['from_sc'] = cc.from_sym.str[:4]; cc['to_sc'] = cc.to_sym.str[:4]
cc['from_mg'] = cc.from_sym.map(mg_of); cc['to_mg'] = cc.to_sym.map(mg_of)
qa = {}
for v, g in cc.groupby('version'):
    qa[v] = dict(entries=int(len(g)), from_syms=int(g.from_sym.nunique()), to_missing=int(g.to_sym.isna().sum()),
                 from_missing=int(g.from_sym.isna().sum()), bad_from_fmt=int((~g.from_sym.fillna('').str.match(SYM)).sum()),
                 bad_to_fmt=int((~g.to_sym.fillna('').str.match(SYM) & g.to_sym.notna()).sum()),
                 dup_pairs=int(g.duplicated(['from_sym', 'to_sym']).sum()), default_missing=int(g.default.isna().sum()),
                 from_eq_to=int((g.from_sym == g.to_sym).sum()), cross_subclass_defaultY=int(((g.from_sc != g.to_sc) & (g.default == 'Y')).sum()),
                 cross_mg_defaultY=int(((g.from_mg != g.to_mg) & (g.default == 'Y')).sum()),
                 from_with_no_defaultY=int((g.groupby('from_sym').default.apply(lambda s: (s == 'Y').sum()) == 0).sum()),
                 from_attr=attrs[v].get('from-version'), to_attr=attrs[v].get('to-version'))
rep['conc_qa'] = qa
cc.to_csv(os.path.join(O, 'wipo_concordance_all.csv'), index=False, encoding='utf-8-sig')
# --- 표제 목록: 버전별 하위클래스·메인그룹 집합
subs, mgs = {}, {}
for v in VERS:
    z = zipfile.ZipFile(os.path.join(D, f'EN_ipc_title_list_{v}.zip')); S, M = {}, set()
    for n in z.namelist():
        if not n.endswith('.txt'): continue
        for line in z.read(n).decode('utf-8', errors='replace').splitlines():
            code = line.split('\t')[0].strip()
            if re.match(r'^[A-H]\d\d[A-Z]$', code): S[code] = line.split('\t')[1].strip() if '\t' in line else ''
            mm = re.match(r'^([A-H]\d\d[A-Z])(\d{4})(\d{6})$', code)
            if mm and mm.group(3) == '000000': M.add(mm.group(1) + ' ' + str(int(mm.group(2))))
    subs[v] = S; mgs[v] = M
rep['n_subclasses'] = {v: len(s) for v, s in subs.items()}; rep['n_maingroups'] = {v: len(m) for v, m in mgs.items()}
new = []
for a, b in zip(VERS[:-1], VERS[1:]):
    for s in sorted(set(subs[b]) - set(subs[a])): new.append(dict(version=b, subclass=s, kind='new', title=subs[b][s][:120]))
    for s in sorted(set(subs[a]) - set(subs[b])): new.append(dict(version=b, subclass=s, kind='deleted', title=subs[a][s][:120]))
    for m in sorted(mgs[b] - mgs[a]): new.append(dict(version=b, subclass=m, kind='new_maingroup', title=''))
    for m in sorted(mgs[a] - mgs[b]): new.append(dict(version=b, subclass=m, kind='deleted_maingroup', title=''))
nw = pd.DataFrame(new); nw.to_csv(os.path.join(O, 'wipo_title_diff.csv'), index=False, encoding='utf-8-sig')
rep['new_subclasses'] = nw[nw.kind == 'new'][['version', 'subclass', 'title']].to_dict('records')
rep['deleted_subclasses'] = nw[nw.kind == 'deleted'][['version', 'subclass']].to_dict('records')
rep['new_maingroups_n'] = nw[nw.kind == 'new_maingroup'].groupby('version').size().to_dict()
# 첫 등장 버전(2021.01 이후) 확인: H10*
rep['H10_presence'] = {s: [v for v in VERS if s in subs[v]] for s in ['H10B', 'H10D', 'H10F', 'H10H', 'H10K', 'H10N', 'H01L']}
# --- 신설 하위클래스의 원천(default=Y 항목 기준, 대응표 2022–2025)
cy = cc[(cc.default == 'Y') & cc.version.isin(VERS[1:5])]
newsc = set(nw[(nw.kind == 'new') & nw.version.isin(VERS[1:5])].subclass)
src = cy[cy.to_sc.isin(newsc) & (cy.from_sc != cy.to_sc)].groupby(['version', 'to_sc', 'from_sc']).size().rename('n_entries').reset_index()
src['share_of_to'] = src.n_entries / src.groupby(['version', 'to_sc']).n_entries.transform('sum')
src.to_csv(os.path.join(O, 'wipo_newsub_sources.csv'), index=False, encoding='utf-8-sig')
rep['new_sub_sources'] = src.to_dict('records')
# 신설이 대응표에 없는 경우(표제에는 새로 있지만 대응표 to-기호에 없음)
rep['new_sub_in_conc'] = {s: bool((cc.to_sc == s).any()) for s in sorted(newsc)}
# 하위클래스 간 이동(default=Y, 2022–2025) 요약
xs = cy[cy.from_sc != cy.to_sc].groupby(['version', 'from_sc', 'to_sc']).size().rename('n').reset_index().sort_values('n', ascending=False)
xs.to_csv(os.path.join(O, 'wipo_cross_subclass_moves.csv'), index=False, encoding='utf-8-sig')
# 메인그룹 간 이동(default=Y)
xm = cy[(cy.from_mg != cy.to_mg) & cy.from_mg.notna() & cy.to_mg.notna()][['version', 'from_mg', 'to_mg', 'project']].drop_duplicates()
xm.to_csv(os.path.join(O, 'wipo_cross_mg_moves.csv'), index=False, encoding='utf-8-sig')
rep['cross_mg_moves_n'] = xm.groupby('version').size().to_dict()
rep['cross_mg_from_mgs'] = int(xm.from_mg.nunique()); rep['cross_mg_to_mgs'] = int(xm.to_mg.nunique())
json.dump(rep, open(os.path.join(O, 's0_report.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
print(json.dumps({k: rep[k] for k in ['md5', 'conc_qa', 'n_subclasses', 'new_subclasses', 'deleted_subclasses', 'H10_presence', 'new_sub_in_conc', 'new_maingroups_n', 'cross_mg_moves_n']}, ensure_ascii=False, indent=1, default=str))
print(src.to_string())
print(xs.head(40).to_string())
