# -*- coding: utf-8 -*-
# C19-T4 s4: 원천 직접 대조(registry_full.sqlite, mode=ro). E-C02-T3-01(95·515행 권리이전 88,323·13.5%),
# E-C15T4-07/07b(79·503행 656,360 = 615,042 + 41,318). 결과는 cache\s4_registry.json.
import sqlite3, os, json, time
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(D))))  # 공모전 루트
db = os.path.join(ROOT, '수집파이프라인', 'registry_full.sqlite')
uri = 'file:' + db.replace('\\', '/') + '?mode=ro'
t = time.time()
con = sqlite3.connect(uri, uri=True)
cur = con.cursor()
tabs = [r[0] for r in cur.execute("select name from sqlite_master where type='table'")]
cols = {tb: [r[1] for r in cur.execute(f'pragma table_info("{tb}")')] for tb in tabs}
out = {'tables': tabs, 'cols': cols}
if 'patent' in tabs:
    pc = cols['patent']
    key = 'rgst_no' if 'rgst_no' in pc else pc[0]
    out['patent_n'] = cur.execute('select count(*) from patent').fetchone()[0]
    out['patent_distinct_key'] = cur.execute(f'select count(distinct "{key}") from patent').fetchone()[0]
    out['prefix'] = dict(cur.execute(f'select substr("{key}",1,2), count(*) from patent group by 1').fetchall())
if 'owner' in tabs:
    oc = cols['owner']
    key = 'rgst_no' if 'rgst_no' in oc else oc[0]
    out['owner_rows'] = cur.execute('select count(*) from owner').fetchone()[0]
    out['owner_patents'] = cur.execute(f'select count(distinct "{key}") from owner').fetchone()[0]
    if 'ownerRgstCsName' in oc:
        out['transfer_reason_patents'] = cur.execute(
            f"select count(distinct \"{key}\") from owner where ownerRgstCsName is not null and trim(ownerRgstCsName)<>''").fetchone()[0]
        out['reason_values_top'] = cur.execute(
            "select ownerRgstCsName, count(*) from owner where ownerRgstCsName is not null and trim(ownerRgstCsName)<>'' group by 1 order by 2 desc limit 15").fetchall()
    out['multi_owner_patents'] = cur.execute(
        f'select count(*) from (select "{key}" from owner group by 1 having count(*)>=2)').fetchone()[0]
con.close()
if 'patent_n' in out and 'transfer_reason_patents' in out:
    out['transfer_share_of_patent'] = out['transfer_reason_patents'] / out['patent_n']
out['sec'] = round(time.time() - t, 1)
json.dump(out, open(os.path.join(D, 'cache', 's4_registry.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1, default=str)
