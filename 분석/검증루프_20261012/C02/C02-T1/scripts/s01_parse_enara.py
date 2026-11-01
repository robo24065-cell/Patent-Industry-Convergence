# e-나라지표 1330(정부 R&D 예산 / 국가연구개발사업 조사분석) 표 파싱 -> CSV (긴 형식)
import re, sys, html, csv, os
D = sys.argv[1]; OUT = sys.argv[2]
rows_out = []
for fn in sorted(os.listdir(D)):
    m = re.match(r'enara_1330_(\d+)_', fn)
    if not m: continue
    t = open(os.path.join(D, fn), 'rb').read().decode('utf-8', errors='replace')
    trs = re.findall(r'<tr[^>]*>(.*?)</tr>', t, flags=re.S)
    tab = []
    for r in trs:
        cells = [html.unescape(re.sub(r'<[^>]+>', '', c)).strip() for c in re.findall(r'<t[hd][^>]*>(.*?)</t[hd]>', r, flags=re.S)]
        if cells: tab.append(cells)
    years = [c for c in tab[0] if re.fullmatch(r'\d{4}', c)]
    has_ratio = any('비율' in c for c in tab[1]) if len(tab) > 1 else False
    for cells in tab[1:]:
        lab = cells[0]
        vals = cells[1:]
        if not lab or all(v == '' for v in vals): continue
        if has_ratio:
            for i, y in enumerate(years):
                a = vals[2*i] if 2*i < len(vals) else ''
                b = vals[2*i+1] if 2*i+1 < len(vals) else ''
                rows_out.append([m.group(1), lab, y, 'amount', a.replace(',', '')])
                rows_out.append([m.group(1), lab, y, 'share', b.replace(',', '')])
        else:
            for i, y in enumerate(years):
                a = vals[i] if i < len(vals) else ''
                rows_out.append([m.group(1), lab, y, 'value', a.replace(',', '')])
with open(OUT, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.writer(f); w.writerow(['stts_cd', 'item', 'year', 'kind', 'value']); w.writerows(rows_out)
print(len(rows_out))
