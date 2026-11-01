# verify_0 (C09 hyundai finding) independent dump: openpyxl read-only, no pandas
import sys, glob, os, openpyxl
base = sys.argv[1]
out = sys.argv[2]
targets = {
 'idx_2015': '002_*', 'f_250_1': '002_*', 'f_267_2': '001_*', 'f_283_2': '006_*',
}
lines = []
for src, pat in targets.items():
    fs = glob.glob(os.path.join(base, src, pat))
    for f in fs:
        wb = openpyxl.load_workbook(f, read_only=True, data_only=True)
        for ws in wb.worksheets:
            if '다출원' not in ws.title:
                continue
            lines.append('#### %s | %s | %s' % (src, os.path.basename(f), ws.title))
            for i, row in enumerate(ws.iter_rows(values_only=True)):
                if i > 40: break
                cells = ['' if v is None else str(v).replace('\n', ' ') for v in row]
                while cells and cells[-1] == '': cells.pop()
                lines.append('%02d: ' % i + ' | '.join(cells))
open(out, 'w', encoding='utf-8').write('\n'.join(lines))
print('ok', len(lines))
