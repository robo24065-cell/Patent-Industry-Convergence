# C16-T3 s1: CNIPA 知识产权统计年报 HTML 표 파싱 → cache/cnipa_official.csv
# 사용: python -I s1_parse_cnipa.py <html_dir> <out_csv>
# 표: 2014–2016 a4(국내 3종 受理/申请), a7(국내 직무 발명 受理), a8(국내 직무 실용신안 受理)
#     2017–2019 a4(국내 3종 申请), a7(직무 발명 申请), a8(직무 실용신안 申请)
#     2020–2024 a6(신청인 유형별 국내 발명), a7(신청인 유형별 국내 실용신안), a2(연도 계열: 합계·국내·국외)
import re, sys, os, csv, hashlib
H, OUT = sys.argv[1], sys.argv[2]
sys.stdout.reconfigure(encoding='utf-8')


def rows(fn):
    t = open(os.path.join(H, fn), 'rb').read().decode('utf-8', 'replace')
    t = re.sub(r'<script.*?</script>|<style.*?</style>', '', t, flags=re.S)
    out = []
    for r in re.findall(r'<tr.*?</tr>', t, re.S):
        out.append([re.sub(r'<[^>]+>|&nbsp;|\s+', ' ', c).strip() for c in re.findall(r'<t[dh][^>]*>(.*?)</t[dh]>', r, re.S)])
    return out


def nums(r):
    return [int(x) for x in r if re.fullmatch(r'\d+', x.replace(',', ''))]


def total_row(rs, keys=('全国总计', '国内总计')):
    for i, r in enumerate(rs):
        if r and any(r[0].startswith(k) for k in keys):
            return i, nums(r)
    raise ValueError('total row not found')


def sha(fn):
    return hashlib.sha256(open(os.path.join(H, fn), 'rb').read()).hexdigest()[:16]


res = []
for y in range(2014, 2025):
    d = dict(year=y)
    if y <= 2019:
        i4, a4 = total_row(rows(f'y{y}_a4.html'))
        i7, a7 = total_row(rows(f'y{y}_a7.html'))
        i8, a8 = total_row(rows(f'y{y}_a8.html'))
        if y <= 2016:   # [累计 inv, um, des, 受理 inv, um, des, 申请 inv, um, des]
            assert len(a4) == 9, a4
            d.update(dom_inv=a4[3], dom_um=a4[4], dom_inv_shenqing=a4[6], dom_um_shenqing=a4[7], basis='受理量')
            assert len(a7) == 10 and len(a8) == 10, (a7, a8)
            s7, s8 = a7[5:], a8[5:]
        else:           # [inv, um, des]
            assert len(a4) == 3, a4
            d.update(dom_inv=a4[0], dom_um=a4[1], basis='申请量')
            s7, s8 = a7[:5], a8[:5]
        assert s7[0] == sum(s7[1:]) and s8[0] == sum(s8[1:]), (s7, s8)
        d.update(svc_inv=s7[0], svc_inv_univ=s7[1], svc_inv_inst=s7[2], svc_inv_ent=s7[3], svc_inv_org=s7[4],
                 svc_um=s8[0], nonsvc_inv=d['dom_inv'] - s7[0], nonsvc_um=d['dom_um'] - s8[0],
                 src_inv=f'year{y}/a/a4.html 全国总计 행; year{y}/a/a7.html 全国总计 행', src_um=f'year{y}/a/a4.html; year{y}/a/a8.html',
                 sha_a4=sha(f'y{y}_a4.html'), sha_a7=sha(f'y{y}_a7.html'), sha_a8=sha(f'y{y}_a8.html'))
    else:
        i6, a6 = total_row(rows(f'y{y}_a6.html'))
        i7, a7 = total_row(rows(f'y{y}_a7.html'))
        assert len(a6) == 6 and len(a7) == 6, (a6, a7)
        assert a6[0] == sum(a6[1:]) and a7[0] == sum(a7[1:]), (a6, a7)
        d.update(dom_inv=a6[0], svc_inv=a6[0] - a6[5], svc_inv_univ=a6[1], svc_inv_inst=a6[2], svc_inv_ent=a6[3], svc_inv_org=a6[4],
                 nonsvc_inv=a6[5], dom_um=a7[0], svc_um=a7[0] - a7[5], nonsvc_um=a7[5], basis='申请量',
                 src_inv=f'year{y}/a/a6.html 国内总计/全国总计 행(个人=비직무)', src_um=f'year{y}/a/a7.html',
                 sha_a6=sha(f'y{y}_a6.html'), sha_a7=sha(f'y{y}_a7.html'))
    d['svc_share_inv'] = d['svc_inv'] / d['dom_inv']
    d['svc_share_um'] = d['svc_um'] / d['dom_um']
    res.append(d)

# 연도 계열(합계·국내·국외 발명) — 2020판 a2(2015–2020), 2024판 a2(2019–2024)
ser = {}
for ed in (2020, 2024):
    rs = rows(f'y{ed}_a2.html'); grp = None
    for r in rs:
        if not r: continue
        if r[0].startswith('合计'): grp = 'tot'
        elif r[0].startswith('国内'): grp = 'dom'
        elif r[0].startswith('国外'): grp = 'for'
        cells = [c for c in r if c]
        yy = [c for c in cells if re.fullmatch(r'20\d\d', c)]
        if grp and yy:
            n = nums(cells[cells.index(yy[0]) + 1:])
            ser[(ed, grp, int(yy[0]))] = n[:3]
for d in res:
    y = d['year']
    for ed in (2020, 2024):
        for g in ('tot', 'dom', 'for'):
            v = ser.get((ed, g, y))
            if v:
                d[f'a2ed{ed}_{g}_inv'] = v[0]; d[f'a2ed{ed}_{g}_um'] = v[1]
keys = []
for d in res:
    for k in d:
        if k not in keys: keys.append(k)
with open(OUT, 'w', newline='', encoding='utf-8-sig') as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(res)
for d in res:
    print(d['year'], d['basis'], d['dom_inv'], d['svc_inv'], round(d['svc_share_inv'], 4), d['dom_um'], round(d['svc_share_um'], 4),
          d.get('a2ed2020_dom_inv'), d.get('a2ed2024_dom_inv'), d.get('a2ed2024_tot_inv') or d.get('a2ed2020_tot_inv'))
