# C09-T4 h0: 한국천문연구원 천문우주지식정보 음력→양력 변환(공개, 로그인 없음)으로 설날(1/1)·부처님오신날(4/8)·추석(8/15) 양력일 2000–2026 확보. 1.2초 간격.
import json, os, time, urllib.request
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(W, 'download', 'kasi'); os.makedirs(D, exist_ok=True)
out = {}
for y in range(1999, 2027):
    for mm, dd in [('01', '01'), ('04', '08'), ('08', '15')]:
        fn = os.path.join(D, f'lunc_{y}{mm}{dd}.json')
        if not os.path.exists(fn):
            req = urllib.request.Request(f'https://astro.kasi.re.kr/life/lunc?yyyy={y}&mm={mm}&dd={dd}', headers={'User-Agent': 'Mozilla/5.0'})
            raw = urllib.request.urlopen(req, timeout=30).read().decode('utf-8')
            open(fn, 'w', encoding='utf-8').write(raw)
            time.sleep(1.2)
        js = json.load(open(fn, encoding='utf-8'))
        r = [x for x in js if x['LUNC_LEAP_MM'] == '평'][0]
        out[f'{y}-{mm}-{dd}'] = f"{r['SOLC_YYYY']}-{r['SOLC_MM']}-{r['SOLC_DD']}"
json.dump(out, open(os.path.join(W, 'cache', 'kasi_lunar.json'), 'w', encoding='utf-8'), indent=1)
print(len(out), list(out.items())[:6])
