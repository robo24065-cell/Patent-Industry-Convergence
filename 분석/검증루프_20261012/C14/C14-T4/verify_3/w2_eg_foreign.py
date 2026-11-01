# C14-T4 verify_3 w2: U6xEg(CN) '외국 상대' 판별 점검. 작업은 '상대 키에 한자 없음'을 외국으로 봤다.
# CN 키는 외국 기업도 한자 음역(三星电子, 高通, 索尼 ...)이라 과소 집계가 의심된다. X1' 엄격 사건(b1 정의 재구현)의 상대 키를
# (1) 작업 규칙 (2) 넓은 규칙(중국 지명·대학·연구원 등 국내 표지가 없으면 '외국 가능') (3) 알려진 외국 기업 표지 목록으로 분류하고
# 잠금 예상 상한을 다시 낸다. 입력 C13 cache(읽기만), pub < 20250101.
import sys, os, re, json
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
C13C = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전\분석\검증루프_20261012\C13\C13-T4\cache'
app = pd.read_parquet(C13C + r'\app_CN.parquet'); assert app.pub.max() < 20250101
HAN = re.compile(r'[一-鿿]')
PROV = '北京|上海|天津|重庆|河北|山西|辽宁|吉林|黑龙江|江苏|浙江|安徽|福建|江西|山东|河南|湖北|湖南|广东|海南|四川|贵州|云南|陕西|甘肃|青海|内蒙古|广西|西藏|宁夏|新疆|深圳|广州|杭州|南京|苏州|武汉|成都|西安|无锡|宁波|青岛|厦门|合肥|长沙|郑州|济南|大连|沈阳|哈尔滨|长春|福州|东莞|佛山|珠海|中山|常州|南通|温州|绍兴|嘉兴|台州|金华|徐州|扬州|镇江|泰州|昆山|烟台|潍坊|淄博|洛阳|株洲|湘潭|芜湖|贵阳|昆明|南宁|兰州|太原|石家庄|唐山|保定|呼和浩特|乌鲁木齐|银川|西宁|拉萨|海口|三亚|南昌|赣州|宜宾|绵阳|柳州|桂林|鞍山|抚顺|吉林|大庆|包头'
DOM = re.compile(PROV + r'|中国|中华|国家|大学|学院|研究院|研究所|科学院|集团|医院|电网|中科|中石|中铁|中建|中交|中船|中航|中电|中车|华为|中兴|京东方|股份|省|市|县|区')
FOR_MARK = re.compile(r'美国|德国|日本|韩国|法国|英国|瑞士|荷兰|瑞典|芬兰|丹麦|挪威|意大利|西班牙|比利时|奥地利|以色列|新加坡|加拿大|澳大利亚|欧洲|株式|三星|索尼|高通|佳能|松下|日立|西门子|巴斯夫|拜耳|LG|英特尔|丰田|本田|日产|电装|东芝|富士|爱普生|精工|三菱|住友|村田|京瓷|夏普|理光|尼康|奥林巴斯|诺基亚|爱立信|飞利浦|博世|采埃孚|默克|赢创|陶氏|杜邦|3M|霍尼韦尔|通用电气|苹果|微软|谷歌|亚马逊|IBM|甲骨文|思科|应用材料|德州仪器|罗伯特|雀巢|联合利华|宝洁|强生|辉瑞|罗氏|诺华|阿斯利康|赛诺菲|葛兰素|现代|起亚|SK|斗山|浦项|乐金')
rows = []
nk = app.groupby('appl_no').key.nunique()
m = app[app.appl_no.map(nk) >= 2][['appl_no', 'key', 'pub']]
eo = m.merge(m[['appl_no', 'key']], on='appl_no', suffixes=('', '_g')); eo = eo[eo.key != eo.key_g]
first = eo.groupby(['key', 'key_g']).pub.min().rename('first').reset_index(); del eo
samples = []
for t0 in range(2016, 2025):
    end = t0 * 10000 + 1231
    h5 = app[(app.pub > end - 50000) & (app.pub <= end)]
    mc = h5.groupby(['key', 'ksic']).size().rename('n').reset_index().sort_values(['key', 'n', 'ksic'], ascending=[True, False, True])
    modal = mc.drop_duplicates('key').set_index('key').ksic
    cnt3 = app[(app.pub > end - 30000) & (app.pub <= end)].groupby('key').appl_no.nunique()
    act = cnt3[cnt3 >= 5].index
    r = {'t0': t0, 'x1p_risk': int(len(act))}
    if t0 <= 2023:
        lo, hi = (t0 + 1) * 10000 + 101, (t0 + 1) * 10000 + 630
        nw_ = first[(first['first'] >= lo) & (first['first'] <= hi) & first.key.isin(act)].copy()
        nw_['mf'] = nw_.key.map(modal); nw_['mg'] = nw_.key_g.map(modal)
        st = nw_[nw_.mg.notna() & (nw_.mg != nw_.mf)].copy()
        g = st.key_g.astype(str); f = st.key.astype(str)
        st['for_work'] = ~g.str.contains(HAN)
        st['for_broad'] = ~g.str.contains(DOM)          # 국내 표지 없음 = 외국 가능(상한 쪽)
        st['for_mark'] = st.for_work | g.str.contains(FOR_MARK)   # 외국 표지 있음(하한 쪽, 대략)
        # 국경 간 = 두 쪽 중 정확히 한 쪽이 외국(넓은 규칙 기준) — 외국 기업끼리 관계는 국경 간이 아닐 수 있다
        fb = ~f.str.contains(DOM)
        st['xb_broad'] = st.for_broad ^ fb
        r.update({'x1p_pos_strict_firms': int(st.key.nunique()), 'pairs': int(len(st))})
        for c in ['for_work', 'for_broad', 'for_mark', 'xb_broad']:
            r[c + '_pair_share'] = float(st[c].mean()); r[c + '_firms'] = int(st[st[c]].key.nunique())
        if t0 == 2023:
            samples = st.sample(min(120, len(st)), random_state=7)[['key', 'key_g', 'for_work', 'for_broad', 'for_mark']].values.tolist()
    rows.append(r); print(r, flush=True)
df = pd.DataFrame(rows)
d = df[df.t0.isin([2022, 2023])]
rk = int(df[df.t0 == 2024].x1p_risk.iloc[0])
res = {'by_t0': rows, 'lock_exp': {c: float(rk * (d[c + '_firms'] / d.x1p_risk).mean()) for c in ['for_work', 'for_broad', 'for_mark', 'xb_broad']},
       'x1p_lock_exp_CN_check': float(rk * (d.x1p_pos_strict_firms / d.x1p_risk).mean()),
       'dev_sum_firms': {c: int(df[c + '_firms'].fillna(0).sum()) for c in ['for_work', 'for_broad', 'for_mark', 'xb_broad']},
       'sample_2023': samples}
json.dump(res, open(os.path.join(HERE, 'w2_out.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print(json.dumps({k: v for k, v in res.items() if k not in ('by_t0', 'sample_2023')}, ensure_ascii=False, indent=1))
for s in samples[:120]:
    print(s)
