# -*- coding: utf-8 -*-
"""연계 프레임워크 산출물: 특허 KSIC 라벨 → 외부 데이터별 연계표·연계율 요약(ext_link_trade.csv, ext_link_summary.csv)."""
from n1lib import *
K50 = load_kosis50(("T02",))["T02"].index; nm = ksic_names()
m = load_map(); x = kr_export_ksic(); c = cn_share()
pat = pd.read_parquet(f"{CACHE}/pat_kcy.parquet")
t = pd.DataFrame({"ksic": K50, "name": [nm.get(k) for k in K50]})
t["n_hs6"] = t.ksic.map(m.groupby("k5").hs6.nunique()).fillna(0).astype(int)
t["n_hs6_multi"] = t.ksic.map(m[m.w < 1].groupby("k5").hs6.nunique()).fillna(0).astype(int)
t["cn_share_2018"] = t.ksic.map(c[c.year == 2018].set_index("ksic").cn_share_imp).round(1)
t["kr_export_2018_musd"] = t.ksic.map((x[2018] / 1e6).round(1)) if 2018 in x.columns else np.nan
t["patents_5c_2008_2018"] = t.ksic.map(pat[pat.y.between(2008, 2018)].groupby("ksic").n.sum()).fillna(0).astype(int)
t.to_csv(f"{HERE}/ext_link_trade.csv", index=False, encoding="utf-8-sig")
io = pd.read_csv(f"{HERE}/ext_link_io.csv", dtype=str); ki = pd.read_csv(f"{HERE}/ext_link_kosis_indices.csv", dtype=str); rg = pd.read_csv(f"{HERE}/ext_link_registry.csv", dtype={"ksic": str})
S = pd.DataFrame([
 ("제공 5개국 특허(KSIC 라벨)", "KSIC 5자리 그대로", 50, 50, "15.1M (출원번호,KSIC) 쌍 → (KSIC,국가,연도) 집계"),
 ("KOSIS 광업제조업조사 5자리(타깃·기준선)", "KSIC 5자리 직접", 50, 50, "10차 2008–2020 + 11차 2021 체인, 2010 부재, T02 마스킹 4건(2008·09·11·12)"),
 ("UN Comtrade 대중·전세계 수입(HS6)", "ksic_hs6_map 1,229쌍(HS6 1,019 → KSIC 48), 다중대응 1/m", 50, int(t.n_hs6.gt(0).sum()), "나머지 2개 순위 0.5"),
 ("UN Comtrade 한국 수출(HS6)", "동일 연계표", 50, int(t.kr_export_2018_musd.notna().sum()), "모듈 D 타깃"),
 ("KOSIS 광공업생산지수(월, 3자리)", "KSIC 앞 3자리", 50, int(ki.pi_code.notna().sum()), "3자리 전부 연결"),
 ("KOSIS 내수·수출 출하지수", "2자리(C26만 3자리)", 50, int(ki.sh_code.notna().sum()), "실험 E10 롤백"),
 ("KOSIS 업종별 수입점유비", "2자리", 50, int(ki.ims_code.notna().sum()), "2010~ , 실험 E13 롤백"),
 ("한국은행 산업연관표 2010·2015(통합소분류)", "KSIC → IO 기본부문 대조표 → 3자리", 50, int(io.k5.nunique()), "실험 E17·E18·E39 롤백, 모듈 D E19 롤백"),
 ("KR 등록원부(수집, registry_full)", "등록특허 KSIC 라벨", 50, int(rg.in_registry.sum()), "실험 E27·E28 롤백"),
], columns=["데이터", "연결 경로", "대상 KSIC", "연결 KSIC", "비고"])
S["연계율"] = (S["연결 KSIC"] / S["대상 KSIC"]).round(2)
S.to_csv(f"{HERE}/ext_link_summary.csv", index=False, encoding="utf-8-sig"); print(S.to_string())
