# -*- coding: utf-8 -*-
"""r3oa 검색어 사전 보강분(B): KSIC 균형 보강 98개 메인그룹 (2026-09-29 재시도 세션).
대상: 활동 비중이 낮던 KSIC에서 미수집 셀을 활동량 순으로 골라 KSIC별 활동 비중 >=60%를 목표(r3oa_balance_targets.csv).
출처: WIPO IPC 2026.01 영문 공개 제목(r3oa_raw/EN_ipc_title_list_20260101.zip)을 사람이 논문 용어로 옮김.
제공 데이터(특허 제목·출원인 등)는 검색어 작성에 쓰지 않았다(타당도 점검에만 로컬 사용)."""
TERMS_B = {
# 15220 신발
'A43B 13': ['shoe sole', 'outsole', 'midsole', 'footwear sole'],
'A43B 17': ['insole', 'foot orthosis', 'orthotic insole', 'shoe insert'],
'A43B 7': ['therapeutic footwear', 'diabetic footwear', 'orthopedic shoe', 'shoe ventilation'],
# 20202 합성수지
'C08L 77': ['polyamide blend', 'polyamide composite', 'nylon composite', 'polyamide 6'],
# 20411 도료
'C09D 133': ['acrylic coating', 'acrylic paint', 'acrylic resin coating', 'waterborne acrylic'],
'C09D 4': ['UV curable coating', 'radiation curable coating', 'photocurable coating', 'UV-curable resin'],
'C09D 167': ['polyester coating', 'powder coating', 'polyester resin coating', 'coil coating'],
# 23992 연마재
'B24B 29': ['polishing pad', 'buffing', 'surface polishing', 'magnetorheological finishing'],
'B24D 3': ['abrasive grain', 'grinding wheel', 'coated abrasive', 'abrasive tool'],
# 24219 비철금속 제련
'C22C 19': ['nickel-based superalloy', 'nickel superalloy', 'cobalt-based alloy', 'Inconel'],
'C22C 23': ['magnesium alloy', 'AZ31', 'AZ91', 'Mg alloy'],
'C22C 1': ['alloy preparation', 'metal matrix composite', 'mechanical alloying', 'powder metallurgy alloy'],
'C22B 7': ['metal recycling', 'scrap recycling', 'recovery of metals from waste', 'urban mining'],
'C22B 3': ['hydrometallurgy', 'leaching of ore', 'solvent extraction of metals', 'bioleaching'],
'C22C 14': ['titanium alloy', 'Ti-6Al-4V', 'beta titanium alloy', 'Ti alloy'],
# 25914 금속 성형
'B21D 37': ['stamping die', 'press die', 'forming tool', 'die wear'],
'B21D 7': ['tube bending', 'profile bending', 'rotary draw bending', 'bending of tubes'],
'B21D 5': ['sheet metal bending', 'press brake', 'air bending', 'roll forming'],
# 25924 절삭공구
'B23C 5': ['milling cutter', 'end mill', 'milling tool', 'face milling cutter'],
# 25932 철물·자물쇠
'E05B 15': ['lock mechanism', 'door lock', 'latch bolt', 'locking device'],
'E05B 47': ['electronic lock', 'electromagnetic lock', 'smart lock', 'electric door lock'],
'E05B 65': ['vehicle door lock', 'drawer lock', 'locking system', 'lock for vehicle'],
'E05B 63': ['mortise lock', 'deadbolt', 'lock structure', 'multipoint lock'],
'E05B 27': ['pin tumbler lock', 'cylinder lock', 'lock picking', 'lock cylinder'],
'E05B 17': ['lock accessory', 'door handle lock', 'keyhole', 'lock indicator'],
'E05B 19': ['mechanical key', 'key duplication', 'key blank', 'physical key'],
'E05D 3': ['door hinge', 'hinge pin', 'hinge mechanism', 'concealed hinge'],
'E05B 67': ['padlock', 'shackle lock', 'combination padlock', 'U-lock'],
# 26529 음향기기
'H04R 1': ['loudspeaker enclosure', 'microphone array', 'earphone', 'acoustic transducer'],
# 27199 의료기기
'A61M 5': ['syringe', 'infusion pump', 'insulin pen', 'intravenous catheter'],
'A61M 16': ['mechanical ventilation', 'ventilator', 'endotracheal tube', 'CPAP'],
# 28511 주방가전
'F25D 23': ['refrigerator cabinet', 'refrigerator door', 'vacuum insulation panel', 'refrigerator design'],
'A47J 27': ['cooking vessel', 'pressure cooker', 'rice cooker', 'cookware'],
'F25D 11': ['domestic refrigerator', 'household refrigerator', 'refrigerator freezer', 'refrigerator energy consumption'],
'A47J 19': ['food strainer', 'juicer', 'food mashing', 'household juicer'],
'A47L 15': ['dishwasher', 'dish washing machine', 'dishwashing', 'automatic dishwasher'],
# 29131 펌프
'F04C 2': ['gear pump', 'vane pump', 'rotary pump', 'screw pump'],
# 29133 밸브
'F16K 1': ['globe valve', 'lift valve', 'poppet valve', 'valve seat'],
'F16K 5': ['ball valve', 'plug valve', 'rotary valve', 'valve cock'],
'F16K 17': ['safety valve', 'pressure relief valve', 'relief valve', 'check valve'],
'F16K 3': ['gate valve', 'sliding valve', 'slide valve', 'valve gate'],
# 29150 노·버너
'F27D 1': ['furnace lining', 'refractory lining', 'furnace wall', 'furnace refractory'],
'F27B 9': ['tunnel kiln', 'continuous furnace', 'roller hearth furnace', 'belt furnace'],
'F27B 14': ['crucible furnace', 'induction melting furnace', 'glass melting furnace', 'tank furnace'],
'F27B 7': ['rotary kiln', 'rotary furnace', 'rotary drum furnace', 'cement kiln'],
'F27B 17': ['laboratory furnace', 'vacuum furnace', 'industrial furnace', 'muffle furnace'],
'F27D 17': ['waste heat recovery furnace', 'furnace waste heat', 'furnace off-gas', 'flue gas heat recovery'],
'F23G 7': ['incineration of hazardous waste', 'sludge incineration', 'medical waste incineration', 'waste gas incineration'],
'F23D 14': ['gas burner', 'premixed burner', 'burner design', 'swirl burner'],
'F23G 5': ['waste incineration', 'incinerator', 'municipal solid waste incineration', 'waste combustion'],
'F27D 3': ['furnace charging', 'charge handling furnace', 'burden distribution', 'furnace tapping'],
'F27B 21': ['sintering machine', 'iron ore sintering', 'sinter plant', 'sinter strand'],
'F24B 1': ['wood stove', 'cookstove', 'biomass stove', 'solid fuel stove'],
# 29162 승강기
'B66B 5': ['elevator safety', 'elevator fault', 'elevator brake', 'elevator emergency'],
'B66B 11': ['elevator car', 'elevator machine', 'traction elevator', 'elevator hoist'],
'B66B 1': ['elevator control', 'elevator group control', 'elevator dispatching', 'elevator scheduling'],
'B66B 7': ['elevator rope', 'elevator guide rail', 'hoisting rope', 'elevator vibration'],
# 29210 농기계
'A01C 7': ['seed drill', 'precision seeding', 'seeder', 'planter metering'],
'A01D 34': ['lawn mower', 'mower', 'robotic mower', 'mowing machine'],
'A01B 49': ['combined tillage', 'tillage machine', 'combined agricultural machine', 'seedbed preparation'],
'A01C 11': ['transplanter', 'rice transplanter', 'seedling transplanting', 'transplanting machine'],
'A01D 41': ['combine harvester', 'grain combine', 'combine header', 'harvester threshing'],
'A01B 33': ['rotary tiller', 'rotavator', 'rotary tillage', 'power tiller'],
'A01F 12': ['threshing drum', 'threshing unit', 'threshing cylinder', 'grain separation'],
'A01D 45': ['crop harvesting machine', 'sugarcane harvester', 'corn harvester', 'forage harvester'],
'A01C 15': ['fertilizer spreader', 'fertilizer distributor', 'variable rate fertilizer', 'manure spreader'],
'A01C 5': ['furrow opener', 'no-till planter', 'furrow opening', 'seed furrow'],
'A01F 11': ['maize sheller', 'corn sheller', 'crop thresher', 'peanut thresher'],
# 29241 건설·광산기계
'E21B 7': ['drilling technology', 'directional drilling', 'horizontal drilling', 'rotary drilling'],
'E02D 5': ['pile foundation', 'sheet pile', 'bored pile', 'driven pile'],
'B28C 5': ['concrete mixer', 'concrete mixing', 'mortar mixing', 'concrete batching'],
'E21B 33': ['well cementing', 'wellbore sealing', 'packer', 'well plugging'],
'E21B 21': ['drilling fluid', 'drilling mud', 'hole cleaning', 'managed pressure drilling'],
'E21B 47': ['well logging', 'measurement while drilling', 'downhole measurement', 'borehole survey'],
# 29280 산업용 로봇
'B25J 9': ['robot manipulator', 'industrial robot', 'robot arm control', 'robot motion planning'],
# 30393 차량 시트
'B60N 2': ['vehicle seat', 'car seat', 'automotive seat', 'child safety seat'],
# 31202 철도차량
'B61F 5': ['railway bogie', 'bogie', 'rail vehicle suspension', 'bogie frame'],
'B61L 27': ['train control system', 'CBTC', 'railway traffic control', 'railway signalling'],
'B61D 17': ['railway carbody', 'rail vehicle body', 'train body', 'carbody structure'],
'B61L 23': ['train collision avoidance', 'level crossing', 'railway safety', 'train protection'],
'B61C 17': ['locomotive', 'railcar', 'traction drive locomotive', 'diesel locomotive'],
'B61D 27': ['train air conditioning', 'railway vehicle ventilation', 'railway HVAC', 'train ventilation'],
'B61B 1': ['railway station', 'rail network', 'marshalling yard', 'railway platform'],
'B61L 25': ['train positioning', 'train location', 'railway vehicle identification', 'train localization'],
'B61L 15': ['train signalling', 'onboard signalling', 'cab signalling', 'train driver display'],
'B61D 19': ['train door', 'railway door', 'platform screen door', 'rail vehicle door'],
# 31312 항공기
'B64C 27': ['helicopter rotor', 'rotorcraft', 'helicopter', 'rotor blade'],
'B64C 39': ['unmanned aerial vehicle', 'drone', 'UAV', 'multirotor'],
'B64D 1': ['aerial delivery', 'drone delivery', 'aerial spraying', 'payload release'],
# 33301 운동기구
'A63B 23': ['exercise device', 'rehabilitation exercise', 'hand exerciser', 'core training'],
'A63B 22': ['treadmill', 'exercise bike', 'elliptical trainer', 'rowing machine'],
'A63B 21': ['resistance training', 'strength training equipment', 'resistance band', 'weight machine'],
# 33309 스포츠 용품
'A63B 53': ['golf club', 'golf club head', 'golf driver', 'golf shaft'],
'A63B 37': ['golf ball', 'ball aerodynamics', 'solid ball', 'soccer ball'],
'A63C 17': ['roller skate', 'skateboard', 'inline skate', 'electric skateboard'],
'A63B 67': ['sports equipment', 'table tennis', 'badminton', 'sporting goods'],
'A63B 59': ['baseball bat', 'cricket bat', 'hockey stick', 'bat performance'],
'A63B 49': ['tennis racket', 'badminton racket', 'racket string', 'racquet'],
}
