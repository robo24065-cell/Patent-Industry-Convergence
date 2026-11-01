# C17-T4 공용 상수·함수
import sys, datetime as _dt
B = r'C:\Users\PC\Downloads\특허-산업 융합데이터 활용 아이디어 공모전'
MC = B + r'\분석\모델경쟁_20260927'
META = (MC + r'\_shared\meta\*.parquet').replace('\\', '/')
W = B + r'\분석\검증루프_20261012\C17\C17-T4'
C = W + r'\cache'
LOCKDIR = W + r'\cache\LOCKED'      # 잠금 이벤트. P0 단계 스크립트는 열지 않는다.
DL = W + r'\download'
LOCK_START = 20250101               # 잠금 창 시작(마감일 D 기준)
SEED = 20261017

# 유지료 납부 코드 → 단계(1=4년차, 2=8년차, 3=12년차), 엔티티(L/S/M)
PAY = {}
for c, s, e in [('M1551', 1, 'L'), ('M1552', 2, 'L'), ('M1553', 3, 'L'),
                ('M2551', 1, 'S'), ('M2552', 2, 'S'), ('M2553', 3, 'S'),
                ('M3551', 1, 'M'), ('M3552', 2, 'M'), ('M3553', 3, 'M'),
                ('M183', 1, 'L'), ('M184', 2, 'L'), ('M185', 3, 'L'),
                ('M283', 1, 'S'), ('M284', 2, 'S'), ('M285', 3, 'S'),
                ('M170', 1, 'S'), ('M171', 2, 'S'), ('M172', 3, 'S'),
                ('M173', 1, 'L'), ('M174', 2, 'L'), ('M175', 3, 'L'),
                ('M273', 1, 'S'), ('M274', 2, 'S'), ('M275', 3, 'S'),
                ('F170', 1, 'S'), ('F171', 2, 'S'), ('F172', 3, 'S'),
                ('F173', 1, 'L'), ('F174', 2, 'L'), ('F175', 3, 'L'),
                ('F273', 1, 'S'), ('F274', 2, 'S'), ('F275', 3, 'S')]:
    PAY[c] = (s, e)
SURCH_GRACE = {'M1554', 'M1555', 'M1556', 'M2554', 'M2555', 'M2556', 'M3554', 'M3555', 'M3556',
               'M181', 'M182', 'M186', 'M281', 'M282', 'M286', 'M176', 'M177', 'M277', 'F176', 'F177', 'F277'}
SURCH_PETITION = {'M1557', 'M1558', 'M3557', 'M3558', 'M187', 'M188', 'M178'}


def add_years(ymd, n):
    y, m, d = ymd // 10000, (ymd // 100) % 100, ymd % 100
    y += n
    if m == 2 and d == 29:
        d = 28
    return y * 10000 + m * 100 + d


def add_months(ymd, n):
    y, m, d = ymd // 10000, (ymd // 100) % 100, ymd % 100
    t = y * 12 + (m - 1) + n
    y, m = t // 12, t % 12 + 1
    import calendar
    d = min(d, calendar.monthrange(y, m)[1])
    return y * 10000 + m * 100 + d


def ymd2date(x):
    return _dt.date(x // 10000, (x // 100) % 100, x % 100)


def deadlines(G):
    """단계별 마감일(유예 6개월 끝 = 미납 시 소멸일) D_s = G + 4/8/12년."""
    return {s: add_years(G, 4 * s) for s in (1, 2, 3)}


def lock_zone(G, e):
    """잠금 구역 이벤트 판정: e >= 2024-01-01 이고, D_s >= 2025-01-01 인 가장 이른 단계 D에 대해 e >= D - 13개월."""
    if e < 20240101:
        return False
    for s in (1, 2, 3):
        D = add_years(G, 4 * s)
        if D >= LOCK_START:
            return e >= add_months(D, -13)
    return False


def out(*a):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    print(*a, flush=True)
