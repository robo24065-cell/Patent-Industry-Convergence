# -*- coding: utf-8 -*-
"""제출용 분석 코드 패키지(zip) 생성기.

git이 추적하는 파일만 `git archive`로 묶는다. 묶기 전에 커밋 가드(.githooks/guard.py)를
묶을 파일 전부에 다시 돌리고, 하나라도 막히면 zip을 만들지 않고 끝낸다.
  - guard.py 규칙: 데이터·압축 확장자, 5MB 초과 파일, 허용 폴더 밖 csv/json, 차단 파일명,
    저장소 루트의 API키_기록.txt(있을 때만)에 적힌 키 문자열, 하드코딩된 키 패턴
  - 이 스크립트가 더 하는 일: 작업 트리와 묶을 커밋이 같은지 확인(가드가 본 내용 = zip 내용),
    zip 안 파일 목록 대조, 이메일 형식 문자열(개인정보 후보) 경고, 절대경로 의존 파일 수 안내

사용(저장소 안팎 어디서든):
  python tools/make_code_package.py                  # HEAD → <저장소>/dist/분석코드_패키지.zip
  python tools/make_code_package.py --strict         # 경고(키 원장 없음, 이메일 후보)도 실패로 처리
  python tools/make_code_package.py --check-only     # 검사만 하고 zip은 만들지 않음
  python tools/make_code_package.py --ref <커밋|태그> --out <경로.zip> --prefix <폴더명/>

종료코드: 0 성공 · 1 가드나 검사에 걸림 · 2 실행 조건 오류(git 없음, 작업 트리 불일치 등)

키 문자열 대조는 guard.py가 하며 키 값은 어디에도 출력하지 않는다. 키 원장은 저장소 루트에서만
찾으므로, 최종 제출본은 원장이 있는 팀 PC 원본 작업 폴더에서 --strict로 만든다.
이 파일에는 키·데이터·개인정보를 넣지 않는다.
"""
import argparse
import hashlib
import os
import re
import subprocess
import sys
import zipfile

DEFAULT_OUT = os.path.join('dist', '분석코드_패키지.zip')
DEFAULT_PREFIX = '분석코드_패키지/'
GUARD_REL = os.path.join('.githooks', 'guard.py')
CMD_BUDGET = 24000  # Windows 명령줄 한도(32,767자)보다 여유 있게 파일 목록을 나눈다
EMAIL = re.compile(
    r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*'
    r'\.(?:com|net|org|edu|gov|mil|int|info|biz|io|ai|me|co|kr|jp|cn|us|uk|de|fr|eu|ac|re|go)\b',
    re.IGNORECASE)
ABS_HOME = re.compile(r'[A-Za-z]:[\\/]+Users[\\/]+', re.IGNORECASE)
CODE_EXT = ('.py', '.js', '.ps1')


class Stop(Exception):
    """실행 중단(code: 1 검사 실패, 2 실행 조건 오류)."""

    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code


def say(msg=''):
    print(msg, flush=True)


def git(args, cwd):
    try:
        return subprocess.run(['git', '-c', 'core.quotepath=false'] + args, cwd=cwd, capture_output=True)
    except FileNotFoundError:
        raise Stop(2, 'git 명령을 찾지 못했다(PATH 확인).')


def text(b):
    return b.decode('utf-8', errors='replace').strip()


def repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    r = git(['rev-parse', '--show-toplevel'], here)
    if r.returncode != 0:
        raise Stop(2, '이 스크립트가 git 저장소 안에 있지 않다: ' + text(r.stderr))
    return os.path.normpath(text(r.stdout))


def resolve_commit(root, ref):
    r = git(['rev-parse', '--verify', '--quiet', ref + '^{commit}'], root)
    if r.returncode != 0:
        raise Stop(2, f'커밋을 찾지 못했다: {ref}')
    return text(r.stdout)


def tracked_files(root, commit):
    r = git(['ls-tree', '-r', '-z', '--name-only', commit], root)
    if r.returncode != 0:
        raise Stop(2, 'git ls-tree 실패: ' + text(r.stderr))
    files = [p.decode('utf-8') for p in r.stdout.split(b'\0') if p]
    if not files:
        raise Stop(2, '묶을 파일이 없다.')
    return files


def ensure_worktree_matches(root, commit):
    """가드는 작업 트리 파일을 읽으므로, 작업 트리가 묶을 커밋과 같아야 검사 내용과 zip 내용이 같다."""
    r = git(['diff', '--name-only', commit, '--'], root)
    if r.returncode != 0:
        raise Stop(2, 'git diff 실패: ' + text(r.stderr))
    changed = [l for l in text(r.stdout).splitlines() if l.strip()]
    if changed:
        lines = '\n'.join('  - ' + c for c in changed[:20])
        more = f'\n  … 외 {len(changed) - 20}개' if len(changed) > 20 else ''
        raise Stop(2, f'작업 트리가 묶을 커밋({commit[:10]})과 다르다({len(changed)}개 파일).\n'
                      f'{lines}{more}\n커밋하거나 되돌린 뒤(다른 커밋을 묶으려면 그 커밋을 체크아웃한 뒤) 다시 실행한다.')


def chunks(paths, budget=CMD_BUDGET):
    part, size = [], 0
    for p in paths:
        cost = len(p) + 3
        if part and size + cost > budget:
            yield part
            part, size = [], 0
        part.append(p)
        size += cost
    if part:
        yield part


def run_guard(root, files):
    """guard.py를 파일 목록 조각마다 실행한다. 반환: (막힌 줄 목록, 대조한 키 패턴 수)."""
    guard = os.path.join(root, GUARD_REL)
    if not os.path.isfile(guard):
        raise Stop(2, f'커밋 가드가 없다: {GUARD_REL}')
    blocked, n_keys = [], None
    for part in chunks(files):
        r = subprocess.run([sys.executable, guard] + part, cwd=root, capture_output=True)
        out = r.stdout.decode('utf-8', errors='replace')
        if r.returncode == 0 and 'GUARD OK' in out:
            m = re.search(r'GUARD OK (\d+) key patterns', out)
            if m:
                n_keys = int(m.group(1))
            continue
        if r.returncode == 1 and 'GUARD BLOCKED' in out:
            blocked += [l.strip() for l in out.splitlines() if l.strip().startswith('- ')]
            continue
        raise Stop(2, f'guard.py가 예상하지 못한 방식으로 끝났다(종료코드 {r.returncode}).\n'
                      + out[-2000:] + r.stderr.decode('utf-8', errors='replace')[-2000:])
    return blocked, (n_keys or 0)


def mask(addr):
    user, _, dom = addr.partition('@')
    head, _, tld = dom.rpartition('.')
    return f'{user[:1]}***@{head[:1]}***.{tld}'


def scan_extra(root, files):
    """개인정보 후보(이메일 형식, 모든 파일)와 사용자 폴더 절대경로에 기대는 코드 파일을 찾는다."""
    emails, abs_files = [], 0
    for f in files:
        try:
            with open(os.path.join(root, f), encoding='utf-8', errors='ignore') as fh:
                body = fh.read()
        except OSError:
            continue
        if f.lower().endswith(CODE_EXT) and ABS_HOME.search(body):
            abs_files += 1
        for i, line in enumerate(body.splitlines(), 1):
            for m in EMAIL.finditer(line):
                emails.append((f, i, mask(m.group(0))))
    return emails, abs_files


def ext_summary(files):
    cnt = {}
    for f in files:
        e = os.path.splitext(f)[1].lower() or os.path.basename(f)
        cnt[e] = cnt.get(e, 0) + 1
    return ', '.join(f'{k} {v}' for k, v in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0])))


def zip_names(zf, prefix):
    names = []
    for info in zf.infolist():
        name = info.filename
        if not info.flag_bits & 0x800:  # UTF-8 표시가 없으면 zipfile이 cp437로 읽는다
            try:
                name = name.encode('cp437').decode('utf-8')
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
        if name.endswith('/'):
            continue
        if prefix and name.startswith(prefix):
            name = name[len(prefix):]
        names.append(name)
    return names


def build_zip(root, commit, files, out_path, prefix):
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    tmp = out_path + '.tmp'
    if os.path.exists(tmp):
        os.remove(tmp)
    args = ['archive', '--format=zip', '-o', tmp]
    if prefix:
        args.append('--prefix=' + prefix)
    r = git(args + [commit], root)
    if r.returncode != 0:
        raise Stop(2, 'git archive 실패: ' + text(r.stderr))
    try:
        with zipfile.ZipFile(tmp) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise Stop(1, f'zip 무결성 검사 실패: {bad}')
            names = zip_names(zf, prefix)
            total = sum(i.file_size for i in zf.infolist())
        extra, missing = sorted(set(names) - set(files)), sorted(set(files) - set(names))
        if extra or missing:
            raise Stop(1, f'zip 목록이 검사한 목록과 다르다(검사 안 된 파일 {len(extra)}개, 빠진 파일 {len(missing)}개). '
                          f'예: {(extra + missing)[:5]}')
        os.replace(tmp, out_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    h = hashlib.sha256()
    with open(out_path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return len(names), total, os.path.getsize(out_path), h.hexdigest()


def main(argv=None):
    ap = argparse.ArgumentParser(description='제출용 분석 코드 패키지(zip) — guard.py 재검사 후 git archive')
    ap.add_argument('--ref', default='HEAD', help='묶을 커밋·태그·브랜치(기본 HEAD)')
    ap.add_argument('--out', default=None, help=f'zip 경로(기본 <저장소>/{DEFAULT_OUT})')
    ap.add_argument('--prefix', default=DEFAULT_PREFIX,
                    help=f"zip 안 최상위 폴더(기본 '{DEFAULT_PREFIX}', 빈 문자열이면 없음)")
    ap.add_argument('--check-only', action='store_true', help='검사만 하고 zip은 만들지 않는다')
    ap.add_argument('--strict', action='store_true', help='경고(키 원장 없음, 이메일 후보)도 실패로 처리')
    a = ap.parse_args(argv)
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    prefix = a.prefix
    if prefix and not prefix.endswith('/'):
        prefix += '/'
    try:
        root = repo_root()
        commit = resolve_commit(root, a.ref)
        files = tracked_files(root, commit)
        say(f'[1/4] 대상: {a.ref} = {commit[:10]}, 추적 파일 {len(files)}개 ({ext_summary(files)})')
        ensure_worktree_matches(root, commit)
        say('[2/4] 작업 트리 = 묶을 커밋 (가드가 읽는 내용과 zip 내용이 같다)')

        blocked, n_keys = run_guard(root, files)
        if blocked:
            say(f'[3/4] 커밋 가드에 걸림 {len(blocked)}건 — zip을 만들지 않는다')
            for l in blocked:
                say('   ' + l)
            return 1
        say(f'[3/4] 커밋 가드 통과 (키 패턴 {n_keys}개 대조)')

        warnings = []
        if n_keys == 0:
            warnings.append('저장소 루트에 키 원장(API키_기록.txt)이 없어 키 문자열 대조를 하지 못했다 '
                            '(하드코딩 키 패턴만 검사됨). 최종본은 원장이 있는 팀 PC 원본 작업 폴더에서 만든다.')
        emails, abs_files = scan_extra(root, files)
        if emails:
            spots = '; '.join(f'{f}:{i} ({m})' for f, i, m in emails[:10])
            warnings.append(f'이메일 형식 문자열 {len(emails)}건(개인정보 후보): {spots}')
        for w in warnings:
            say('   경고: ' + w)
        if abs_files:
            say(f'   안내: 사용자 폴더 절대경로(<드라이브>:\\Users\\…)를 쓰는 코드 파일 {abs_files}개 '
                '— 팀 PC 전용 경로(docs/SUBMISSION_CHECKLIST.md 참고)')
        if warnings and a.strict:
            say('[4/4] --strict: 경고가 있어 zip을 만들지 않는다')
            return 1
        if a.check_only:
            say('[4/4] --check-only: 검사만 했다(zip 생성 안 함)')
            return 0

        out_path = os.path.abspath(a.out) if a.out else os.path.join(root, DEFAULT_OUT)
        n, total, size, digest = build_zip(root, commit, files, out_path, prefix)
        say(f'[4/4] 생성: {out_path}')
        say(f'      파일 {n}개, 원본 {total / 1e6:.2f} MB → zip {size / 1e6:.2f} MB, '
            f'zip 안 최상위 폴더 {prefix or "(없음)"}')
        say(f'      SHA-256 {digest}')
        return 0
    except Stop as e:
        say(('오류: ' if e.code == 2 else '실패: ') + str(e))
        return e.code


if __name__ == '__main__':
    sys.exit(main())
