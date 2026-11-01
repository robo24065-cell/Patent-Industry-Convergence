# 3단계 래퍼: final_run.py를 그대로 실행하고, 종료 시 이 프로세스의 최대 작업집합(PeakWorkingSetSize, Windows)을 출력 — 4GB 규칙 확인용. 모형·코드 변경 없음.
import sys, atexit, ctypes, ctypes.wintypes as wt, runpy, pathlib
class PMC(ctypes.Structure):
    _fields_ = [('cb', wt.DWORD), ('PageFaultCount', wt.DWORD), ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
                ('QuotaNonPagedPoolUsage', ctypes.c_size_t), ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
def peak():
    c = PMC(); c.cb = ctypes.sizeof(PMC)
    k = ctypes.WinDLL('kernel32'); psapi = ctypes.WinDLL('psapi')
    k.GetCurrentProcess.restype = wt.HANDLE; psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.POINTER(PMC), wt.DWORD]
    psapi.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    print(f'[mem] peak working set {c.PeakWorkingSetSize/2**30:.2f} GB, peak commit {c.PeakPagefileUsage/2**30:.2f} GB', flush=True)
atexit.register(peak)
here = pathlib.Path(__file__).resolve().parent
sys.argv = [str(here/'final_run.py')] + sys.argv[1:]
runpy.run_path(str(here/'final_run.py'), run_name='__main__')
