import ctypes,os,time;ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
while os.popen('tasklist /FI "PID eq 14808"').read().count('python')>0: time.sleep(60)
