@rem set CUDA_VISIBLE_DEVICES=-1
@rem set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.0\libnvvp;%PATH%
@mkdir NetLogs
@copy *.py NetLogs\
"C:\Program Files\Python35\python.exe" activations.py 2>err