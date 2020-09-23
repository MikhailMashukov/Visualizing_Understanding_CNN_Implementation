@mkdir NetLogs
@copy *.py NetLogs\
@mkdir NetLogs\PyTorch
@copy PyTorch\* NetLogs\PyTorch\
"C:\Program Files\Python35\python.exe" PyTorch/PyTChipNets.py %1 >res 2>err