@echo off
rem NOX US tarama gorevleri - Task Scheduler wrapper
rem Kullanim: us_task.bat script1.py [script2.py ...]
cd /d D:\Nyx_trading\nox-project
set PYTHONIOENCODING=utf-8
:loop
if "%~1"=="" goto done
echo ==== %date% %time% %1 ==== >> output\usdata\task_runs.log
C:\Users\PC\AppData\Local\Programs\Python\Python312\python.exe -u tools\%1 >> output\usdata\task_runs.log 2>&1
shift
goto loop
:done
