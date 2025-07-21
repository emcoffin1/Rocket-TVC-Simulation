echo off
set LOCALHOST=%COMPUTERNAME%
set KILL_CMD="D:\Ansys\ANSYSI~1\v241\fluent/ntbin/win64/winkill.exe"

start "tell.exe" /B "D:\Ansys\ANSYSI~1\v241\fluent\ntbin\win64\tell.exe" DESKTOP-094SKA2 51252 CLEANUP_EXITING
timeout /t 1
"D:\Ansys\ANSYSI~1\v241\fluent\ntbin\win64\kill.exe" tell.exe
if /i "%LOCALHOST%"=="DESKTOP-094SKA2" (%KILL_CMD% 20772) 
if /i "%LOCALHOST%"=="DESKTOP-094SKA2" (%KILL_CMD% 21876) 
if /i "%LOCALHOST%"=="DESKTOP-094SKA2" (%KILL_CMD% 19816) 
if /i "%LOCALHOST%"=="DESKTOP-094SKA2" (%KILL_CMD% 18180) 
if /i "%LOCALHOST%"=="DESKTOP-094SKA2" (%KILL_CMD% 13328) 
if /i "%LOCALHOST%"=="DESKTOP-094SKA2" (%KILL_CMD% 20716) 
if /i "%LOCALHOST%"=="DESKTOP-094SKA2" (%KILL_CMD% 17192) 
if /i "%LOCALHOST%"=="DESKTOP-094SKA2" (%KILL_CMD% 3964)
del "C:\Users\emcof\PycharmProjects\ControlTheorySimulation\cleanup-fluent-DESKTOP-094SKA2-17192.bat"
