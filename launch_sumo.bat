@echo off
title ResQFlow - SUMO Simulation Launcher
color 0B
echo.
echo  ============================================
echo    ResQFlow ML - SUMO Traffic Simulation
echo    Mysore City Road Network - SUMO-GUI
echo  ============================================
echo.
echo  Starting SUMO-GUI...
echo  Once open, click the green PLAY button to start the simulation.
echo  Then go to the web dashboard and click "Connect SUMO".
echo.

cd /d "%~dp0sumo\scenarios"

"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo-gui.exe" ^
    -c mysore.sumocfg ^
    --remote-port 8813 ^
    --start ^
    --quit-on-end false

echo.
echo  SUMO-GUI closed.
pause
