# ANPR Station 


### Making executable
To make exe of this project use this command:

`pyinstaller --noconfirm --onedir --console --icon "D:/ANPR/ANPR/bullet-camera.ico" --add-data "D:/ANPR/ANPR/Activator.dll;." --add-data "D:/ANPR/ANPR/activator.o;." --add-data "D:/ANPR/ANPR/bullet-camera.ico;." --add-data "D:/ANPR/ANPR/config.ini;." --add-data "D:/ANPR/ANPR/database.txt;." --add-data "D:/ANPR/ANPR/IEShims.dll;." --add-data "D:/ANPR/ANPR/libActivator.a;." --add-data "D:/ANPR/ANPR/libgcc_s_seh-1.dll;." --add-data "D:/ANPR/ANPR/libstdc++-6.dll;." --add-data "D:/ANPR/ANPR/libwinpthread-1.dll;." --add-data "D:/ANPR/ANPR/winsockhc.dll;." --add-data "D:/ANPR/ANPR/weights;weights/" --hidden-import "configparser"  "D:/ANPR/ANPR/run.py"`


### Setup
After making exe to make a setup I used Inno setup software.
To integrate this software with the weighing software developed by Mr. Navidi I omited UI and I move LIVE displaying window to verbose mode. with the help of commented UI and using Inno setup software or any other software we can make setup file to install and run this software individually.
