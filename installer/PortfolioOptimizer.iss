[Setup]
AppName=Оптимизатор Портфеля
AppVersion=1.0
DefaultDirName={autopf}\PortfolioOptimizer
OutputDir=..\output              
OutputBaseFilename=PortfolioOptimizer_Setup
Compression=lzma
SolidCompression=yes

[Files]
Source: "app_files\PortfolioOptimizer.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{commondesktop}\Оптимизатор Портфеля"; Filename: "{app}\PortfolioOptimizer.exe"
Name: "{commonprograms}\Оптимизатор Портфеля"; Filename: "{app}\PortfolioOptimizer.exe"

[Run]
Filename: "{app}\PortfolioOptimizer.exe"; Description: "Запустить приложение"; Flags: postinstall nowait