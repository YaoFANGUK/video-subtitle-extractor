; Video Subtitle Extractor - Inno Setup 安装脚本
; 版本: 2.2.0

#define MyAppName "Video Subtitle Extractor"
#define MyAppVersion "2.2.0"
#define MyAppPublisher "YaoFANGUK"
#define MyAppURL "https://github.com/YaoFANGUK/video-subtitle-extractor"
#define MyAppExeName "VideoSubtitleExtractor.exe"
#define MyAppCopyright "Copyright (C) 2021-2026 YaoFANGUK"

[Setup]
; 应用信息
AppId={{B8D7C8A1-3E5F-4A9D-8C2B-7F1A6E0D9C3A}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
AllowNoIcons=yes
LicenseFile={#SourcePath}\LICENSE
OutputDir={#SourcePath}\installer_output
OutputBaseFilename=VSE_Setup_{#MyAppVersion}
SetupIconFile={#SourcePath}\design\vse.ico
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
WizardSizePercent=100

; 权限与兼容性
PrivilegesRequired=admin
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; 卸载
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}

; 安装界面
; 禁用以管理员运行安装程序时的 UAC 提示（非必要）
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "chinesesimplified"; MessagesFile: "compiler:Languages\ChineseSimplified.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; 复制整个 dist 目录内容
Source: "{#SourcePath}\dist\VideoSubtitleExtractor\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\_internal\design\vse.ico"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\_internal\design\vse.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
Type: filesandordirs; Name: "{app}"
