# Start a Claude Code session with PyCLM on a microscope PC that several people
# share. Each person gets a profile: their own Claude sign-in, settings, session
# history and plugin install (CLAUDE_CONFIG_DIR), and their name for the PyCLM
# start-of-session note (PYCLM_PERSON). Then a project folder, then a new,
# continued or resumed session.
#
# It runs the Claude Code command line (claude.exe), not the desktop app.
# Windows PowerShell 5.1 or later. Lives in <lab context>\launcher\.
# Machine settings, all optional: launcher\local.json (see local.example.json).

$ErrorActionPreference = 'Stop'
$LauncherDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$LabRoot = Split-Path -Parent $LauncherDir

function Ask-Choice([string]$Prompt, [string[]]$Valid) {
    while ($true) {
        $a = (Read-Host $Prompt).Trim().ToLower()
        if ($Valid -contains $a) { return $a }
        Write-Host "  Please type one of: $($Valid -join ', ')"
    }
}

function Num-Range([int]$n) {
    if ($n -gt 0) { return @(1..$n | ForEach-Object { "$_" }) }
    return @()
}

function Setting($obj, [string]$key, $default) {
    if ($null -ne $obj -and $obj.PSObject.Properties.Name -contains $key -and $obj.$key) {
        return $obj.$key
    }
    return $default
}

function Safe-Name([string]$s) {
    return (($s.Trim().ToLower()) -replace '[^a-z0-9._-]+', '-').Trim('-')
}

# --- machine settings ---------------------------------------------------------
$local = $null
$localPath = Join-Path $LauncherDir 'local.json'
if (Test-Path $localPath) { $local = Get-Content $localPath -Raw | ConvertFrom-Json }

$claudeExe = Setting $local 'claude_exe' $null
if (-not $claudeExe) {
    $cmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($cmd) { $claudeExe = $cmd.Source }
    else { $claudeExe = Join-Path $env:USERPROFILE '.local\bin\claude.exe' }
}
if (-not (Test-Path $claudeExe)) {
    Write-Host "Claude Code was not found at $claudeExe."
    Write-Host "Install it (https://code.claude.com), or set claude_exe in $localPath."
    exit 1
}
$profilesDir = Setting $local 'profiles_dir' (Join-Path $env:USERPROFILE '.pyclm\profiles')
# project folders: shared by everyone on this PC unless local.json says otherwise
$projectsDir = Setting $local 'projects_dir' (Join-Path $env:PUBLIC 'pyclm-projects')
$extraDirs = @(Setting $local 'extra_dirs' @())
$marketplace = Setting $local 'plugin_marketplace' 'Harrison-Oatman/PyCLM@plugin'
$plugin = Setting $local 'plugin' 'pyclm@pyclm-plugins'
# guests: one-off visitors; their profile is deleted when Claude closes
$allowGuest = [bool](Setting $local 'allow_guest' $false)

# --- bring the lab context up to date ------------------------------------------
if (Test-Path (Join-Path $LabRoot '.git')) {
    Write-Host "Updating the lab context..."
    Push-Location $LabRoot
    try { git pull --ff-only 2>&1 | Out-Host } catch { Write-Host "  git pull failed; using the copy on this PC." }
    Pop-Location
}

# --- is the microscope in use? (PyCLM's run lock, PyCLM 1.2 and later) ----------
$lockDir = $env:PYCLM_LOCK_DIR
if (-not $lockDir) { $lockDir = Join-Path $env:ProgramData 'pyclm' }
$lockFile = Join-Path $lockDir 'microscope.lock'
if (Test-Path $lockFile) {
    try {
        $lock = Get-Content $lockFile -Raw | ConvertFrom-Json
        $alive = $false
        if ($lock.pid) { $alive = [bool](Get-Process -Id $lock.pid -ErrorAction SilentlyContinue) }
        if ($alive) {
            Write-Host ""
            Write-Host "MICROSCOPE IN USE: a run of $($lock.directory), started $($lock.started) by $($lock.user)." -ForegroundColor Yellow
            Write-Host "You can plan, design or look at data; a second run will refuse to start."
        }
    } catch { }
}

# --- who are you ----------------------------------------------------------------
if (-not (Test-Path $profilesDir)) { New-Item -ItemType Directory -Force -Path $profilesDir | Out-Null }
# guest profiles left behind by a window that was closed early
Get-ChildItem $profilesDir -Directory -Filter '_guest-*' |
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddHours(-12) } |
    ForEach-Object { Remove-Item -Recurse -Force $_.FullName -ErrorAction SilentlyContinue }
$profiles = @(Get-ChildItem $profilesDir -Directory | Where-Object {
    $_.Name -notlike '_guest-*' -and (Test-Path (Join-Path $_.FullName 'pyclm_profile.json')) })

Write-Host ""
Write-Host "Who are you?"
for ($i = 0; $i -lt $profiles.Count; $i++) {
    $p = Get-Content (Join-Path $profiles[$i].FullName 'pyclm_profile.json') -Raw | ConvertFrom-Json
    Write-Host "  $($i + 1). $($p.name)"
}
Write-Host "  a. Add me"
$valid = @('a') + (Num-Range $profiles.Count)
if ($allowGuest) {
    Write-Host "  g. Guest (sign in for this session only; nothing about you is kept)"
    $valid += 'g'
}
$c = Ask-Choice "Choice" $valid

$newProfile = $false
$guest = $false
if ($c -eq 'g') {
    $name = (Read-Host "Your name (for this session)").Trim()
    if (-not $name) { $name = 'guest' }
    $pdir = Join-Path $profilesDir ("_guest-" + (Get-Date -Format 'yyyyMMdd-HHmmss'))
    New-Item -ItemType Directory -Force -Path $pdir | Out-Null
    $labSettings = Join-Path $LauncherDir 'profile_settings.json'
    if (Test-Path $labSettings) { Copy-Item $labSettings (Join-Path $pdir 'settings.json') }
    @{ name = "$name (guest)"; git_name = ''; git_email = '' } | ConvertTo-Json |
        Set-Content -Encoding UTF8 (Join-Path $pdir 'pyclm_profile.json')
    $guest = $true
} elseif ($c -eq 'a') {
    $name = (Read-Host "Your preferred username").Trim()
    $short = Safe-Name $name
    if (-not $short) { Write-Host "A name is needed."; exit 1 }
    $gitName = (Read-Host "Name for git commits (Enter to skip)").Trim()
    $gitEmail = ''
    if ($gitName) { $gitEmail = (Read-Host "Email for git commits").Trim() }
    $pdir = Join-Path $profilesDir $short
    New-Item -ItemType Directory -Force -Path $pdir | Out-Null
    # A lab may ship its own Claude Code settings for everyone (permissions,
    # environment); the plugin ships none.
    $labSettings = Join-Path $LauncherDir 'profile_settings.json'
    if (Test-Path $labSettings) { Copy-Item $labSettings (Join-Path $pdir 'settings.json') }
    @{ name = $name; git_name = $gitName; git_email = $gitEmail } | ConvertTo-Json |
        Set-Content -Encoding UTF8 (Join-Path $pdir 'pyclm_profile.json')
    $newProfile = $true
} else {
    $pdir = $profiles[[int]$c - 1].FullName
}
$prof = Get-Content (Join-Path $pdir 'pyclm_profile.json') -Raw | ConvertFrom-Json

$env:CLAUDE_CONFIG_DIR = $pdir
$env:PYCLM_PERSON = $prof.name
$env:PYCLM_LAB_CONTEXT = $LabRoot
if ($prof.git_name) {
    $env:GIT_AUTHOR_NAME = $prof.git_name; $env:GIT_COMMITTER_NAME = $prof.git_name
    $env:GIT_AUTHOR_EMAIL = $prof.git_email; $env:GIT_COMMITTER_EMAIL = $prof.git_email
}

# --- the plugin, once per profile -------------------------------------------------
$installed = Join-Path $pdir 'plugins\installed_plugins.json'
$hasPlugin = (Test-Path $installed) -and ((Get-Content $installed -Raw) -match [regex]::Escape($plugin))
if (-not $hasPlugin) {
    Write-Host ""
    Write-Host "Installing the PyCLM plugin for $($prof.name)..."
    & $claudeExe plugin marketplace add $marketplace 2>&1 | Out-Host
    & $claudeExe plugin install $plugin 2>&1 | Out-Host
} elseif ((Read-Host "Update the PyCLM plugin first? (y, or Enter to skip)").Trim().ToLower() -eq 'y') {
    # offline, the installed copy is kept
    Write-Host "Updating the PyCLM plugin..."
    try {
        & $claudeExe plugin marketplace update ($plugin -split '@')[1] 2>&1 | Out-Null
        & $claudeExe plugin update $plugin 2>&1 | Out-Host
    } catch { Write-Host "  update failed; using the installed plugin." }
}
if ($newProfile -or $guest) {
    Write-Host ""
    Write-Host "Profile made. Claude will ask you to sign in with your own account the first time."
}

# --- project ------------------------------------------------------------------------
if (-not (Test-Path $projectsDir)) { New-Item -ItemType Directory -Force -Path $projectsDir | Out-Null }
$projects = @(Get-ChildItem $projectsDir -Directory | Where-Object { $_.Name -notlike '_*' })
Write-Host ""
Write-Host "Which project?"
for ($i = 0; $i -lt $projects.Count; $i++) { Write-Host "  $($i + 1). $($projects[$i].Name)" }
Write-Host "  n. New project"
Write-Host "  s. No project (setup, a question, looking around)"
$c = Ask-Choice "Choice" (@('n', 's') + (Num-Range $projects.Count))
if ($c -eq 'n') {
    $pname = Safe-Name (Read-Host "Project name")
    if (-not $pname) { Write-Host "A name is needed."; exit 1 }
    $projDir = Join-Path $projectsDir $pname
    New-Item -ItemType Directory -Force -Path $projDir | Out-Null
    $pointer = Join-Path $LabRoot 'projects\_template\pyclm_project.md'
    if (Test-Path $pointer) { Copy-Item $pointer (Join-Path $projDir 'pyclm_project.md') }
} elseif ($c -eq 's') {
    $projDir = Join-Path $profilesDir "$(Split-Path -Leaf $pdir)-workspace"
    New-Item -ItemType Directory -Force -Path $projDir | Out-Null
} else {
    $projDir = $projects[[int]$c - 1].FullName
}

# --- session --------------------------------------------------------------------------
Write-Host ""
Write-Host "Session:"
Write-Host "  n. New session"
$valid = @('n')
if (-not $guest) {
    Write-Host "  c. Continue my last session here"
    Write-Host "  r. Pick one of my earlier sessions here"
    $valid += @('c', 'r')
}
$c = Ask-Choice "Choice" $valid

$claudeArgs = @('--add-dir', $LabRoot)
foreach ($d in $extraDirs) { if ($d) { $claudeArgs += @('--add-dir', $d) } }
if ($c -eq 'c') { $claudeArgs += '--continue' }
if ($c -eq 'r') { $claudeArgs += '--resume' }

Write-Host ""
Write-Host "Starting Claude Code for $($prof.name) in $projDir"
Set-Location $projDir
& $claudeExe @claudeArgs
$code = $LASTEXITCODE
if ($guest) {
    # the guest's sign-in, settings and session history go with the profile;
    # data and files written in the project folder stay
    Set-Location $env:TEMP
    Remove-Item -Recurse -Force $pdir -ErrorAction SilentlyContinue
    Write-Host "Guest profile removed. Your files in $projDir are still there."
}
exit $code
