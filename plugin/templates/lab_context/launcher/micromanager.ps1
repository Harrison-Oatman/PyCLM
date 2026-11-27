# Start the Micro-Manager GUI on this microscope PC, with any steps it needs
# first (a controller to switch on, a device server, an environment
# variable). Refuses while a PyCLM run holds the microscope, because the GUI
# and a run cannot both hold the hardware.
#
# Settings: "micromanager" in launcher\local.json (see local.example.json):
# one entry, or a list of entries with a "name" each (for example one per
# .cfg or per camera); a menu is shown when there is more than one.
# Windows PowerShell 5.1 or later.

$ErrorActionPreference = 'Stop'
$LauncherDir = Split-Path -Parent $MyInvocation.MyCommand.Path

$local = $null
$localPath = Join-Path $LauncherDir 'local.json'
if (Test-Path $localPath) { $local = Get-Content $localPath -Raw | ConvertFrom-Json }

$entries = @()
if ($null -ne $local -and $local.PSObject.Properties.Name -contains 'micromanager') {
    $entries = @($local.micromanager)
}
if ($entries.Count -eq 0) {
    $entries = @([pscustomobject]@{ name = 'Micro-Manager'; dir = 'C:\Program Files\Micro-Manager-2.0' })
}

# --- not while a PyCLM run holds the microscope ---------------------------------
$lockDir = $env:PYCLM_LOCK_DIR
if (-not $lockDir) { $lockDir = Join-Path $env:ProgramData 'pyclm' }
$lockFile = Join-Path $lockDir 'microscope.lock'
if (Test-Path $lockFile) {
    try {
        $lock = Get-Content $lockFile -Raw | ConvertFrom-Json
        if ($lock.pid -and (Get-Process -Id $lock.pid -ErrorAction SilentlyContinue)) {
            Write-Host ""
            Write-Host "A PyCLM run is using the microscope: $($lock.directory), started $($lock.started) by $($lock.user)." -ForegroundColor Yellow
            Write-Host "Opening Micro-Manager now can stop that run or the GUI from reaching the hardware."
            $a = (Read-Host "Type 'open anyway' to continue, or Enter to stop").Trim().ToLower()
            if ($a -ne 'open anyway') { exit 0 }
        }
    } catch { }
}

# --- which one ---------------------------------------------------------------------
$e = $entries[0]
if ($entries.Count -gt 1) {
    Write-Host ""
    Write-Host "Which Micro-Manager?"
    for ($i = 0; $i -lt $entries.Count; $i++) { Write-Host "  $($i + 1). $($entries[$i].name)" }
    while ($true) {
        $c = (Read-Host "Choice").Trim()
        if ($c -match '^\d+$' -and [int]$c -ge 1 -and [int]$c -le $entries.Count) { $e = $entries[[int]$c - 1]; break }
        Write-Host "  Please type a number from 1 to $($entries.Count)."
    }
}

$mmDir = $e.dir
if (-not $mmDir) { $mmDir = 'C:\Program Files\Micro-Manager-2.0' }
$exe = $e.exe
if (-not $exe) { $exe = 'ImageJ.exe' }
if (-not [System.IO.Path]::IsPathRooted($exe)) { $exe = Join-Path $mmDir $exe }
if (-not (Test-Path $exe)) {
    Write-Host "Micro-Manager was not found at $exe. Set micromanager.dir (and exe) in $localPath."
    exit 1
}

# --- steps before the GUI ------------------------------------------------------------
if ($e.PSObject.Properties.Name -contains 'env' -and $e.env) {
    foreach ($p in $e.env.PSObject.Properties) { Set-Item -Path "env:$($p.Name)" -Value $p.Value }
}
foreach ($step in @($e.before)) {
    if (-not $step) { continue }
    Write-Host "> $step"
    & cmd /c $step
    if ($LASTEXITCODE -ne 0) {
        Write-Host "That step failed (exit $LASTEXITCODE); Micro-Manager was not started." -ForegroundColor Yellow
        exit 1
    }
}
if ($e.wait_seconds) { Start-Sleep -Seconds ([int]$e.wait_seconds) }

# --- the GUI ---------------------------------------------------------------------------
Write-Host "Starting $($e.name) ($exe)"
if ($e.note) { Write-Host $e.note }
Write-Host "Close Micro-Manager before a PyCLM run starts; the two cannot share the hardware."
$args_ = @($e.args | Where-Object { $_ })
if ($args_.Count -gt 0) { Start-Process -FilePath $exe -WorkingDirectory $mmDir -ArgumentList $args_ }
else { Start-Process -FilePath $exe -WorkingDirectory $mmDir }
