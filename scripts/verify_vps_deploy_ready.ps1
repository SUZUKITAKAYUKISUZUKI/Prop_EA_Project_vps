# verify_vps_deploy_ready.ps1 — push 前に dev と _vps が一致しているか確認
#
#   .\scripts\verify_vps_deploy_ready.ps1
#   .\scripts\verify_vps_deploy_ready.ps1 -AfterSync

param(
    [switch]$AfterSync
)

$ErrorActionPreference = "Stop"
$SourceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TargetDir = Join-Path (Split-Path $SourceRoot -Parent) "Prop_EA_Project_vps"

$critical = @(
    "main_platform.py",
    "mt5_bridge.py",
    "bridge_runtime.py",
    "mt5_executor.py",
    "src\daemon\import_daemon.py",
    "src\daemon\daemon_runner.py",
    "mt5\PropEA_Bridge.mq5"
)

Write-Host "=== VPS deploy readiness check ===" -ForegroundColor Cyan
Write-Host "Dev : $SourceRoot"
Write-Host "VPS : $TargetDir"
Write-Host ""

if (-not (Test-Path $TargetDir)) {
    Write-Host "[FAIL] Prop_EA_Project_vps not found. Run sync_vps_min.cmd first." -ForegroundColor Red
    exit 1
}

$mismatches = @()
foreach ($rel in $critical) {
    $devPath = Join-Path $SourceRoot ($rel -replace '/', '\')
    $vpsPath = Join-Path $TargetDir ($rel -replace '/', '\')
    if (-not (Test-Path $devPath)) {
        Write-Host "[WARN] missing in dev: $rel" -ForegroundColor Yellow
        continue
    }
    if (-not (Test-Path $vpsPath)) {
        Write-Host "[FAIL] missing in _vps (run sync): $rel" -ForegroundColor Red
        $mismatches += $rel
        continue
    }
    $dh = (Get-FileHash $devPath -Algorithm SHA256).Hash
    $vh = (Get-FileHash $vpsPath -Algorithm SHA256).Hash
    if ($dh -ne $vh) {
        Write-Host "[MISMATCH] $rel" -ForegroundColor Red
        $mismatches += $rel
    } else {
        Write-Host "[OK] $rel"
    }
}

$marker = "LIVE_SETUP_MATCH_VERSION"
$devMp = Join-Path $SourceRoot "main_platform.py"
$vpsMp = Join-Path $TargetDir "main_platform.py"
if ((Test-Path $devMp) -and (Test-Path $vpsMp)) {
    $devLine = (Select-String -Path $devMp -Pattern $marker | Select-Object -First 1).Line
    $vpsLine = (Select-String -Path $vpsMp -Pattern $marker | Select-Object -First 1).Line
    Write-Host ""
    Write-Host "Live marker (dev): $($devLine.Trim())"
    Write-Host "Live marker (vps): $($vpsLine.Trim())"
}

Push-Location $TargetDir
try {
    $status = git status --porcelain 2>&1
    $branch = git status -sb 2>&1 | Select-Object -First 1
    $head = git log -1 --format="%h %ci %s" 2>&1
    Write-Host ""
    Write-Host "Git (_vps): $branch"
    Write-Host "HEAD: $head"
    if ($status) {
        Write-Host "Uncommitted changes:" -ForegroundColor Yellow
        $status | ForEach-Object { Write-Host "  $_" }
        Write-Host "[INFO] Commit + push these files, then git pull on VPS." -ForegroundColor Yellow
    } else {
        Write-Host "Uncommitted: (none)"
        if ($mismatches.Count -eq 0) {
            Write-Host "[INFO] Working tree clean — GitHub already has current _vps files." -ForegroundColor DarkGray
            Write-Host "       If dev was just edited, run sync_vps_min.cmd then re-run this script." -ForegroundColor DarkGray
        }
    }

    $ahead = git rev-list --count origin/main..HEAD 2>$null
    if ($ahead -and [int]$ahead -gt 0) {
        Write-Host "[WARN] $ahead commit(s) not pushed to origin/main — run git push" -ForegroundColor Yellow
    }
} finally {
    Pop-Location
}

Write-Host ""
if ($mismatches.Count -gt 0) {
    Write-Host "[FAIL] Dev and _vps differ. Save all files, run sync_vps_min.cmd, re-run this script." -ForegroundColor Red
    exit 1
}

if ($AfterSync -and -not $status) {
    Write-Host "[WARN] Sync done but _vps has nothing to commit — dev may not have been saved before sync." -ForegroundColor Yellow
    exit 2
}

Write-Host "[OK] Dev and _vps critical files match. Safe to commit/push (if git shows changes)." -ForegroundColor Green
exit 0
