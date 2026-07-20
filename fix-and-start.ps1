# search-agent one-click fix & start
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " Search Agent - One-Click Fix & Start" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# -----------------------------------------------------------
# Step 0: Check Docker
# -----------------------------------------------------------
Write-Host "[Step 0/5] Checking Docker..." -ForegroundColor Yellow

$dockerCmd = "docker.exe"
$dockerInfo = & $dockerCmd info 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker not running. Starting Docker Desktop..." -ForegroundColor Yellow
    $ddPath = "${env:ProgramFiles}\Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $ddPath)) {
        $ddPath = "${env:ProgramFiles(x86)}\Docker\Docker\Docker Desktop.exe"
    }
    if (-not (Test-Path $ddPath)) {
        Write-Host "ERROR: Docker Desktop not found." -ForegroundColor Red
        exit 1
    }
    Start-Process $ddPath
    Write-Host "Waiting for Docker Desktop (max 120s)..." -ForegroundColor Yellow
    $waited = 0
    do {
        Start-Sleep -Seconds 5
        $waited += 5
        Write-Host "  waiting... ${waited}s" -ForegroundColor Gray
        & $dockerCmd info 2>$null | Out-Null
    } while ($LASTEXITCODE -ne 0 -and $waited -lt 120)
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Docker Desktop startup timed out." -ForegroundColor Red
        exit 1
    }
}
Write-Host "  OK: Docker is ready" -ForegroundColor Green
Write-Host ""

# -----------------------------------------------------------
# Step 1: Reset Docker gRPC state
# -----------------------------------------------------------
Write-Host "[Step 1/5] Resetting Docker Desktop gRPC state..." -ForegroundColor Yellow

# Shutdown WSL if present
$hasWsl = Get-Command "wsl.exe" -ErrorAction SilentlyContinue
if ($hasWsl) {
    Write-Host "  Shutting down WSL..." -ForegroundColor Gray
    wsl --shutdown 2>$null
    Start-Sleep -Seconds 3
}

# Clean Docker internal logs
$logDirs = @("$env:APPDATA\Docker\log", "$env:LOCALAPPDATA\Docker\log")
foreach ($dir in $logDirs) {
    if (Test-Path $dir) {
        Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        Write-Host "  Cleaned: $dir" -ForegroundColor Gray
    }
}

# Restart Docker Desktop
Write-Host "  Restarting Docker Desktop..." -ForegroundColor Gray
Stop-Process -Name "Docker Desktop" -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 5

$ddPath = "${env:ProgramFiles}\Docker\Docker\Docker Desktop.exe"
if (-not (Test-Path $ddPath)) {
    $ddPath = "${env:ProgramFiles(x86)}\Docker\Docker\Docker Desktop.exe"
}
if (Test-Path $ddPath) {
    Start-Process $ddPath
    Write-Host "  Waiting for Docker Desktop (max 90s)..." -ForegroundColor Gray
    $waited = 0
    do {
        Start-Sleep -Seconds 5
        $waited += 5
        Write-Host "    waiting... ${waited}s" -ForegroundColor Gray
        & $dockerCmd info 2>$null | Out-Null
    } while ($LASTEXITCODE -ne 0 -and $waited -lt 90)
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  OK: Docker Desktop restarted" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: Timed out, continuing anyway..." -ForegroundColor Yellow
    }
}
Write-Host ""

# -----------------------------------------------------------
# Step 2: Check .env
# -----------------------------------------------------------
Write-Host "[Step 2/5] Checking .env config..." -ForegroundColor Yellow

if (Test-Path ".env") {
    $envContent = Get-Content ".env" -Raw
    if (($envContent -match "sk-your-api-key") -or ($envContent -match "tvly-your-api-key")) {
        Write-Host "  WARNING: .env contains placeholder API keys!" -ForegroundColor Yellow
    } else {
        Write-Host "  OK: .env configured" -ForegroundColor Green
    }
} else {
    Write-Host "  ERROR: .env file not found!" -ForegroundColor Red
    Write-Host "  Create a .env file with LLM_API_KEY and TAVILY_API_KEY" -ForegroundColor Red
    exit 1
}
Write-Host ""

# -----------------------------------------------------------
# Step 3: Verify frontend files
# -----------------------------------------------------------
Write-Host "[Step 3/5] Verifying frontend files..." -ForegroundColor Yellow

$required = @(
    "frontend\lib\types.ts",
    "frontend\lib\api.ts",
    "frontend\providers\ChatProvider.tsx",
    "frontend\components\Sidebar.tsx",
    "frontend\components\ChatArea.tsx",
    "frontend\app\page.tsx",
    "frontend\app\layout.tsx",
    "frontend\app\globals.css",
    "frontend\next.config.ts",
    "frontend\package.json"
)

$allOk = $true
foreach ($f in $required) {
    if (-not (Test-Path $f)) {
        Write-Host "  MISSING: $f" -ForegroundColor Red
        $allOk = $false
    }
}
if ($allOk) {
    Write-Host "  OK: All frontend source files present" -ForegroundColor Green
} else {
    Write-Host "  ERROR: Missing frontend files." -ForegroundColor Red
    exit 1
}
Write-Host ""

# -----------------------------------------------------------
# Step 4: Clean old Docker resources
# -----------------------------------------------------------
Write-Host "[Step 4/5] Cleaning old Docker resources..." -ForegroundColor Yellow

& $dockerCmd compose down --remove-orphans 2>$null
& $dockerCmd rm -f search-backend 2>$null
& $dockerCmd rm -f search-frontend 2>$null

Write-Host "  OK: Old resources cleaned" -ForegroundColor Green
Write-Host ""

# -----------------------------------------------------------
# Step 5: Build and start (with BuildKit disabled to avoid gRPC bug)
# -----------------------------------------------------------
Write-Host "[Step 5/5] Building images and starting services..." -ForegroundColor Yellow
Write-Host "  (BuildKit disabled to work around Docker Desktop gRPC bug)" -ForegroundColor Gray
Write-Host "  (This may take several minutes)" -ForegroundColor Gray
Write-Host ""

$env:DOCKER_BUILDKIT = "0"
& $dockerCmd compose up --build -d

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host " START FAILED! Check the error output above." -ForegroundColor Red
    Write-Host "============================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "  1. Ensure Docker Desktop uses WSL2 backend (Settings -> General)" -ForegroundColor White
    Write-Host "  2. Run: wsl --update (update WSL kernel)" -ForegroundColor White
    Write-Host "  3. Docker Desktop -> Troubleshoot -> Reset to factory defaults" -ForegroundColor White
    exit 1
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " START SUCCESS!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host " Backend API:  http://localhost:8000" -ForegroundColor White
Write-Host " Health check: http://localhost:8000/api/health" -ForegroundColor White
Write-Host " Frontend:     http://localhost:3000" -ForegroundColor White
Write-Host " API Docs:     http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host " View logs:    docker compose logs -f" -ForegroundColor Gray
Write-Host " Stop:         docker compose down" -ForegroundColor Gray
Write-Host ""
