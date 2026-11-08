param(
    [switch]$NoBrowser,
    [switch]$NoPipeline
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$CvApi = Join-Path $Root "cv_api"
$Frontend = Join-Path $Root "frontend"
$Python = Join-Path $CvApi ".venv\Scripts\python.exe"
$Mosquitto = "C:\Program Files\mosquitto\mosquitto.exe"
$MosquittoConfig = Join-Path $CvApi "config\mosquitto.conf"
$PipelineConfig = Join-Path $CvApi "config\pipeline.yaml"
$LogDir = Join-Path $Root "runtime-logs"
$StartedProcesses = New-Object System.Collections.Generic.List[System.Diagnostics.Process]

New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

function Test-PortListening {
    param([int]$Port)

    $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    return $null -ne $connection
}

function Wait-ForPort {
    param(
        [int]$Port,
        [string]$Name,
        [int]$TimeoutSeconds = 30
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-PortListening $Port) {
            Write-Host "$Name is listening on port $Port." -ForegroundColor Green
            return
        }
        Start-Sleep -Seconds 1
    }
    throw "$Name did not start listening on port $Port."
}

function Start-ManagedProcess {
    param(
        [string]$Name,
        [string]$FilePath,
        [string[]]$Arguments,
        [string]$WorkingDirectory
    )

    $safeName = $Name.ToLowerInvariant() -replace "[^a-z0-9]+", "-"
    $stdout = Join-Path $LogDir "$safeName.out.log"
    $stderr = Join-Path $LogDir "$safeName.err.log"
    Write-Host "Starting $Name..." -ForegroundColor Cyan
    $process = Start-Process `
        -FilePath $FilePath `
        -ArgumentList $Arguments `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru
    $StartedProcesses.Add($process)
    return $process
}

function Stop-StartedProcesses {
    for ($index = $StartedProcesses.Count - 1; $index -ge 0; $index--) {
        $process = $StartedProcesses[$index]
        if ($null -ne $process -and -not $process.HasExited) {
            Write-Host "Stopping process $($process.Id)..." -ForegroundColor Yellow
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
    }
}

try {
    if (-not (Test-Path $Python)) {
        throw "Python virtual environment not found: $Python"
    }
    if (-not (Test-Path (Join-Path $Frontend "node_modules"))) {
        throw "Frontend dependencies are missing. Run 'npm install' in $Frontend."
    }
    if (-not (Test-Path $PipelineConfig)) {
        throw "Pipeline configuration not found: $PipelineConfig"
    }
    if (-not (Test-Path $Mosquitto)) {
        throw "Mosquitto not found: $Mosquitto"
    }

    if (Test-PortListening 1883) {
        Write-Host "Mosquitto is already running on port 1883." -ForegroundColor Green
    } else {
        $mosquittoArgs = @("-c", $MosquittoConfig, "-v")
        Start-ManagedProcess "Mosquitto" $Mosquitto $mosquittoArgs $CvApi
        Wait-ForPort 1883 "Mosquitto"
    }

    if (Test-PortListening 8000) {
        Write-Host "FastAPI is already running on port 8000." -ForegroundColor Green
    } else {
        $apiArgs = @(
            "-m", "uvicorn", "src.dashboard.api:app",
            "--host", "0.0.0.0", "--port", "8000"
        )
        Start-ManagedProcess "FastAPI" $Python $apiArgs $CvApi
        Wait-ForPort 8000 "FastAPI"
        $health = Invoke-WebRequest `
            -Uri "http://127.0.0.1:8000/health" `
            -UseBasicParsing `
            -TimeoutSec 10
        if ($health.StatusCode -ne 200) {
            throw "FastAPI health check returned HTTP $($health.StatusCode)."
        }
        Write-Host "FastAPI health check passed." -ForegroundColor Green
    }

    if (Test-PortListening 5173) {
        Write-Host "Vite is already running on port 5173." -ForegroundColor Green
    } else {
        $npm = (Get-Command npm.cmd -ErrorAction SilentlyContinue).Source
        if (-not $npm) {
            throw "npm.cmd was not found on PATH."
        }
        Start-ManagedProcess "Frontend" $npm @("run", "dev", "--", "--host", "127.0.0.1") $Frontend
        Wait-ForPort 5173 "Frontend"
    }

    if (-not $NoBrowser) {
        Start-Process "http://127.0.0.1:5173"
    }

    if (-not $NoPipeline) {
        $pipelineExists = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
            Where-Object {
                $_.CommandLine -and
                $_.CommandLine -like "*src.analytics.pipeline*" -and
                $_.CommandLine -like "*config\pipeline.yaml*"
            }
        if ($pipelineExists) {
            Write-Host "The configured CV pipeline is already running." -ForegroundColor Green
        } else {
            $pipelineArgs = @(
                "-m", "src.analytics.pipeline",
                "--config", ".\config\pipeline.yaml"
            )
            Start-ManagedProcess "CV Pipeline" $Python $pipelineArgs $CvApi
        }
    }

    Write-Host ""
    Write-Host "Suvirfy is running:" -ForegroundColor Green
    Write-Host "  Dashboard: http://127.0.0.1:5173"
    Write-Host "  API:       http://127.0.0.1:8000/health"
    Write-Host "  MQTT:      localhost:1883"
    Write-Host ""
    Write-Host "Press Ctrl+C to stop processes started by this script." -ForegroundColor Yellow

    while ($true) {
        Start-Sleep -Seconds 2
        foreach ($process in $StartedProcesses) {
            if ($process.HasExited -and $process.ExitCode -ne 0) {
                Write-Warning "A managed process exited with code $($process.ExitCode). Check $LogDir."
            }
        }
    }
} finally {
    Stop-StartedProcesses
}
