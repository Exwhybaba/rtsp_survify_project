[CmdletBinding()]
param(
    [int]$MqttPort = 1883,
    [int]$MqttWebSocketPort = 9001,
    [int]$EvidencePort = 8001,
    [int]$ViewerPort = 8080,
    [string]$PythonPath = "",
    [string]$MosquittoPath = "",
    [string]$ConfigPath = "",
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

$LogDir = Join-Path $ProjectRoot "logs"
$EvidenceDir = Join-Path $ProjectRoot "evidence"
$BrokerConfig = Join-Path $LogDir "mosquitto-local.conf"
$ViewerFile = Join-Path $ProjectRoot "tools\viewer.html"

$ResolvedConfig = if ($ConfigPath) {
    if ([System.IO.Path]::IsPathRooted($ConfigPath)) {
        (Resolve-Path -LiteralPath $ConfigPath).Path
    } else {
        (Resolve-Path -LiteralPath (Join-Path $ProjectRoot $ConfigPath)).Path
    }
} else {
    Join-Path $ProjectRoot "config\pipeline.yaml"
}

if (-not (Test-Path -LiteralPath $ResolvedConfig -PathType Leaf)) {
    throw "Pipeline configuration not found: $ResolvedConfig"
}

if (-not (Test-Path -LiteralPath $ViewerFile -PathType Leaf)) {
    throw "Viewer file not found: $ViewerFile"
}

New-Item -ItemType Directory -Force -Path $LogDir, $EvidenceDir | Out-Null

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

function Test-TcpPort {
    param([int]$Port)

    $Client = [System.Net.Sockets.TcpClient]::new()

    try {
        $Connect = $Client.BeginConnect("127.0.0.1", $Port, $null, $null)
        $Connected = $Connect.AsyncWaitHandle.WaitOne(500)

        if (-not $Connected) {
            return $false
        }

        $Client.EndConnect($Connect)
        return $true
    } catch {
        return $false
    } finally {
        $Client.Close()
    }
}

function Resolve-Executable {
    param(
        [string]$RequestedPath,
        [string]$DefaultPath,
        [string]$DisplayName
    )

    if ($RequestedPath) {
        if (-not (Test-Path -LiteralPath $RequestedPath -PathType Leaf)) {
            throw "$DisplayName executable not found: $RequestedPath"
        }

        return (Resolve-Path -LiteralPath $RequestedPath).Path
    }

    if (Test-Path -LiteralPath $DefaultPath -PathType Leaf) {
        return (Resolve-Path -LiteralPath $DefaultPath).Path
    }

    $Command = Get-Command $DisplayName -ErrorAction SilentlyContinue

    if ($null -eq $Command) {
        throw "Could not find $DisplayName. Provide its path."
    }

    return $Command.Source
}

function Start-LoggedProcess {
    param(
        [string]$FilePath,
        [string[]]$ArgumentList,
        [string]$WorkingDirectory,
        [string]$OutputPath,
        [string]$ErrorPath
    )

    return Start-Process `
        -FilePath $FilePath `
        -ArgumentList $ArgumentList `
        -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $OutputPath `
        -RedirectStandardError $ErrorPath `
        -PassThru
}

# --------------------------------------------------
# VALIDATE PORTS
# --------------------------------------------------

$Ports = @(
    $MqttPort,
    $MqttWebSocketPort,
    $EvidencePort,
    $ViewerPort
)

if (($Ports | Sort-Object -Unique).Count -ne $Ports.Count) {
    throw "All four ports must be different."
}

foreach ($Port in $Ports) {
    if ($Port -lt 1 -or $Port -gt 65535) {
        throw "Invalid port: $Port"
    }
}

# --------------------------------------------------
# DETECT EXISTING MOSQUITTO BROKER
# --------------------------------------------------

Write-Host "`nChecking Mosquitto broker..." -ForegroundColor Cyan

$MqttAvailable = Test-TcpPort -Port $MqttPort
$WebSocketAvailable = Test-TcpPort -Port $MqttWebSocketPort

$UseExistingBroker = $false
$BrokerProcess = $null

if ($MqttAvailable -and $WebSocketAvailable) {

    $UseExistingBroker = $true

    Write-Host "Existing Mosquitto broker detected." -ForegroundColor Green
    Write-Host "Reusing MQTT port $MqttPort."
    Write-Host "Reusing WebSocket port $MqttWebSocketPort."

} elseif ($MqttAvailable -or $WebSocketAvailable) {

    throw @"
A partial Mosquitto configuration was detected.

MQTT port ${MqttPort}: $MqttAvailable
WebSocket port ${MqttWebSocketPort}: $WebSocketAvailable

Both ports must be available before starting Suvirfy.
Check your Mosquitto configuration and Windows service.
"@

} else {

    Write-Host "No existing broker detected. Starting Mosquitto..." -ForegroundColor Yellow

    $Mosquitto = Resolve-Executable `
        -RequestedPath $MosquittoPath `
        -DefaultPath (Join-Path $ProjectRoot "mosquitto.exe") `
        -DisplayName "mosquitto"

    @"
allow_anonymous true
persistence false
log_type error
log_type warning
log_type notice

listener $MqttPort 127.0.0.1
protocol mqtt

listener $MqttWebSocketPort 127.0.0.1
protocol websockets
"@ | Set-Content -LiteralPath $BrokerConfig -Encoding ascii
}

# --------------------------------------------------
# CHECK EVIDENCE AND VIEWER PORTS
# --------------------------------------------------

foreach ($Port in @($EvidencePort, $ViewerPort)) {
    if (Test-TcpPort -Port $Port) {
        throw "Port $Port is already in use. Stop the application using it or choose another port."
    }
}

# --------------------------------------------------
# START APPLICATION SERVICES
# --------------------------------------------------

$Python = Resolve-Executable `
    -RequestedPath $PythonPath `
    -DefaultPath (Join-Path $ProjectRoot ".venv\Scripts\python.exe") `
    -DisplayName "python"

$StartedProcesses = [System.Collections.Generic.List[System.Diagnostics.Process]]::new()

try {

    # ----------------------------------------------
    # 1. MOSQUITTO
    # ----------------------------------------------

    if (-not $UseExistingBroker) {

        Write-Host "`n[1/4] Starting Mosquitto..." -ForegroundColor Yellow

        $BrokerProcess = Start-LoggedProcess `
            -FilePath $Mosquitto `
            -ArgumentList @("-c", "`"$BrokerConfig`"") `
            -WorkingDirectory $ProjectRoot `
            -OutputPath (Join-Path $LogDir "mosquitto.stdout.log") `
            -ErrorPath (Join-Path $LogDir "mosquitto.stderr.log")

        $StartedProcesses.Add($BrokerProcess)

        $BrokerReady = $false

        for ($i = 0; $i -lt 20; $i++) {
            Start-Sleep -Milliseconds 500

            if ($BrokerProcess.HasExited) {
                throw "Mosquitto failed. Check logs\mosquitto.stderr.log."
            }

            if (
                (Test-TcpPort -Port $MqttPort) -and
                (Test-TcpPort -Port $MqttWebSocketPort)
            ) {
                $BrokerReady = $true
                break
            }
        }

        if (-not $BrokerReady) {
            throw "Mosquitto started but its ports did not become available."
        }

        Write-Host "Mosquitto started successfully." -ForegroundColor Green

    } else {

        Write-Host "`n[1/4] Reusing existing Mosquitto service." -ForegroundColor Green
    }

    # ----------------------------------------------
    # 2. EVIDENCE SERVER
    # ----------------------------------------------

    Write-Host "`n[2/4] Starting evidence server..." -ForegroundColor Yellow

    $EvidenceProcess = Start-LoggedProcess `
        -FilePath $Python `
        -ArgumentList @(
            "-m",
            "http.server",
            "$EvidencePort",
            "--directory",
            "`"$EvidenceDir`""
        ) `
        -WorkingDirectory $ProjectRoot `
        -OutputPath (Join-Path $LogDir "evidence-server.stdout.log") `
        -ErrorPath (Join-Path $LogDir "evidence-server.stderr.log")

    $StartedProcesses.Add($EvidenceProcess)

    Start-Sleep -Milliseconds 700

    if ($EvidenceProcess.HasExited) {
        throw "Evidence server failed. Check logs\evidence-server.stderr.log."
    }

    Write-Host "Evidence server: http://localhost:$EvidencePort/"

    # ----------------------------------------------
    # 3. VIEWER SERVER
    # ----------------------------------------------

    Write-Host "`n[3/4] Starting viewer server..." -ForegroundColor Yellow

    $ViewerProcess = Start-LoggedProcess `
        -FilePath $Python `
        -ArgumentList @(
            "-m",
            "http.server",
            "$ViewerPort",
            "--directory",
            "`"$ProjectRoot`""
        ) `
        -WorkingDirectory $ProjectRoot `
        -OutputPath (Join-Path $LogDir "viewer-server.stdout.log") `
        -ErrorPath (Join-Path $LogDir "viewer-server.stderr.log")

    $StartedProcesses.Add($ViewerProcess)

    Start-Sleep -Milliseconds 700

    if ($ViewerProcess.HasExited) {
        throw "Viewer server failed. Check logs\viewer-server.stderr.log."
    }

    $ViewerUrl = "http://localhost:$ViewerPort/tools/viewer.html?broker=ws://localhost:$MqttWebSocketPort&evidence=http://localhost:$EvidencePort"

    Write-Host "Viewer server started."

    # ----------------------------------------------
    # 4. ANALYTICS PIPELINE
    # ----------------------------------------------

    Write-Host "`n[4/4] Starting analytics pipeline..." -ForegroundColor Yellow

    $OldMqttPort = $env:MQTT_PORT
    $env:MQTT_PORT = "$MqttPort"

    try {
        $PipelineProcess = Start-LoggedProcess `
            -FilePath $Python `
            -ArgumentList @(
                "-m",
                "src.analytics.pipeline",
                "--config",
                "`"$ResolvedConfig`""
            ) `
            -WorkingDirectory $ProjectRoot `
            -OutputPath (Join-Path $LogDir "pipeline.stdout.log") `
            -ErrorPath (Join-Path $LogDir "pipeline.stderr.log")
    } finally {
        $env:MQTT_PORT = $OldMqttPort
    }

    $StartedProcesses.Add($PipelineProcess)

    Start-Sleep -Seconds 2

    if ($PipelineProcess.HasExited) {
        Write-Warning "Pipeline exited shortly after launch. Check logs\pipeline.stderr.log."
    } else {
        Write-Host "Analytics pipeline started." -ForegroundColor Green
    }

    # ----------------------------------------------
    # OPEN VIEWER
    # ----------------------------------------------

    if (-not $NoBrowser) {
        Start-Process $ViewerUrl | Out-Null
    }

    # ----------------------------------------------
    # SUMMARY
    # ----------------------------------------------

    Write-Host "`n========================================" -ForegroundColor Green
    Write-Host "       SU VIRFY STARTUP COMPLETE" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green

    Write-Host "`nViewer:          $ViewerUrl"
    Write-Host "Evidence:        http://localhost:$EvidencePort/"
    Write-Host "MQTT:            localhost:$MqttPort"
    Write-Host "MQTT WebSocket:  ws://localhost:$MqttWebSocketPort"
    Write-Host "Logs:            $LogDir"

    if ($StartedProcesses.Count -gt 0) {
        $ProcessIds = ($StartedProcesses | ForEach-Object { $_.Id }) -join ","
        Write-Host "`nProcesses started by this script: $ProcessIds"
        Write-Host "Stop them with: Get-Process -Id $ProcessIds | Stop-Process"
    }

    if ($UseExistingBroker) {
        Write-Host "Mosquitto was reused. It was not started or stopped by this script."
    }

} catch {

    Write-Host "`nStartup failed. Cleaning up processes started by this script..." -ForegroundColor Red

    foreach ($Process in $StartedProcesses) {
        if (-not $Process.HasExited) {
            Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        }
    }

    throw
}