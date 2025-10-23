param(
    [switch]$SmokeTest,
    [string]$Prompt,
    [string]$WSLDistro = "Ubuntu",
    [string]$WindowsEnvFile
)

$ErrorActionPreference = 'Stop'

function Write-Log {
    param([string]$Message)
    $timestamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
    Write-Host "[$timestamp] $Message"
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$rootDir = Split-Path -Parent $scriptDir
if (-not $WindowsEnvFile) {
    $WindowsEnvFile = Join-Path $rootDir '.env.windows'
}
$controllerPath = Join-Path $rootDir 'big_3_Windows.py'
if (-not (Test-Path $controllerPath)) {
    throw "Unable to locate big_3_Windows.py at $controllerPath"
}

function Import-DotEnv {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path $Path)) {
        Write-Log "No .env file present at $Path"
        return
    }
    Write-Log "Loading environment variables from $Path"
    Get-Content $Path | Where-Object { $_ -and ($_ -notmatch '^\s*#') } | ForEach-Object {
        if ($_ -match '^(?<key>[^=]+)=(?<value>.*)$') {
            $key = $Matches['key'].Trim()
            $value = $Matches['value']
            $Env:$key = $value
        }
    }
}

function Get-PythonCommand {
    if (Get-Command python.exe -ErrorAction SilentlyContinue) {
        return @{ Executable = 'python'; Args = @('-') }
    }
    if (Get-Command python3.exe -ErrorAction SilentlyContinue) {
        return @{ Executable = 'python3'; Args = @('-') }
    }
    if (Get-Command uv.exe -ErrorAction SilentlyContinue) {
        return @{ Executable = 'uv'; Args = @('run', 'python', '-') }
    }
    throw 'Unable to locate python executable. Install Python or uv.'
}

function Invoke-LoggingStartup {
    param([string]$Distro)
    $loggingDir = "/home/rawleysm/dev/claude-code-hooks-multi-agent-osier ability/scripts"
    $command = "cd '$loggingDir' && ./start-system.sh"
    Write-Log "Starting logging utility via WSL ($Distro)"
    & wsl.exe -d $Distro -- bash -lc $command
}

function Invoke-BridgeSmokeTest {
    param(
        [Parameter(Mandatory = $true)][string]$ControllerPath,
        [switch]$Silent
    )
    $controllerDir = Split-Path $ControllerPath
    $pythonCmd = Get-PythonCommand
    $code = @"
import json
import sys
sys.path.insert(0, r"""$controllerDir""")
from big_3_Windows import WSLToolBridge
bridge = WSLToolBridge()
result = bridge.list_agents()
print(json.dumps(result))
"@
    if (-not $Silent) { Write-Log "Running bridge smoke test from Windows" }
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = $pythonCmd.Executable
    $psi.Arguments = ($pythonCmd.Args -join ' ')
    $psi.WorkingDirectory = $controllerDir
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $null = $process.Start()
    $process.StandardInput.WriteLine($code)
    $process.StandardInput.Close()
    $process.WaitForExit()

    $output = $process.StandardOutput.ReadToEnd()
    $errorOutput = $process.StandardError.ReadToEnd()
    if ($process.ExitCode -ne 0) {
        throw "Bridge smoke test failed with exit code $($process.ExitCode): $errorOutput"
    }
    if (-not $Silent) {
        Write-Log "Bridge response: $output"
    }
    return $output
}

function Get-AgentLauncher {
    param([string]$Controller)
    if (Get-Command uv.exe -ErrorAction SilentlyContinue) {
        return @{ Executable = 'uv'; Args = @('run', $Controller) }
    }
    $pythonCmd = Get-PythonCommand
    $args = @($Controller)
    if ($pythonCmd.Executable -eq 'uv') {
        $args = @('run', $Controller)
    }
    return @{ Executable = $pythonCmd.Executable; Args = $args }
}

Import-DotEnv -Path $WindowsEnvFile

if (-not $Env:WSL_TOOL_BRIDGE_CMD) {
    $Env:WSL_TOOL_BRIDGE_CMD = "wsl.exe -d $WSLDistro -- python3"
}

if ($SmokeTest) {
    Invoke-LoggingStartup -Distro $WSLDistro
    Invoke-BridgeSmokeTest -ControllerPath $controllerPath | Out-Null
    Write-Log 'Smoke test succeeded.'
    return
}

Invoke-LoggingStartup -Distro $WSLDistro

try {
    Invoke-BridgeSmokeTest -ControllerPath $controllerPath -Silent | Out-Null
} catch {
    Write-Log "Warning: bridge smoke test failed - $_"
}

$launcher = Get-AgentLauncher -Controller $controllerPath
$arguments = $launcher.Args
if ($Prompt) {
    $arguments += @('--prompt', $Prompt)
}

Write-Log "Starting realtime controller via $($launcher.Executable)"
Start-Process -FilePath $launcher.Executable -ArgumentList $arguments -WorkingDirectory (Split-Path $controllerPath) -WindowStyle Normal
Write-Log 'Launch command dispatched.'


