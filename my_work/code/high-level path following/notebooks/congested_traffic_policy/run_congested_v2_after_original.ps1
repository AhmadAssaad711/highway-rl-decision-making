param(
    [int] $PollSeconds = 300,
    [int] $MaxWaitHours = 72
)

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$OriginalResultsDir = Join-Path $ProjectRoot "artifacts\dqn\congested_traffic_policy"
$V2ResultsDir = Join-Path $ProjectRoot "artifacts\dqn\congested_traffic_policy_v2"
$Runner = Join-Path $PSScriptRoot "run_congested_traffic_policy_v2.py"
$LogPath = Join-Path $V2ResultsDir "deferred_v2_runner.log"

New-Item -ItemType Directory -Force -Path $V2ResultsDir | Out-Null

$RequiredRuns = @(
    "congested_baseline_dqn_20k",
    "congested_baseline_dqn_safety_reward_20k",
    "congested_attention_dqn_20k",
    "congested_attention_dqn_safety_reward_20k",
    "congested_adaptive_wide_band_20k",
    "congested_adaptive_wide_band_safety_reward_20k",
    "congested_adaptive_wide_band_attention_20k"
)

function Write-Log {
    param([string] $Message)
    $Line = "[{0}] {1}" -f (Get-Date).ToString("yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -LiteralPath $LogPath -Value $Line
}

function Get-MissingOriginalMarkers {
    $Missing = @()
    foreach ($RunName in $RequiredRuns) {
        $SummaryPath = Join-Path $OriginalResultsDir "$RunName\summary.json"
        $EvalSummaryPath = Join-Path $OriginalResultsDir "$RunName\saved_model_eval_1000_episodes\evaluation_summary.json"
        if (-not ((Test-Path -LiteralPath $SummaryPath) -and (Test-Path -LiteralPath $EvalSummaryPath))) {
            $Missing += $RunName
        }
    }
    return $Missing
}

function Get-BusyOriginalNotebookSessions {
    $BusySessions = @()
    $RuntimeDir = Join-Path $env:APPDATA "jupyter\runtime"
    if (-not (Test-Path -LiteralPath $RuntimeDir)) {
        return $BusySessions
    }

    $ServerFiles = Get-ChildItem -LiteralPath $RuntimeDir -Filter "jpserver-*.json" -ErrorAction SilentlyContinue
    foreach ($ServerFile in $ServerFiles) {
        try {
            $Server = Get-Content -LiteralPath $ServerFile.FullName -Raw | ConvertFrom-Json
            $BaseUrl = [string] $Server.url
            $Token = [string] $Server.token
            if ([string]::IsNullOrWhiteSpace($BaseUrl) -or [string]::IsNullOrWhiteSpace($Token)) {
                continue
            }
            $SessionUrl = ($BaseUrl.TrimEnd("/") + "/api/sessions?token=" + $Token)
            $Sessions = Invoke-RestMethod -Uri $SessionUrl -TimeoutSec 5 -ErrorAction Stop
            foreach ($Session in $Sessions) {
                $Path = [string] $Session.path
                $State = [string] $Session.kernel.execution_state
                if (
                    $Path.StartsWith("congested_traffic_policy") -and
                    -not $Path.Contains("_v2") -and
                    $State -eq "busy"
                ) {
                    $BusySessions += "$Path ($($Session.kernel.id))"
                }
            }
        } catch {
            continue
        }
    }
    return $BusySessions
}

Write-Log "Waiting for original congested traffic run markers under $OriginalResultsDir"
$Deadline = (Get-Date).AddHours($MaxWaitHours)

while ($true) {
    $Missing = @(Get-MissingOriginalMarkers)
    $BusySessions = @(Get-BusyOriginalNotebookSessions)
    if (($Missing.Count -eq 0) -and ($BusySessions.Count -eq 0)) {
        Write-Log "Original congested traffic run appears complete and idle. Starting v2 runner."
        break
    }

    if ((Get-Date) -ge $Deadline) {
        Write-Log "Timed out waiting for original run. Missing: $($Missing -join ', '); busy sessions: $($BusySessions -join ', ')"
        exit 2
    }

    Write-Log "Still waiting. Missing: $($Missing -join ', '); busy sessions: $($BusySessions -join ', ')"
    Start-Sleep -Seconds $PollSeconds
}

Set-Location -LiteralPath $ProjectRoot
Write-Log "Launching: python $Runner"
python $Runner *>> $LogPath
$ExitCode = $LASTEXITCODE
Write-Log "v2 runner exited with code $ExitCode"
exit $ExitCode
