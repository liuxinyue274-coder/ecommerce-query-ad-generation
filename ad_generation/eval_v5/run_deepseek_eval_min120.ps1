$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)

if (-not $env:DEEPSEEK_API_KEY) {
    $userKey = [Environment]::GetEnvironmentVariable("DEEPSEEK_API_KEY", "User")
    if ($userKey) {
        $env:DEEPSEEK_API_KEY = $userKey
    }
}

if (-not $env:DEEPSEEK_BASE_URL) {
    $userBase = [Environment]::GetEnvironmentVariable("DEEPSEEK_BASE_URL", "User")
    if ($userBase) {
        $env:DEEPSEEK_BASE_URL = $userBase
    }
}

if (-not $env:DEEPSEEK_MODEL) {
    $userModel = [Environment]::GetEnvironmentVariable("DEEPSEEK_MODEL", "User")
    if ($userModel) {
        $env:DEEPSEEK_MODEL = $userModel
    }
}

if (-not $env:DEEPSEEK_API_KEY) {
    throw "DEEPSEEK_API_KEY is not visible in this session. Please set it in the current PowerShell or save it as a User environment variable, then reopen the terminal."
}

if (-not $env:DEEPSEEK_BASE_URL) {
    $env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
}

if (-not $env:DEEPSEEK_MODEL) {
    $env:DEEPSEEK_MODEL = "deepseek-chat"
}

$env:PYTHONDONTWRITEBYTECODE = "1"

$buildScript = Join-Path $scriptDir "build_v5_eval_set.py"
$outputPath = Join-Path $projectRoot "outputs\demo_v5_eval_min120_deepseek.jsonl"
$statsPath = Join-Path $projectRoot "outputs\demo_v5_eval_min120_deepseek_stats.json"
$notesPath = Join-Path $projectRoot "outputs\demo_v5_eval_min120_deepseek_sampling_notes.md"

Write-Host "Running deepseek V5 eval build..."
Write-Host "Output: $outputPath"
Write-Host "Stats:  $statsPath"
Write-Host "Notes:  $notesPath"

python -B -X utf8 $buildScript `
  --provider deepseek_chat `
  --output_path $outputPath `
  --stats_path $statsPath `
  --notes_path $notesPath
