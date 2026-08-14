[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$videosRoot = Join-Path $projectRoot "downloads\youtube\init"
$python = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $videosRoot -PathType Container)) {
    throw "Dossier des videos introuvable : $videosRoot"
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Executable Python introuvable : $python"
}

$oldArtifacts = Get-ChildItem -LiteralPath $videosRoot -Recurse -File |
    Where-Object {
        $_.Name -eq "transcript_plain.txt" -or
        $_.Name -eq "transcript_chunks.json" -or
        $_.Name -like "*_embedding.json"
    }

Write-Host "Suppression de $($oldArtifacts.Count) ancien(s) transcript(s) plain, fichier(s) de chunks et embedding(s)..."
$oldArtifacts | Remove-Item -Force

function Invoke-PipelineTask {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Task,

        [Parameter(Mandatory = $true)]
        [string]$Selection
    )

    Write-Host "`n=== $Task ($Selection) ==="
    & $python -m pipeline task --force $Task $Selection

    if ($LASTEXITCODE -ne 0) {
        throw "Echec de la tache : $Task ($Selection, code $LASTEXITCODE)"
    }
}

Push-Location $projectRoot
try {
    Invoke-PipelineTask "transcript.create_plain" "all"
    Invoke-PipelineTask "chunks.create" "all"

    $longVideoIds = @(
        Get-ChildItem -LiteralPath $videosRoot -Recurse -File -Filter "transcript_chunks.json" |
            ForEach-Object {
                $payload = Get-Content -LiteralPath $_.FullName -Raw | ConvertFrom-Json
                if ($payload.chunking.profile -eq "long") {
                    $_.Directory.Parent.Parent.Name
                }
            }
    )

    Write-Host "`n$($longVideoIds.Count) video(s) longue(s) a resumer."
    foreach ($task in @("chunks.summarize_sections", "chunks.summarize_video")) {
        foreach ($videoId in $longVideoIds) {
            Invoke-PipelineTask $task $videoId
        }
    }

    Invoke-PipelineTask "embeddings.create" "all"
}
finally {
    Pop-Location
}

Write-Host "`nRegeneration terminee."
