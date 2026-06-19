<#
.SYNOPSIS
  Reclaim disk by clearing regenerable gameplay/lore render artifacts and caches.
  DRY-RUN by default — prints what it would do. Pass -Apply to act.

.DESCRIPTION
  SAFE to remove (regenerable from the kept source + transcript):
    - output/gameplay/*/  reframed.mp4, captioned.mp4, fx.mp4, captions.ass
    - output/gameplay/*/  scratch: _preview*, _audio16k*, _na_*, _yt_*, _hook_*, _bg*, _tone*
    - output/_work/*      (lore-pipeline intermediates)
    - output/demo_render.mp4
    - caches: __pycache__/ dirs, *.pyc, .pytest_cache/   (deleted directly — unambiguous)

  NEVER TOUCHED (the KEEP list):
    - output/gameplay/*/source.*        the imported clips (often the only copy)
    - output/gameplay/*/transcript.json, candidates.json, preview thumbnails
    - .venv/, assets/, fonts/, refs/, source code, config, tests, .git/

  Media is MOVED into _cleanup_quarantine/ (mirroring its path) so it is trivially
  recoverable — delete that folder yourself once a pipeline run confirms all is well.
  Only caches are hard-deleted.

.PARAMETER Apply
  Actually move/delete. Without it, the script only reports (dry-run).

.PARAMETER IncludeFinals
  Also quarantine the finished *_short.mp4 deliverables (regenerable). Off by default
  so the safe re-run never removes a deliverable unless you ask.

.EXAMPLE
  pwsh scripts/clean_outputs.ps1                 # dry-run, intermediates + caches
  pwsh scripts/clean_outputs.ps1 -Apply          # do it, keep the finals
  pwsh scripts/clean_outputs.ps1 -Apply -IncludeFinals
#>
param([switch]$Apply, [switch]$IncludeFinals)

$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$out  = Join-Path $root "output"
$quar = Join-Path $root "_cleanup_quarantine"

$derivedNames = @("reframed.mp4", "captioned.mp4", "fx.mp4", "captions.ass")
$scratchPat   = '^(_preview|_audio16k|_na_|_yt_|_hook_|_bg|_tone)'

function Is-SafeMedia($f) {
    $rel = $f.FullName.Substring($root.Length).TrimStart('\','/')
    if ($rel -like '_cleanup_quarantine*') { return $false }
    # gameplay per-clip derived/scratch (never source.* / transcript / thumbnails)
    if ($rel -like 'output\gameplay\*') {
        if ($f.Name -like 'source.*') { return $false }
        if ($f.Name -in 'transcript.json','candidates.json') { return $false }
        if ($f.Extension -in '.jpg','.jpeg','.png') { return $false }
        if ($derivedNames -contains $f.Name) { return $true }
        if ($f.Name -match $scratchPat) { return $true }
        if ($IncludeFinals -and $f.Name -like '*_short.mp4') { return $true }
        return $false
    }
    if ($rel -like 'output\_work\*') { return $true }
    if ($rel -ieq 'output\demo_render.mp4') { return $true }
    return $false
}

# ---- collect media to quarantine ----
$media = @()
if (Test-Path $out) {
    $media = Get-ChildItem -LiteralPath $out -Recurse -File -Force -ErrorAction SilentlyContinue |
             Where-Object { Is-SafeMedia $_ }
}
$mediaMB = [math]::Round((($media | Measure-Object Length -Sum).Sum)/1MB, 1)

# ---- collect caches to delete (repo only, never .venv) ----
$pyc = Get-ChildItem -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue |
       Where-Object { $_.FullName -notlike "*\.venv\*" -and
                      ($_.Name -eq '__pycache__' -and $_.PSIsContainer -or $_.Extension -eq '.pyc' -or $_.Name -eq '.pytest_cache') }
$cacheMB = [math]::Round((((Get-ChildItem -LiteralPath $root -Recurse -File -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -notlike "*\.venv\*" -and ($_.Extension -eq '.pyc' -or $_.FullName -like '*\__pycache__\*' -or $_.FullName -like '*\.pytest_cache\*') }) |
            Measure-Object Length -Sum).Sum)/1MB, 1)

$mode = if ($Apply) { "APPLY" } else { "DRY-RUN (no changes — pass -Apply to act)" }
Write-Output "clean_outputs.ps1  [$mode]   finals: $(if($IncludeFinals){'quarantined'}else{'kept'})"
Write-Output ("  quarantine media : {0} files, {1} MB -> _cleanup_quarantine/" -f $media.Count, $mediaMB)
Write-Output ("  delete caches    : {0} MB (__pycache__, *.pyc, .pytest_cache)" -f $cacheMB)

if (-not $Apply) {
    $media | Select-Object -First 8 | ForEach-Object { "    would quarantine: " + $_.FullName.Substring($root.Length+1) }
    Write-Output "  (run again with -Apply to perform)"
    return
}

# ---- perform: move media into the mirrored quarantine path ----
foreach ($f in $media) {
    $rel  = $f.FullName.Substring($root.Length).TrimStart('\','/')
    $dest = Join-Path $quar $rel
    $ddir = Split-Path $dest -Parent
    if (-not (Test-Path $ddir)) { New-Item -ItemType Directory -Path $ddir -Force | Out-Null }
    Move-Item -LiteralPath $f.FullName -Destination $dest -Force
}
# ---- perform: hard-delete caches ----
Get-ChildItem -LiteralPath $root -Recurse -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.PSIsContainer -and ($_.Name -eq '__pycache__' -or $_.Name -eq '.pytest_cache') } |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }
Get-ChildItem -LiteralPath $root -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.FullName -notlike "*\.venv\*" -and $_.Extension -eq '.pyc' } |
    ForEach-Object { Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue }

Write-Output "Done. Quarantined media is in _cleanup_quarantine/ (delete it yourself after a clean run)."
