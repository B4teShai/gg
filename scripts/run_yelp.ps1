#Requires -Version 5.1
# run_yelp.ps1 — Full final-submission sweep on yelp-merchant.
#
# Phase A : 4 SelfGNN variants (base | node | edge | node+edge) -> Results2\
# Phase B : 5 baselines (popularity, bprmf, lightgcn, sasrec, bert4rec)
#           -> Results_baselines\

param(
    [string]$Device          = "cuda",
    [int]$Epoch              = 150,
    [int[]]$Seeds            = @(100, 999),
    [int]$BaselineEpochs     = 100,
    [int]$Patience           = 10,
    [switch]$SkipBaselines
)

$ErrorActionPreference = 'Continue'

$Dataset     = "yelp-merchant"
$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir     = Split-Path -Parent $ScriptDir
$ResultsDir  = Join-Path $RootDir "Results2"
$BaselineDir = Join-Path $RootDir "Results_baselines"
New-Item -ItemType Directory -Force -Path $ResultsDir  | Out-Null
New-Item -ItemType Directory -Force -Path $BaselineDir | Out-Null

Write-Host "============================================================"
Write-Host "  Dataset : $Dataset"
Write-Host "  Device  : $Device"
Write-Host "  SelfGNN epochs  : $Epoch"
Write-Host "  Baseline epochs : $BaselineEpochs (patience $Patience)"
Write-Host "  Seeds   : $($Seeds -join ', ')"
Write-Host "============================================================"

function Clear-GPUCache {
    Write-Host "  [mem] Clearing GPU/CPU memory cache..."
    python -c "import gc; gc.collect(); $(if ($Device -eq 'cuda') { 'import torch; torch.cuda.empty_cache()' } else { 'pass' })" 2>$null
    Start-Sleep -Seconds 2
}

function Show-Results {
    param([string]$Dir, [string]$Tag)
    $metrics = [ordered]@{ 'HR@10'=@(); 'NDCG@10'=@(); 'HR@20'=@(); 'NDCG@20'=@() }
    foreach ($s in $Seeds) {
        $f = Join-Path $Dir "${Tag}_seed${s}.json"
        if (Test-Path $f) {
            $d  = Get-Content $f -Raw | ConvertFrom-Json
            $tr = if ($d.test_results) { $d.test_results } else { $null }
            if ($tr) {
                foreach ($m in $metrics.Keys) {
                    if ($null -ne $tr.$m) { $metrics[$m] += [double]$tr.$m }
                }
            }
        }
    }
    Write-Host ""
    Write-Host "* $Tag"
    foreach ($m in $metrics.Keys) {
        $vals = $metrics[$m]
        if ($vals.Count -gt 0) {
            $mean = ($vals | Measure-Object -Average).Average
            $std  = if ($vals.Count -gt 1) {
                $sq = ($vals | ForEach-Object { [Math]::Pow($_ - $mean, 2) } | Measure-Object -Sum).Sum
                [Math]::Sqrt($sq / ($vals.Count - 1))
            } else { 0.0 }
            Write-Host ("  {0,-10} {1:F4} +/- {2:F4}  (n={3})" -f $m, $mean, $std, $vals.Count)
        } else {
            Write-Host "  ${m}: no results found"
        }
    }
}

# ============================================================
# PHASE A -- SelfGNN variants
# ============================================================

Write-Host "`n>>> [A1/4] selfGNN-Base on $Dataset"
foreach ($Seed in $Seeds) {
    Push-Location (Join-Path $RootDir "selfGNN-Base")
    try {
        python train.py --data $Dataset --device $Device --epoch $Epoch --seed $Seed `
            --save_path "yelp_merchant_base_seed${Seed}"
    } catch {
        Write-Warning "  [!] Run failed (seed $Seed): $_"
    } finally {
        Pop-Location
        Clear-GPUCache
    }
}


Write-Host "`n>>> [A3/4] selfGNN-Feature (edge only) on $Dataset"
foreach ($Seed in $Seeds) {
    Push-Location (Join-Path $RootDir "selfGNN-Feature")
    try {
        python train.py --data $Dataset --device $Device --epoch $Epoch --seed $Seed `
            --use_edge_features --save_path "yelp_merchant_edge_seed${Seed}"
    } catch {
        Write-Warning "  [!] Run failed (seed $Seed): $_"
    } finally {
        Pop-Location
        Clear-GPUCache
    }
}

# ============================================================
# Summary
# ============================================================

Write-Host ""
Write-Host "============================================================"
Write-Host "  RESULTS SUMMARY  --  $Dataset  (mean +/- std)"
Write-Host "============================================================"

Write-Host "`n--- SelfGNN variants (Results2/) ---"
foreach ($tag in @("yelp_merchant_base","yelp_merchant_node","yelp_merchant_edge","yelp_merchant_node_edge")) {
    Show-Results -Dir $ResultsDir -Tag $tag
}

if (-not $SkipBaselines) {
    Write-Host "`n--- Baselines (Results_baselines/) ---"
    foreach ($Model in @("popularity","bprmf","lightgcn","sasrec","bert4rec")) {
        Show-Results -Dir $BaselineDir -Tag "yelp_merchant_${Model}"
    }
}

Write-Host ""
Write-Host "All runs complete for $Dataset."
