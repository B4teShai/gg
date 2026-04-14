#Requires -Version 5.1
# run_synthetic.ps1 — Train all model variants on synthetic-merchant and print results.
#
# Usage:
#   .\run_synthetic.ps1 [-Device cuda|mps|cpu] [-Epoch N] [-Seeds 42,100,123]

param(
    [string]$Device  = "cuda",
    [int]$Epoch      = 150,
    [int[]]$Seeds    = @(42, 100, 123)
)

$ErrorActionPreference = 'Stop'

$Dataset    = "synthetic-merchant"
$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir    = Split-Path -Parent $ScriptDir
$ResultsDir = Join-Path $RootDir "Results"
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

Write-Host "============================================================"
Write-Host "  Dataset : $Dataset"
Write-Host "  Device  : $Device"
Write-Host "  Epochs  : $Epoch"
Write-Host "  Seeds   : $($Seeds -join ', ')"
Write-Host "============================================================"

function Show-Results {
    param([string]$Tag)
    $metrics = [ordered]@{ 'HR@10'=@(); 'NDCG@10'=@(); 'HR@20'=@(); 'NDCG@20'=@() }
    foreach ($s in $Seeds) {
        $f = Join-Path $ResultsDir "${Tag}_seed${s}.json"
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

# ── 0. Extract improved 8+8 node features ───────────────────────
Write-Host ""
Write-Host ">>> [0/4] Extracting improved features for $Dataset"
Write-Host "------------------------------------------------------------"
Push-Location $RootDir
try {
    python scripts/extract_features_v2.py --dataset $Dataset
} finally { Pop-Location }

# ── 1. SelfGNN-Base (no features) ───────────────────────────────
Write-Host ""
Write-Host ">>> [1/4] selfGNN-Base on $Dataset"
Write-Host "------------------------------------------------------------"
foreach ($Seed in $Seeds) {
    Write-Host "  -- seed $Seed"
    Push-Location (Join-Path $RootDir "selfGNN-Base")
    try {
        python train.py `
            --data      $Dataset `
            --device    $Device  `
            --epoch     $Epoch   `
            --seed      $Seed    `
            --graphNum  12       `
            --save_path "synthetic_merchant_base_seed${Seed}"
    } finally { Pop-Location }
}

# ── 2. SelfGNN-Feature (node features only) ─────────────────────
Write-Host ""
Write-Host ">>> [2/4] selfGNN-Feature (node only) on $Dataset"
Write-Host "------------------------------------------------------------"
foreach ($Seed in $Seeds) {
    Write-Host "  -- seed $Seed"
    Push-Location (Join-Path $RootDir "selfGNN-Feature")
    try {
        python train.py `
            --data      $Dataset `
            --device    $Device  `
            --epoch     $Epoch   `
            --seed      $Seed    `
            --graphNum  12       `
            --use_node_features  `
            --node_mlp_hidden 128 `
            --save_path "synthetic_merchant_node_seed${Seed}"
    } finally { Pop-Location }
}

# ── 3. SelfGNN-Feature (edge features only) ─────────────────────
Write-Host ""
Write-Host ">>> [3/4] selfGNN-Feature (edge only) on $Dataset"
Write-Host "------------------------------------------------------------"
foreach ($Seed in $Seeds) {
    Write-Host "  -- seed $Seed"
    Push-Location (Join-Path $RootDir "selfGNN-Feature")
    try {
        python train.py `
            --data      $Dataset `
            --device    $Device  `
            --epoch     $Epoch   `
            --seed      $Seed    `
            --graphNum  12       `
            --use_edge_features  `
            --save_path "synthetic_merchant_edge_seed${Seed}"
    } finally { Pop-Location }
}

# ── 4. SelfGNN-Feature (node + edge features) ───────────────────
Write-Host ""
Write-Host ">>> [4/4] selfGNN-Feature (node + edge) on $Dataset"
Write-Host "------------------------------------------------------------"
foreach ($Seed in $Seeds) {
    Write-Host "  -- seed $Seed"
    Push-Location (Join-Path $RootDir "selfGNN-Feature")
    try {
        python train.py `
            --data      $Dataset `
            --device    $Device  `
            --epoch     $Epoch   `
            --seed      $Seed    `
            --graphNum  12       `
            --use_node_features  `
            --use_edge_features  `
            --node_mlp_hidden 128 `
            --save_path "synthetic_merchant_node_edge_seed${Seed}"
    } finally { Pop-Location }
}

# ── Summary ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================"
Write-Host "  RESULTS SUMMARY  --  $Dataset  (mean +/- std)"
Write-Host "============================================================"

foreach ($tag in @(
    "synthetic_merchant_base",
    "synthetic_merchant_node",
    "synthetic_merchant_edge",
    "synthetic_merchant_node_edge"
)) {
    Show-Results -Tag $tag
}

Write-Host ""
Write-Host "All runs complete for $Dataset."
