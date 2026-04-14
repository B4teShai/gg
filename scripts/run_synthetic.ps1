#Requires -Version 5.1
# run_synthetic.ps1 — Train all model variants on synthetic-merchant and print results.
#
# Usage:
#   .\run_synthetic.ps1 [-Device cuda|mps|cpu] [-Epoch N] [-Seed N]

param(
    [string]$Device = "cuda",
    [int]$Epoch     = 150,
    [int]$Seed      = 100
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
Write-Host "  Seed    : $Seed"
Write-Host "============================================================"

function Show-Result {
    param([string]$Tag, [string]$File)
    if (Test-Path $File) {
        Write-Host ""
        Write-Host "* $Tag"
        $d    = Get-Content $File -Raw | ConvertFrom-Json
        $best = if ($null -ne $d.best_epoch) { $d.best_epoch } else { "?" }
        Write-Host "  Best epoch : $best"
        if ($d.test_results) {
            $tr   = $d.test_results
            $hr10 = if ($null -ne $tr.'HR@10')   { "{0:F4}" -f [double]$tr.'HR@10' }   else { "0.0000" }
            $nd10 = if ($null -ne $tr.'NDCG@10') { "{0:F4}" -f [double]$tr.'NDCG@10' } else { "0.0000" }
            $hr20 = if ($null -ne $tr.'HR@20')   { "{0:F4}" -f [double]$tr.'HR@20' }   else { "0.0000" }
            $nd20 = if ($null -ne $tr.'NDCG@20') { "{0:F4}" -f [double]$tr.'NDCG@20' } else { "0.0000" }
            Write-Host "  Test  HR@10=$hr10  NDCG@10=$nd10  HR@20=$hr20  NDCG@20=$nd20"
        }
        if ($d.val_results) {
            $vr   = $d.val_results
            $hr10 = if ($null -ne $vr.'HR@10')   { "{0:F4}" -f [double]$vr.'HR@10' }   else { "0.0000" }
            $nd10 = if ($null -ne $vr.'NDCG@10') { "{0:F4}" -f [double]$vr.'NDCG@10' } else { "0.0000" }
            $hr20 = if ($null -ne $vr.'HR@20')   { "{0:F4}" -f [double]$vr.'HR@20' }   else { "0.0000" }
            $nd20 = if ($null -ne $vr.'NDCG@20') { "{0:F4}" -f [double]$vr.'NDCG@20' } else { "0.0000" }
            Write-Host "  Val   HR@10=$hr10  NDCG@10=$nd10  HR@20=$hr20  NDCG@20=$nd20"
        }
    } else {
        Write-Host "  ${Tag}: result file not found ($File)"
    }
}

# ── 1. SelfGNN-Base (no features) ───────────────────────────────
Write-Host ""
Write-Host ">>> [1/4] selfGNN-Base on $Dataset"
Write-Host "------------------------------------------------------------"
Push-Location (Join-Path $RootDir "selfGNN-Base")
try {
    python train.py `
        --data      $Dataset `
        --device    $Device  `
        --epoch     $Epoch   `
        --seed      $Seed    `
        --save_path "synthetic_merchant_base"
} finally { Pop-Location }
Write-Host "---- done -> $ResultsDir\synthetic_merchant_base.json"

# ── 2. SelfGNN-Feature (node features only) ─────────────────────
Write-Host ""
Write-Host ">>> [2/4] selfGNN-Feature (node only) on $Dataset"
Write-Host "------------------------------------------------------------"
Push-Location (Join-Path $RootDir "selfGNN-Feature")
try {
    python train.py `
        --data      $Dataset `
        --device    $Device  `
        --epoch     $Epoch   `
        --seed      $Seed    `
        --use_node_features  `
        --save_path "synthetic_merchant_node"
} finally { Pop-Location }
Write-Host "---- done -> $ResultsDir\synthetic_merchant_node.json"

# ── 3. SelfGNN-Feature (edge features only) ─────────────────────
Write-Host ""
Write-Host ">>> [3/4] selfGNN-Feature (edge only) on $Dataset"
Write-Host "------------------------------------------------------------"
Push-Location (Join-Path $RootDir "selfGNN-Feature")
try {
    python train.py `
        --data      $Dataset `
        --device    $Device  `
        --epoch     $Epoch   `
        --seed      $Seed    `
        --use_edge_features  `
        --save_path "synthetic_merchant_edge"
} finally { Pop-Location }
Write-Host "---- done -> $ResultsDir\synthetic_merchant_edge.json"

# ── 4. SelfGNN-Feature (node + edge features) ───────────────────
Write-Host ""
Write-Host ">>> [4/4] selfGNN-Feature (node + edge) on $Dataset"
Write-Host "------------------------------------------------------------"
Push-Location (Join-Path $RootDir "selfGNN-Feature")
try {
    python train.py `
        --data      $Dataset `
        --device    $Device  `
        --epoch     $Epoch   `
        --seed      $Seed    `
        --use_node_features  `
        --use_edge_features  `
        --save_path "synthetic_merchant_node_edge"
} finally { Pop-Location }
Write-Host "---- done -> $ResultsDir\synthetic_merchant_node_edge.json"

# ── Summary ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================"
Write-Host "  RESULTS SUMMARY  --  $Dataset"
Write-Host "============================================================"

foreach ($tag in @(
    "synthetic_merchant_base",
    "synthetic_merchant_node",
    "synthetic_merchant_edge",
    "synthetic_merchant_node_edge"
)) {
    Show-Result -Tag $tag -File (Join-Path $ResultsDir "$tag.json")
}

Write-Host ""
Write-Host "All runs complete for $Dataset."
