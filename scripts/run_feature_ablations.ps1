#Requires -Version 5.1

param(
    [ValidateSet("yelp-merchant",  "finance-merchant")]
    [string]$Dataset = "yelp-merchant",
    [string]$Device = "cuda",
    [int]$Epoch = 150,
    [int[]]$Seeds = @(42),
    [string[]]$FeatureGroups = @("value", "time", "category", "repeat", "degree", "all", "all_plus_degree"),
    [switch]$RunNodeEdge,
    [switch]$KeepNodeValueWithEdges
)

$ErrorActionPreference = "Continue"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
$Prefix = $Dataset -replace "-", "_"

function Clear-GPUCache {
    python -c "import gc; gc.collect(); import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>$null
    Start-Sleep -Seconds 2
}

Write-Host "============================================================"
Write-Host "  Feature ablations"
Write-Host "  Dataset : $Dataset"
Write-Host "  Groups  : $($FeatureGroups -join ', ')"
Write-Host "  Seeds   : $($Seeds -join ', ')"
Write-Host "============================================================"

foreach ($Group in $FeatureGroups) {
    foreach ($Seed in $Seeds) {
        Push-Location (Join-Path $RootDir "selfGNN-Feature")
        try {
            Write-Host "`n>>> node-only group=$Group seed=$Seed"
            python train.py --data $Dataset --device $Device --epoch $Epoch --seed $Seed `
                --use_node_features --node_feature_groups $Group `
                --save_path "${Prefix}_node_${Group}_seed${Seed}"
        } catch {
            Write-Warning "node-only failed for group=$Group seed=${Seed}: $_"
        } finally {
            Pop-Location
            Clear-GPUCache
        }

        if ($RunNodeEdge) {
            Push-Location (Join-Path $RootDir "selfGNN-Feature")
            try {
                Write-Host "`n>>> node+edge group=$Group seed=$Seed"
                if ($KeepNodeValueWithEdges) {
                    python train.py --data $Dataset --device $Device --epoch $Epoch --seed $Seed `
                        --use_node_features --use_edge_features --node_feature_groups $Group `
                        --keep_node_value_with_edges `
                        --save_path "${Prefix}_node_edge_${Group}_seed${Seed}"
                } else {
                    python train.py --data $Dataset --device $Device --epoch $Epoch --seed $Seed `
                        --use_node_features --use_edge_features --node_feature_groups $Group `
                        --save_path "${Prefix}_node_edge_${Group}_seed${Seed}"
                }
            } catch {
                Write-Warning "node+edge failed for group=$Group seed=${Seed}: $_"
            } finally {
                Pop-Location
                Clear-GPUCache
            }
        }
    }
}

Write-Host "`nAblation runs complete."
