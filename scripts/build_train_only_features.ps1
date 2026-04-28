#Requires -Version 5.1

param(
    [ValidateSet("all", "yelp-merchant", "finance-merchant")]
    [string]$Data = "all",
    [double]$MinMatchRate = 0.98
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir

Push-Location $RootDir
try {
    python features/build_train_only_features.py --data $Data --min-match-rate $MinMatchRate
} finally {
    Pop-Location
}
