# Fresh-seed confirmatory runs for the presence-gated head.

# Seeds 1009 / 20260905 / 31337 appear nowhere in models/*.json. Seeds 42/123/2024
# are the development seeds on which the gated remedy was designed; 7/99 were used
# in extra_seeds.json and paired_bootstrap_5seed.json for an MLP-vs-RF MAE tie on
# Fe-Cr-Mo and Fe-Cr-V, so they are excluded here as well.
$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"
$seeds = @(1009, 20260905, 31337)

Write-Output "=== fresh-seed gated runs: seeds $($seeds -join ',') ==="

# Five ternaries share one output file, matching the development-run layout.
py -3.12 train_remedy.py --variants gated `
    --systems fecrmn fecrni fecrmo fecrv femnni `
    --seeds $seeds --out models/results_remedy_freshseeds.json --force
if ($LASTEXITCODE -ne 0) { throw "ternary group failed ($LASTEXITCODE)" }

# Each extension system has its own output file, as in the development runs.
foreach ($sys in @("fecrnic", "fecrc", "crconi", "crnimn")) {
    py -3.12 train_remedy.py --variants gated --system $sys `
        --seeds $seeds --out "models/results_remedy_freshseeds_$sys.json" --force
    if ($LASTEXITCODE -ne 0) { throw "$sys failed ($LASTEXITCODE)" }
}

Write-Output "=== ALL FRESH-SEED RUNS COMPLETE ==="
