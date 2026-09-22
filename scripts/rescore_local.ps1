# Re-score the historical ablations on the relabelled eval set, on a local GPU.
#
# The five systems that appear beside the r6 rows (r4, r4_ctl, r3_e32, gtcrn_finetuned and the
# tier46 cascade) are scored on the training box; this covers the rest, which are ablations whose
# published numbers were measured on the pre-relabel set.
#
# Reads data/eval_r2_relabel - the local render, verified against the box's set: no item differs in
# any selection field and the audio is bit-identical (the EVALSET_HASH differs only because it
# digests float text; see scripts/render_eval_sets.py).
#
# Writes to results_r2/r6_local/ so machine-produced results never collide. Re-runnable: a complete
# csv is skipped, a short one is deleted and redone, so Ctrl-C costs at most the run in flight.
#
#   uv run powershell -File scripts/rescore_local.ps1        (or just: .\scripts\rescore_local.ps1)

$ErrorActionPreference = 'Continue'
# vaani.eval runs the model on CPU in parallel worker processes on purpose - see _init() in
# vaani/eval.py: "the tiny model on CPU (no 8x CUDA contexts) and one torch thread so 8 workers
# don't oversubscribe". A 50k-param model plus PESQ, STOI and DNSMOS is CPU work, so --workers is
# the throughput knob and an idle GPU is expected, not a fault.
$Workers = [Math]::Max(4, [int]$env:NUMBER_OF_PROCESSORS - 4)
$EvalRoot = 'data/eval_r2_relabel'
$OutDir   = 'results_r2/r6_local'
$Expected = 2281           # 2280 items + header
$OnTheBox = @('vaani_full_r4','vaani_full_r4_ctl','vaani_full_r3_e32',
              'gtcrn_finetuned','vaani_tier46_refiner')
# most load-bearing first: these two back the "the DSP controller is a measured null" claim
$First    = @('vaani_full_r3_nodsp','vaani_no_controller_r3')

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$all   = Get-ChildItem results_r2/runs -Directory | ForEach-Object { $_.Name }
$queue = @($First | Where-Object { $all -contains $_ }) +
         @($all   | Where-Object { $OnTheBox -notcontains $_ -and $First -notcontains $_ })

Write-Host "$($queue.Count) systems queued, $Workers workers."
$i = 0
foreach ($n in $queue) {
    $i++
    $ck  = "results_r2/runs/$n/best.pt"
    $out = "$OutDir/${n}_eval_r2.csv"
    if (-not (Test-Path $ck)) { Write-Host "[$i/$($queue.Count)] skip $n (no best.pt)"; continue }
    if (Test-Path $out) {
        $rows = (Get-Content $out | Measure-Object -Line).Lines
        if ($rows -ge $Expected) { Write-Host "[$i/$($queue.Count)] skip $n (complete)"; continue }
        Write-Host "[$i/$($queue.Count)] redo $n (short: $rows rows)"
        Remove-Item $out
    }
    Write-Host "[$i/$($queue.Count)] $n  started $(Get-Date -Format 'HH:mm')"
    uv run python -m vaani.eval --system "ckpt:$ck" --split test --eval-root $EvalRoot `
        --workers $Workers --dnsmos --out $out
    Write-Host "[$i/$($queue.Count)] $n  done $(Get-Date -Format 'HH:mm')  rc=$LASTEXITCODE"
}
Write-Host "queue finished $(Get-Date -Format 'HH:mm')"
