# Sample Windows-side GPU memory every 30 s: adapter dedicated/shared/committed plus the desktop
# processes holding more than 50 MB. `Total Committed` above the physical card size is the VidMm
# paging signal that invalidated the 2026-09-09 daytime batches (ADR 0007). Appends; run for ~10 h.
#
# usage: powershell.exe -NoProfile -ExecutionPolicy Bypass -File scripts\win\vram-sampler.ps1 [-Out <path>] [-Samples 1200]
param(
  [string]$Out = "$env:TEMP\w2-win-vram.log",
  [int]$Samples = 1200,
  [int]$IntervalSeconds = 30
)

if (-not (Test-Path $Out)) {
  "t_iso dedicated_mb shared_mb committed_mb top_nonvm_dedicated" | Out-File -FilePath $Out -Encoding utf8
}
for ($i = 0; $i -lt $Samples; $i++) {
  try {
    $c = (Get-Counter -Counter '\GPU Adapter Memory(*)\Dedicated Usage','\GPU Adapter Memory(*)\Shared Usage','\GPU Adapter Memory(*)\Total Committed' -ErrorAction Stop).CounterSamples
    $d = ($c | Where-Object { $_.Path -like '*dedicated usage' } | Measure-Object CookedValue -Sum).Sum / 1MB
    $s = ($c | Where-Object { $_.Path -like '*shared usage' } | Measure-Object CookedValue -Sum).Sum / 1MB
    $t = ($c | Where-Object { $_.Path -like '*total committed' } | Measure-Object CookedValue -Sum).Sum / 1MB
    $p = (Get-Counter -Counter '\GPU Process Memory(*)\Dedicated Usage' -ErrorAction Stop).CounterSamples | Sort-Object CookedValue -Descending
    $top = ""
    foreach ($x in $p) {
      $id = [regex]::Match($x.InstanceName, 'pid_(\d+)').Groups[1].Value
      $n = (Get-Process -Id $id -ErrorAction SilentlyContinue).ProcessName
      if ($n -and $n -ne 'vmwp' -and $n -ne 'vmmem' -and $x.CookedValue -gt 50MB) {
        $top += ('{0}:{1:N0} ' -f $n, ($x.CookedValue / 1MB))
      }
      if ($top.Length -gt 120) { break }
    }
    ('{0} {1:F0} {2:F0} {3:F0} {4}' -f (Get-Date -Format 'HH:mm:ss'), $d, $s, $t, $top.Trim()) | Out-File -FilePath $Out -Encoding utf8 -Append
  } catch {
    ('{0} error {1}' -f (Get-Date -Format 'HH:mm:ss'), $_.Exception.Message) | Out-File -FilePath $Out -Encoding utf8 -Append
  }
  Start-Sleep -Seconds $IntervalSeconds
}
