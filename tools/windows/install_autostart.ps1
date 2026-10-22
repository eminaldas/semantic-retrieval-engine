# start_rag.ps1'i oturum acilisinda calisan bir zamanlanmis gorev olarak kaydeder (yonetici gerekmez).
# Kaldirmak icin: Unregister-ScheduledTask -TaskName "RAG MCP autostart" -Confirm:$false

$script = Join-Path $PSScriptRoot "start_rag.ps1"
$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$script`""
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = "PT30S"  # Docker Desktop'in kendi acilisina pay
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 15)
Register-ScheduledTask -TaskName "RAG MCP autostart" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Docker, RAG MCP sunucusu ve Cloudflare tunelini baslatir" -Force | Out-Null
Write-Output "Kayitli: RAG MCP autostart"
