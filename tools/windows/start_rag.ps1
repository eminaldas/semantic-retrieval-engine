# RAG MCP sunucusunu ayaga kaldirir: Docker (Qdrant/Neo4j), HTTP MCP sunucusu, Cloudflare tuneli.
# Oturum acilisinda zamanlanmis gorevle calisir (tools\windows\install_autostart.ps1); elle de calistirilabilir.
# Calisan parcaya dokunmaz, sadece eksik olani baslatir.

$ErrorActionPreference = "Continue"
$root = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $root
$logs = Join-Path $root "artifacts\logs"
New-Item -ItemType Directory -Force $logs | Out-Null
$log = Join-Path $logs "autostart.log"
function Log($msg) { Add-Content $log "$(Get-Date -Format s) $msg" }
$port = if ($env:RAG_MCP_PORT) { $env:RAG_MCP_PORT } else { 47821 }
$tunnel = if ($env:RAG_TUNNEL_NAME) { $env:RAG_TUNNEL_NAME } else { "rag" }

Log "basladi"

# 1. Docker motoru
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    $desktop = "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if (-not (Get-Process "Docker Desktop" -ErrorAction SilentlyContinue)) { Start-Process $desktop }
    $deadline = (Get-Date).AddMinutes(4)
    do { Start-Sleep 5; docker info *> $null } while ($LASTEXITCODE -ne 0 -and (Get-Date) -lt $deadline)
    if ($LASTEXITCODE -ne 0) { Log "HATA: Docker 4 dakikada acilmadi"; exit 1 }
}
Log "docker hazir"

# 2. Qdrant + Neo4j
docker compose up -d *>> $log
$deadline = (Get-Date).AddMinutes(2)
do {
    Start-Sleep 2
    try { $ready = (Invoke-WebRequest -UseBasicParsing http://localhost:6333/readyz -TimeoutSec 3).StatusCode -eq 200 } catch { $ready = $false }
} while (-not $ready -and (Get-Date) -lt $deadline)
Log "qdrant hazir=$ready"

# 3. MCP HTTP sunucusu (modelleri acilista kendisi yukler)
if (-not (Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)) {
    if (Test-Path "$logs\mcp_http.log") { Copy-Item "$logs\mcp_http.log" "$logs\mcp_http.prev.log" -Force }
    $env:PYTHONIOENCODING = "utf-8"
    Start-Process -WindowStyle Hidden -FilePath ".\.venv\Scripts\python.exe" -ArgumentList "-m", "backend.mcp.http_server", "--port", $port `
        -WorkingDirectory $root -RedirectStandardError "$logs\mcp_http.log" -RedirectStandardOutput "$logs\mcp_http.out.log"
    Log "sunucu baslatildi (port $port)"
} else { Log "sunucu zaten calisiyor" }

# 4. Cloudflare tuneli (~/.cloudflared/config.yml)
if (-not (Get-Process cloudflared -ErrorAction SilentlyContinue)) {
    if (Test-Path "$logs\cloudflared.log") { Move-Item "$logs\cloudflared.log" "$logs\cloudflared.prev.log" -Force }
    Start-Process -WindowStyle Hidden -FilePath "cloudflared" -ArgumentList "tunnel", "--logfile", "$logs\cloudflared.log", "run", $tunnel
    Log "tunel baslatildi ($tunnel)"
} else { Log "tunel zaten calisiyor" }

Log "bitti"
