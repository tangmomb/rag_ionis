# Mémo — voir et arrêter les instances

## Lister les processus Uvicorn et serveurs Python

```powershell
Get-CimInstance Win32_Process |
  Where-Object { $_.CommandLine -match "uvicorn|http.server" } |
  Select-Object ProcessId, ParentProcessId, CommandLine
```

## Voir les processus qui écoutent les ports

```powershell
Get-NetTCPConnection -State Listen |
  Where-Object { $_.LocalPort -in 8001,8006 } |
  Select-Object LocalAddress, LocalPort, OwningProcess
```

Alternative :

```powershell
netstat -ano | findstr "8001 8006"
```

## Identifier un PID

```powershell
Get-Process -Id <PID>
```

## Arrêter une instance et ses processus enfants

```powershell
taskkill /PID <PID> /T /F
```

`uvicorn --reload` lance généralement un processus parent de surveillance et un processus enfant serveur. Il faut donc arrêter toute la chaîne avec `/T`.

## Fermer toutes les instances du projet

Cette commande cible uniquement l’API RAG, le database browser et le serveur SQL formatter :

```powershell
$patterns = @(
  "uvicorn interface.app:app",
  "uvicorn utils.database_browser.app:app",
  "http.server 8002"
)

$processes = @(Get-CimInstance Win32_Process | Where-Object {
  $process = $_
  $process.Name -eq "python.exe" -and
    ($patterns | Where-Object { $process.CommandLine -like ("*" + $_ + "*") })
})

foreach ($process in $processes) {
  taskkill /PID $process.ProcessId /T /F
}
```

Ne pas utiliser `taskkill /IM python.exe /F` sauf si tu veux fermer tous les programmes Python de la machine.
