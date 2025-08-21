# PowerShell wrapper for the Agent to startup
# This script runs the batch file with proper error handling

$ScriptDir = "C:\Users\varun\varun-agent"
$BatchFile = Join-Path $ScriptDir "start_server.bat"

# Set working directory
Set-Location $ScriptDir

# Log startup attempt
$LogFile = Join-Path $ScriptDir "startup.log"
$Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$Timestamp] Starting Varun AI Agent from $ScriptDir..." | Out-File -FilePath $LogFile -Append

# Verify batch file exists
if (-not (Test-Path $BatchFile)) {
    "[$Timestamp] ERROR: Batch file not found at $BatchFile" | Out-File -FilePath $LogFile -Append
    exit 1
}

try {
    # Run the batch file and capture output
    "[$Timestamp] Executing: $BatchFile" | Out-File -FilePath $LogFile -Append
    
    $Process = Start-Process -FilePath "cmd.exe" -ArgumentList "/c `"$BatchFile`"" -WorkingDirectory $ScriptDir -NoNewWindow -PassThru -RedirectStandardOutput "$ScriptDir\output.log" -RedirectStandardError "$ScriptDir\error.log"
    
    "[$Timestamp] Process started with PID: $($Process.Id)" | Out-File -FilePath $LogFile -Append
    
    # Wait a bit to ensure it started properly
    Start-Sleep -Seconds 10
    
    if (-not $Process.HasExited) {
        "[$Timestamp] Varun AI Agent started successfully" | Out-File -FilePath $LogFile -Append
    } else {
        "[$Timestamp] Process exited unexpectedly with code: $($Process.ExitCode)" | Out-File -FilePath $LogFile -Append
    }
}
catch {
    $ErrorMsg = $_.Exception.Message
    "[$Timestamp] Error starting agent: $ErrorMsg" | Out-File -FilePath $LogFile -Append
}

# Keep PowerShell window hidden by exiting
exit