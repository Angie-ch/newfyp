# SSH Server Setup for Remote Access in Cursor

## Quick Setup (Run as Administrator)

1. **Open PowerShell as Administrator:**
   - Right-click on PowerShell
   - Select "Run as Administrator"

2. **Navigate to project directory:**
   ```powershell
   cd C:\Users\fyp\Desktop\fyp\typhoon_prediction
   ```

3. **Run the setup script:**
   ```powershell
   .\setup_ssh_server.ps1
   ```

## Manual Setup Steps

If the script doesn't work, follow these steps:

### Step 1: Install OpenSSH Server

```powershell
# Check if OpenSSH Server is available
Get-WindowsCapability -Online | Where-Object Name -like 'OpenSSH.Server*'

# Install OpenSSH Server
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
```

### Step 2: Start SSH Service

```powershell
# Start SSH service
Start-Service sshd

# Set SSH service to start automatically
Set-Service -Name sshd -StartupType 'Automatic'
```

### Step 3: Configure Firewall

```powershell
# Allow SSH through firewall (port 22)
New-NetFirewallRule -Name "OpenSSH-Server-In-TCP" -DisplayName "OpenSSH SSH Server (sshd)" -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

### Step 4: Get Connection Information

```powershell
# Get your IP address
Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -notlike "127.*" -and $_.IPAddress -notlike "169.254.*"}

# Get computer name
$env:COMPUTERNAME

# Get username
$env:USERNAME
```

## Connect from Cursor (Mac)

Once SSH is set up, you can connect from your Mac:

```bash
# Using IP address
ssh fyp@<IP_ADDRESS>

# Or using computer name (if on same network)
ssh fyp@DESKTOP-BF5LLH4
```

## Using ngrok for External Access

If you need to access from outside your local network:

1. **Install ngrok** (if not already installed)

2. **Create TCP tunnel:**
   ```powershell
   ngrok tcp 22
   ```

3. **Connect from Mac:**
   ```bash
   ssh fyp@<ngrok_hostname> -p <ngrok_port>
   ```

## Troubleshooting

### Check SSH Service Status
```powershell
Get-Service sshd
```

### Check if port 22 is listening
```powershell
netstat -an | findstr :22
```

### View SSH logs
```powershell
Get-EventLog -LogName Application -Source OpenSSH -Newest 10
```

### Test SSH connection locally
```powershell
ssh localhost
```

## Security Notes

- **Change default port (optional):** Edit `C:\ProgramData\ssh\sshd_config` and change `Port 22` to another port
- **Use key-based authentication:** More secure than passwords
- **Disable password authentication:** After setting up SSH keys

## Next Steps

After SSH is set up:
1. Test connection from Mac
2. Configure SSH keys for passwordless login
3. Set up port forwarding if needed
4. Connect from Cursor using Remote SSH extension

