# Fix SSH Connection Timeout

## The Problem
"Operation timed out" means your Mac can't reach the Windows machine on port 22.

## Quick Checks

### 1. Are Both Computers on the Same Network?

**On your Mac, check your IP:**
```bash
ifconfig | grep "inet "
```

Look for an IP starting with `10.119.178.x` (should be on same network as Windows: `10.119.178.85`)

**If your Mac has a different IP range** (like `192.168.x.x` or `172.x.x.x`), you're on different networks!

### 2. Check Windows Firewall

**On Windows (this machine), run PowerShell as Admin:**
```powershell
# Check if firewall rule exists
Get-NetFirewallRule -Name sshd

# If it doesn't exist or is disabled, create it:
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22
```

### 3. Check if SSH is Listening

**On Windows, run:**
```powershell
netstat -an | findstr :22
```

You should see something like:
```
TCP    0.0.0.0:22             0.0.0.0:0              LISTENING
```

### 4. Test from Windows Itself

**On Windows, try connecting to itself:**
```powershell
ssh localhost
```

If this works, SSH is running. If not, SSH service might not be started.

## Solutions

### Solution 1: Ensure Both on Same Network

- Connect both computers to the **same Wi-Fi network** or **same router**
- Make sure neither is on a VPN that might route traffic differently

### Solution 2: Check Windows Firewall

**Run PowerShell as Admin:**
```powershell
# Allow SSH through firewall
New-NetFirewallRule -Name sshd -DisplayName 'OpenSSH Server' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 -ErrorAction SilentlyContinue

# Verify SSH service is running
Get-Service sshd
Start-Service sshd  # if not running
```

### Solution 3: Temporarily Disable Firewall (for testing)

**Only for testing! Re-enable after:**
```powershell
# Disable firewall temporarily
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False

# Test connection from Mac
# If it works, firewall was the issue

# Re-enable firewall
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True

# Then properly configure the rule above
```

### Solution 4: Check Router Settings

Some routers block incoming connections. You might need to:
- Check router firewall settings
- Ensure both devices are on the same subnet
- Some corporate networks block SSH

## Alternative: Use Different Port

If port 22 is blocked, try a different port:

**On Windows (PowerShell as Admin):**
```powershell
# Edit SSH config
notepad C:\ProgramData\ssh\sshd_config

# Change: #Port 22
# To: Port 2222

# Restart service
Restart-Service sshd

# Update firewall
New-NetFirewallRule -Name sshd-alt -DisplayName 'OpenSSH Server Alt' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 2222
```

**On Mac, connect with:**
```bash
ssh -p 2222 fyp@10.119.178.85
```

## Quick Diagnostic Commands

**On Windows:**
```powershell
# Check SSH service
Get-Service sshd

# Check if listening
netstat -an | findstr :22

# Check firewall rules
Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*SSH*"}
```

**On Mac:**
```bash
# Test if port is reachable
nc -zv 10.119.178.85 22

# Or use telnet
telnet 10.119.178.85 22
```

If `nc` or `telnet` can't connect, it's a network/firewall issue, not SSH configuration.

