# Fix Network/Router Blocking SSH

## Solution 1: Check Router Firewall Settings (Most Common)

### Steps:
1. **Access your router admin panel:**
   - Usually: `192.168.1.1` or `192.168.0.1` or `10.0.0.1`
   - Check router label or documentation
   - Open in browser: `http://192.168.1.1` (or your router's IP)

2. **Look for:**
   - "Firewall" settings
   - "Security" settings
   - "Access Control" or "Device Isolation"
   - "AP Isolation" or "Client Isolation"

3. **Disable these if enabled:**
   - **AP Isolation** / **Client Isolation** - This blocks devices from talking to each other!
   - **Device Isolation**
   - **Wireless Isolation**

4. **Save and restart router**

### Common Router Brands:
- **Netgear**: Advanced → Wireless Settings → AP Isolation (disable)
- **Linksys**: Wireless → Advanced → AP Isolation (disable)
- **TP-Link**: Advanced → Wireless → AP Isolation (disable)
- **ASUS**: Wireless → Professional → AP Isolation (disable)
- **Google Nest**: Settings → Advanced Networking → AP Isolation (disable)

---

## Solution 2: Ensure Both Devices on Same Network

### Check:
1. **Windows IP**: `10.119.178.85` (we know this)
2. **Mac IP**: Run `ifconfig | grep "inet "` on Mac
   - Should be `10.119.178.x` (same first 3 numbers)

### If different:
- Connect both to the **same Wi-Fi network**
- Make sure neither is on a guest network
- Some routers separate guest networks from main network

---

## Solution 3: Windows Firewall - Allow Private Network

**On Windows, run PowerShell as Admin:**

```powershell
# Allow SSH on Private network profile
Set-NetFirewallProfile -Profile Private -Enabled True
New-NetFirewallRule -Name sshd-private -DisplayName 'OpenSSH Server Private' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 -Profile Private

# Also check Domain and Public if needed
New-NetFirewallRule -Name sshd-domain -DisplayName 'OpenSSH Server Domain' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 -Profile Domain
New-NetFirewallRule -Name sshd-public -DisplayName 'OpenSSH Server Public' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 22 -Profile Public
```

---

## Solution 4: Temporarily Disable Windows Firewall (Testing Only!)

**Only for testing! Re-enable after:**

```powershell
# Disable all firewall profiles temporarily
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled False

# Test connection from Mac
# If it works, firewall was blocking

# Re-enable firewall
Set-NetFirewallProfile -Profile Domain,Public,Private -Enabled True

# Then properly configure rules above
```

---

## Solution 5: Use Different Port (If Router Blocks 22)

Some routers block port 22 by default. Use a different port:

**On Windows (PowerShell as Admin):**
```powershell
# Edit SSH config
notepad C:\ProgramData\ssh\sshd_config

# Find: #Port 22
# Change to: Port 2222
# (Remove the # and change 22 to 2222)

# Save and close

# Restart SSH
Restart-Service sshd

# Add firewall rule for new port
New-NetFirewallRule -Name sshd-2222 -DisplayName 'OpenSSH Server 2222' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 2222
```

**On Mac, connect with:**
```bash
ssh -p 2222 fyp@10.119.178.85
```

---

## Solution 6: Check Windows Network Profile

**On Windows:**
1. Open **Settings** → **Network & Internet** → **Wi-Fi** (or Ethernet)
2. Click on your network
3. Make sure it's set to **"Private"** (not Public)
4. Public networks have stricter firewall rules

**Or via PowerShell:**
```powershell
# Check current profile
Get-NetConnectionProfile

# Set to Private if needed
Set-NetConnectionProfile -InterfaceAlias "Wi-Fi" -NetworkCategory Private
```

---

## Quick Test After Each Fix

**On Mac:**
```bash
# Test port connectivity
nc -zv 10.119.178.85 22

# Or try SSH
ssh fyp@10.119.178.85
```

---

## Most Common Fix: AP Isolation

**90% of the time, it's AP Isolation enabled on the router!**

1. Access router admin (usually `192.168.1.1`)
2. Find "AP Isolation" or "Client Isolation"
3. **Disable it**
4. Save and restart router
5. Try connecting again

