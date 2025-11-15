# SSH Remote Connection Guide

## Current Setup Status

✅ SSH Server: Running on port 22  
✅ Firewall: Configured  
✅ Password Authentication: Enabled  
⚠️  Network: Private IP (10.119.178.85) - only works on same network

## Solution 1: ngrok (Quick Setup - Already Started)

### If ngrok window is open:

1. **Look at the ngrok output window** for a line like:
   ```
   Forwarding  tcp://0.tcp.ngrok.io:12345 -> localhost:22
   ```

2. **Copy the address** (e.g., `0.tcp.ngrok.io:12345`)

3. **In Cursor (Mac):**
   - Press `Cmd+Shift+P`
   - Type "Remote-SSH: Connect to Host"
   - Enter: `fyp@0.tcp.ngrok.io`
   - When prompted for port, enter: `12345` (use the port from ngrok)

4. **Keep the ngrok window open** while using SSH!

**Note:** The address changes each time you restart ngrok.

---

## Solution 2: Tailscale (Permanent Setup)

### On Windows:

1. **Install Tailscale:**
   ```powershell
   winget install Tailscale.Tailscale
   ```
   Or download from: https://tailscale.com/download

2. **Open Tailscale app** (search in Start menu)

3. **Sign up or login** with your account

4. **Note your Tailscale IP** (shown in app, e.g., `100.64.1.2`)

### On Mac:

1. **Install Tailscale:**
   ```bash
   brew install tailscale
   ```
   Or download from: https://tailscale.com/download

2. **Sign in** with the **same account** as Windows

3. **Connect:**
   ```bash
   ssh fyp@<tailscale-ip>
   ```

### In Cursor (Mac):

- Host: `<tailscale-ip>` (e.g., `100.64.1.2`)
- Port: `22`
- User: `fyp`

**Advantages:**
- ✅ Permanent IP (doesn't change)
- ✅ Works from anywhere
- ✅ Secure VPN
- ✅ Free for personal use

---

## Troubleshooting

### ngrok not working?
- Make sure ngrok window is still open
- Check if the address is correct
- Try restarting ngrok: Close window, run `ngrok tcp 22` again

### Tailscale not connecting?
- Make sure both machines are signed in with same account
- Check Tailscale status on both machines (should show "Connected")
- Verify the Tailscale IP is correct

### Still can't connect?
- Check SSH service is running: `Get-Service sshd` (on Windows)
- Verify port 22 is listening: `netstat -an | findstr ":22"`
- Check firewall: `Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*SSH*"}`

---

## Quick Commands

### Check SSH Status (Windows):
```powershell
Get-Service sshd
netstat -an | findstr ":22"
```

### Check Tailscale IP (Windows):
```powershell
tailscale ip
```

### Check Tailscale Status (Mac):
```bash
tailscale status
tailscale ip
```

---

## Recommended: Use Tailscale

For permanent, reliable remote access, **Tailscale is recommended**.

Run on Windows:
```powershell
.\setup_tailscale.ps1
```

Then follow the steps above.

