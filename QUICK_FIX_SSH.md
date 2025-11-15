# Quick Fix: SSH Connection Timeout

## Problem
`ssh fyp@10.119.178.85` times out even though SSH server is running.

## Root Cause
The IP `10.119.178.85` is a **private IP** - only works on the same local network.

## Solution 1: Use ngrok (Fastest - 2 minutes)

### On Windows:
```powershell
# 1. Start ngrok tunnel (if not already running)
ngrok tcp 22

# 2. Copy the forwarding address from output:
#    Example: tcp://0.tcp.ngrok.io:12345 -> localhost:22
```

### On Mac (in Cursor):
Use the ngrok address:
```
Host: 0.tcp.ngrok.io
Port: 12345
User: fyp
```

**Note:** ngrok address changes each time you restart it.

---

## Solution 2: Use Tailscale (Best - Permanent)

### On Windows:
```powershell
# Install Tailscale
winget install Tailscale.Tailscale

# After installation:
# 1. Open Tailscale app
# 2. Sign up/login
# 3. Note the Tailscale IP (e.g., 100.x.x.x)
```

### On Mac:
```bash
# Install Tailscale
brew install tailscale
# Or download from: https://tailscale.com/download

# After installation:
# 1. Sign in with same account
# 2. Connect using Tailscale IP:
ssh fyp@<tailscale-ip>
```

**Advantages:**
- ✅ Permanent IP address
- ✅ Works from anywhere
- ✅ Secure VPN
- ✅ Free for personal use

---

## Which Solution?

- **Need quick test?** → Use ngrok
- **Need permanent access?** → Use Tailscale (recommended)

---

## Current Status

✅ SSH Server: Running on port 22  
✅ Firewall: Configured  
❌ Remote Connection: Timeout (network issue)

**Next Step:** Choose ngrok (quick) or Tailscale (permanent) above.

