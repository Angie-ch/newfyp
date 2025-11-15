# SSH Remote Access Solutions

## Problem
SSH connection times out from remote machine (Mac) even though SSH server is running correctly on Windows.

## Root Cause
The IP address `10.119.178.85` is a **private IP address** (10.x.x.x range). This means:
- ✅ Works on the **same local network** (same WiFi/router)
- ❌ **Does NOT work** from different networks (different WiFi, internet, etc.)

## Solutions

### Option 1: Tailscale (Recommended - VPN Solution)

**Best for**: Permanent, secure remote access

1. **Install Tailscale on Windows:**
   ```powershell
   winget install Tailscale.Tailscale
   ```

2. **Install Tailscale on Mac:**
   ```bash
   brew install tailscale
   # Or download from: https://tailscale.com/download
   ```

3. **Sign in on both machines** with the same account

4. **Connect using Tailscale IP:**
   ```bash
   # Find Tailscale IP on Windows:
   tailscale ip
   
   # Then connect from Mac:
   ssh fyp@<tailscale-ip>
   ```

**Advantages:**
- ✅ Works from anywhere (internet, different networks)
- ✅ Secure (encrypted VPN)
- ✅ Free for personal use
- ✅ No router configuration needed

---

### Option 2: ngrok (TCP Tunnel - Quick Solution)

**Best for**: Temporary access, testing

1. **Install ngrok** (if not already installed):
   ```powershell
   winget install ngrok
   ```

2. **Authenticate** (if not done):
   ```powershell
   ngrok config add-authtoken YOUR_TOKEN
   ```

3. **Create TCP tunnel:**
   ```powershell
   ngrok tcp 22
   ```

4. **Use the ngrok address** from the output:
   ```
   Forwarding  tcp://0.tcp.ngrok.io:12345 -> localhost:22
   ```
   
   Connect from Mac:
   ```bash
   ssh fyp@0.tcp.ngrok.io -p 12345
   ```

**Advantages:**
- ✅ Quick setup
- ✅ Works immediately
- ✅ No installation on Mac needed

**Disadvantages:**
- ❌ Free tier has connection limits
- ❌ Address changes each time (unless paid plan)
- ❌ Less secure than VPN

---

### Option 3: Router Port Forwarding

**Best for**: Permanent access from internet (requires router access)

1. **Find your public IP:**
   ```powershell
   (Invoke-WebRequest -Uri "https://api.ipify.org").Content
   ```

2. **Configure router:**
   - Access router admin panel (usually 192.168.1.1 or 192.168.0.1)
   - Set up port forwarding: External Port 22 → Internal IP 10.119.178.85:22
   - Forward to Windows machine's IP

3. **Connect using public IP:**
   ```bash
   ssh fyp@<your-public-ip>
   ```

**Advantages:**
- ✅ Direct connection
- ✅ No third-party service

**Disadvantages:**
- ❌ Requires router access
- ❌ Security risk (exposes SSH to internet)
- ❌ Public IP may change (unless static)
- ❌ Not recommended for security reasons

---

### Option 4: Same Network Connection

**Best for**: Both machines on same WiFi/router

If both Mac and Windows are on the **same network**, try:

1. **Verify both are on same network:**
   - Windows: `10.119.178.85`
   - Mac: Should be `10.119.178.x` (same subnet)

2. **Check Mac can ping Windows:**
   ```bash
   ping 10.119.178.85
   ```

3. **If ping works but SSH doesn't:**
   - Check Windows Firewall (should be OK based on our config)
   - Check router firewall settings
   - Try disabling Windows Firewall temporarily to test

---

## Recommended: Tailscale Setup

Let's set up Tailscale for you:

1. **On Windows**, run:
   ```powershell
   winget install Tailscale.Tailscale
   ```

2. **After installation**, sign in with your account

3. **On Mac**, install and sign in with same account

4. **Connect** using the Tailscale IP shown in the Windows Tailscale app

---

## Quick Test: Verify Network Connectivity

**On Mac**, test if you can reach Windows:
```bash
# Test ping
ping 10.119.178.85

# Test port 22
nc -zv 10.119.178.85 22
# or
telnet 10.119.178.85 22
```

If these fail, the machines are likely on different networks.

---

## Current Status

✅ SSH Server: Running  
✅ Port 22: Listening  
✅ Firewall: Configured  
✅ Password Auth: Enabled  
❌ Remote Connection: Timeout (likely network issue)

**Next Step**: Choose one of the solutions above. **Tailscale is recommended** for secure, permanent remote access.

