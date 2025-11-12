# Fix ngrok Authentication Issues

## The Problem
"failed to send authentication request" means ngrok can't reach ngrok servers or authtoken isn't set.

## Solutions

### Solution 1: Update ngrok First

Your version (3.24.0) is outdated. Update it:

**On Windows:**
```powershell
# Update ngrok
ngrok update

# Or download latest from: https://ngrok.com/download
```

### Solution 2: Verify Authtoken is Set

**Check if authtoken is configured:**
```powershell
ngrok config check
```

**If not set, add it:**
```powershell
ngrok config add-authtoken YOUR_TOKEN
```

**Get your token from:**
- https://dashboard.ngrok.com/get-started/your-authtoken
- Make sure you're logged in
- Copy the ENTIRE token (no spaces)

### Solution 3: Check Network/Firewall

ngrok needs to connect to ngrok servers. If firewall is blocking:

**Temporarily test with firewall off:**
```powershell
# Check if Windows Firewall is blocking
Get-NetFirewallProfile | Select-Object Name, Enabled
```

**Or allow ngrok through firewall:**
```powershell
New-NetFirewallRule -DisplayName "ngrok" -Direction Outbound -Program "C:\path\to\ngrok.exe" -Action Allow
```

### Solution 4: Try Different ngrok Server

If your network blocks ngrok, try using a different region:

```powershell
# Use US server
ngrok tcp 2222 --region us

# Or EU server
ngrok tcp 2222 --region eu

# Or AP (Asia Pacific)
ngrok tcp 2222 --region ap
```

### Solution 5: Check ngrok Config File

**View config:**
```powershell
ngrok config check
```

**Or manually check:**
```powershell
notepad $env:APPDATA\ngrok\ngrok.yml
```

Should contain:
```yaml
authtoken: YOUR_TOKEN_HERE
```

### Solution 6: Use Web Interface

ngrok provides a web interface at http://127.0.0.1:4040

1. Open browser: http://127.0.0.1:4040
2. Check for error messages
3. See connection status

## Step-by-Step Fix

### Step 1: Update ngrok
```powershell
ngrok update
```

### Step 2: Get Fresh Authtoken
1. Go to: https://dashboard.ngrok.com/get-started/your-authtoken
2. Make sure you're logged in
3. Copy the token (entire string)

### Step 3: Add Authtoken
```powershell
ngrok config add-authtoken YOUR_TOKEN_HERE
```

### Step 4: Verify
```powershell
ngrok config check
```

Should show your authtoken is configured.

### Step 5: Start Tunnel
```powershell
ngrok tcp 2222 --region us
```

Try different regions if one doesn't work.

## Alternative: Use Tailscale Instead

If ngrok keeps failing, **Tailscale is more reliable**:

1. Download on Windows: https://tailscale.com/download
2. Download on Mac: https://tailscale.com/download
3. Sign up/login on both
4. Connect using Tailscale IP

No authentication issues, works from anywhere!

