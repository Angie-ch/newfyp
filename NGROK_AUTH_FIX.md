# Fix ngrok Authentication Error

## Problem
```
failed to send authentication request: read tcp...
```

This means ngrok needs an authtoken to work.

## Solution: Add Your Authtoken

### Step 1: Get Your Authtoken

1. **Go to ngrok dashboard:**
   - Visit: https://dashboard.ngrok.com/get-started/your-authtoken
   - Or: https://dashboard.ngrok.com → Your Authtoken

2. **Sign up or login:**
   - Create a free account if needed
   - Login to dashboard

3. **Copy your authtoken:**
   - It's a long string like: `2abc123def456ghi789jkl012mno345pqr678stu901vwx234yz`
   - Copy the ENTIRE token

### Step 2: Add Authtoken to ngrok

**On Windows (PowerShell):**

```powershell
ngrok config add-authtoken YOUR_TOKEN_HERE
```

**Replace `YOUR_TOKEN_HERE` with your actual token!**

Example:
```powershell
ngrok config add-authtoken 2abc123def456ghi789jkl012mno345pqr678stu901vwx234yz
```

You should see: `Authtoken saved to configuration file.`

### Step 3: Verify Configuration

```powershell
ngrok config check
```

Should show your authtoken is configured (may still show update_channel warning, but that's OK).

### Step 4: Start Tunnel Again

```powershell
ngrok tcp 22
```

Now it should work! Look for:
```
Forwarding  tcp://0.tcp.ngrok.io:12345 -> localhost:22
```

---

## Alternative: Use Tailscale (Recommended)

If ngrok keeps having issues, **Tailscale is more reliable**:

### On Windows:
```powershell
winget install Tailscale.Tailscale
```
Then open Tailscale app and sign in.

### On Mac:
```bash
brew install tailscale
# Or download from: https://tailscale.com/download
```
Sign in with the same account.

### Connect:
```bash
ssh fyp@<tailscale-ip>
```

**Advantages:**
- ✅ No authentication tokens needed
- ✅ Permanent IP address
- ✅ More reliable
- ✅ Free for personal use

---

## Quick Fix Script

Run:
```powershell
.\fix_ngrok_auth.ps1
```

This will show you step-by-step instructions.

