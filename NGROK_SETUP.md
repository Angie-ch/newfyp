# ngrok Setup Guide

## The Error
"failed to send authentication request" means ngrok isn't authenticated yet.

## Step-by-Step Setup

### Step 1: Get Your Authtoken

1. **Go to ngrok dashboard:**
   - Visit: https://dashboard.ngrok.com/get-started/your-authtoken
   - Or: https://dashboard.ngrok.com → Your Authtoken

2. **Sign up/Login:**
   - Create free account if needed
   - Login to dashboard

3. **Copy your authtoken:**
   - It's a long string like: `2abc123def456ghi789jkl012mno345pqr678stu901vwx234yz`
   - Copy the entire token

### Step 2: Authenticate ngrok on Windows

**On Windows (PowerShell):**

```powershell
# Replace YOUR_TOKEN with the actual token from dashboard
ngrok config add-authtoken YOUR_TOKEN
```

**Example:**
```powershell
ngrok config add-authtoken 2abc123def456ghi789jkl012mno345pqr678stu901vwx234yz
```

You should see: `Authtoken saved to configuration file.`

### Step 3: Start TCP Tunnel

**On Windows:**

```powershell
ngrok tcp 2222
```

You should see:
```
Forwarding    tcp://0.tcp.ngrok.io:12345 -> localhost:2222
```

**Copy the address!** (e.g., `0.tcp.ngrok.io:12345`)

### Step 4: Connect from Mac

**On your Mac:**

```bash
ssh -p 12345 fyp@0.tcp.ngrok.io
```

Replace `12345` and `0.tcp.ngrok.io` with the actual values from ngrok!

## Full Example

**Windows:**
```powershell
# 1. Authenticate
ngrok config add-authtoken YOUR_TOKEN_HERE

# 2. Start tunnel
ngrok tcp 2222
```

**Output:**
```
Forwarding    tcp://0.tcp.ngrok.io:54321 -> localhost:2222
```

**Mac:**
```bash
ssh -p 54321 fyp@0.tcp.ngrok.io
```

## Troubleshooting

### If "authtoken not found":
- Make sure you copied the ENTIRE token
- No spaces before/after
- Run: `ngrok config add-authtoken YOUR_TOKEN` again

### If connection fails:
- Make sure ngrok is still running on Windows
- Check the forwarding address is correct
- Try restarting ngrok: Press Ctrl+C, then `ngrok tcp 2222` again

### If you get a new address each time:
- That's normal for free ngrok
- Just use the new address shown

## Note

Free ngrok gives you:
- New address each time you restart
- Limited connections
- Good for testing

For permanent access, consider Tailscale instead!

