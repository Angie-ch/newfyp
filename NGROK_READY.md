# ngrok Setup Complete! ✅

## Status
- ✅ Authtoken: Added successfully
- ✅ ngrok Tunnel: Should be running

## How to Connect

### Step 1: Find the ngrok Address

Look at the **ngrok PowerShell window** that opened. You should see a line like:

```
Forwarding  tcp://0.tcp.ngrok.io:12345 -> localhost:22
```

**Copy these two parts:**
- **Hostname**: `0.tcp.ngrok.io` (yours will be different)
- **Port**: `12345` (yours will be different)

### Step 2: Connect in Cursor

1. **Open Cursor on your Mac**
2. **Press** `Cmd+Shift+P` (or `Ctrl+Shift+P`)
3. **Type**: "Remote-SSH: Connect to Host"
4. **Enter**: `fyp@0.tcp.ngrok.io` (use YOUR hostname)
5. **When prompted for port**, enter: `12345` (use YOUR port)

### Step 3: Keep ngrok Running

⚠️ **Important**: Keep the ngrok window open while using SSH!

If you close it, the connection will break.

---

## Troubleshooting

### Can't find ngrok window?
- Check your taskbar for a PowerShell window
- Or restart manually:
  ```powershell
  ngrok tcp 22
  ```

### Connection still fails?
- Make sure ngrok window is still open
- Verify the address is correct (check for typos)
- Try restarting ngrok: Close window, then `ngrok tcp 22` again

### ngrok address changed?
- This is normal - ngrok free tier gives new addresses each time
- For permanent access, consider Tailscale instead

---

## Quick Commands

### Check if ngrok is running:
```powershell
Get-Process -Name ngrok
```

### Restart ngrok:
```powershell
ngrok tcp 22
```

### Check SSH status:
```powershell
Get-Service sshd
netstat -an | findstr ":22"
```

---

## Alternative: Tailscale (For Permanent Access)

If you want a permanent IP that doesn't change:

1. **Windows**: `winget install Tailscale.Tailscale`
2. **Mac**: `brew install tailscale`
3. Sign in on both with same account
4. Connect using Tailscale IP

Tailscale is more reliable and doesn't require keeping a window open!

