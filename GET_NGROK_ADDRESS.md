# How to Get ngrok Address

## If ngrok Window is Open:

1. **Look at the ngrok PowerShell window** that opened
2. **Find the line** that says:
   ```
   Forwarding  tcp://0.tcp.ngrok.io:12345 -> localhost:22
   ```
3. **Copy these parts:**
   - **Host**: `0.tcp.ngrok.io` (the hostname)
   - **Port**: `12345` (the port number - yours will be different!)

## Use in Cursor:

1. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows)
2. Type "Remote-SSH: Connect to Host"
3. Enter: `fyp@0.tcp.ngrok.io`
4. When asked for port, enter: `12345` (use YOUR port number)

## If ngrok Window is Not Visible:

1. Check if ngrok process is running:
   ```powershell
   Get-Process -Name ngrok
   ```

2. If not running, start it manually:
   ```powershell
   ngrok tcp 22
   ```

## Troubleshooting:

- **"Connection refused"**: Make sure SSH service is running
- **"Address not found"**: ngrok may have stopped, restart it
- **Different address each time**: This is normal for free ngrok

---

## Better Solution: Use Tailscale

For permanent access that doesn't change, use Tailscale instead.

See `CONNECTION_GUIDE.md` for Tailscale setup.

