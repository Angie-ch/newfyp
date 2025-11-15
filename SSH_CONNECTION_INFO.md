# SSH Connection Information

## ✅ SSH Server Status
- **Service**: Running
- **Port**: 22 (standard SSH port)
- **Password Authentication**: Enabled

## 🔌 Connection Details

**IP Address**: `10.119.178.85`  
**Username**: `fyp`  
**Port**: `22`

## 📱 How to Connect

### From Mac/Linux Terminal:
```bash
ssh fyp@10.119.178.85
```

### From Cursor (Remote SSH):
1. Open Cursor
2. Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
3. Type "Remote-SSH: Connect to Host"
4. Enter: `fyp@10.119.178.85`
5. Or add to SSH config:
   ```
   Host typhoon-prediction
       HostName 10.119.178.85
       User fyp
       Port 22
   ```

## 🔒 First Connection
- You may be prompted to accept the host key fingerprint
- Enter your Windows password when prompted
- For passwordless login, set up SSH keys (see below)

## 🔑 Optional: Set Up SSH Keys (Passwordless Login)

### On your Mac/Linux:
```bash
# Generate SSH key if you don't have one
ssh-keygen -t rsa -b 4096

# Copy public key to Windows
ssh-copy-id fyp@10.119.178.85
```

### Or manually:
1. Copy your public key (`~/.ssh/id_rsa.pub`) content
2. On Windows, create/edit: `C:\Users\fyp\.ssh\authorized_keys`
3. Paste your public key content
4. Set permissions (run as admin):
   ```powershell
   icacls "C:\Users\fyp\.ssh\authorized_keys" /inheritance:r
   icacls "C:\Users\fyp\.ssh\authorized_keys" /grant "fyp:F"
   ```

## 🛠️ Troubleshooting

### Connection Refused
- Check if SSH service is running: `Get-Service sshd` (on Windows)
- Check firewall: `Get-NetFirewallRule | Where-Object {$_.DisplayName -like "*SSH*"}`

### Permission Denied
- Verify username is correct: `fyp`
- Check password authentication is enabled
- Try using SSH keys instead

### Port 22 Not Accessible
- Check if port 22 is listening: `netstat -an | findstr ":22"`
- Verify firewall allows port 22

## 📝 Notes
- The SSH server is configured to start automatically
- Configuration file: `C:\ProgramData\ssh\sshd_config`
- Logs: `C:\ProgramData\ssh\logs\sshd.log`

