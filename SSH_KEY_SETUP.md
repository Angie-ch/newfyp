# SSH Key Authentication Setup (Passwordless)

Since you don't use a password to log into Windows, let's set up SSH key authentication for passwordless remote access.

## Step 1: Generate SSH Key on Remote Computer

On your **REMOTE computer** (the one you'll use to connect), open Terminal/PowerShell and run:

```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
```

- Press Enter to accept default location (`~/.ssh/id_rsa`)
- Press Enter twice for no passphrase (or set one if you want extra security)
- This creates two files: `id_rsa` (private) and `id_rsa.pub` (public)

## Step 2: Copy Public Key to Windows

### Option A: Using ssh-copy-id (if available)
```bash
ssh-copy-id fyp@10.119.178.85
```

### Option B: Manual Copy (if ssh-copy-id not available)

1. **On remote computer**, display your public key:
```bash
cat ~/.ssh/id_rsa.pub
```

2. **Copy the entire output** (starts with `ssh-rsa` and ends with your email)

3. **On Windows machine** (this computer), create the authorized_keys file:

Open PowerShell as Administrator and run:
```powershell
# Create .ssh directory if it doesn't exist
New-Item -ItemType Directory -Force -Path C:\Users\fyp\.ssh

# Create authorized_keys file (you'll paste your key here)
notepad C:\Users\fyp\.ssh\authorized_keys
```

4. **Paste your public key** into the notepad file
5. **Save and close**

## Step 3: Set Permissions on Windows

In PowerShell as Administrator:
```powershell
# Set correct permissions
icacls C:\Users\fyp\.ssh\authorized_keys /inheritance:r
icacls C:\Users\fyp\.ssh\authorized_keys /grant "fyp:(R)"
```

## Step 4: Configure SSH Server (if needed)

Make sure SSH server allows public key authentication:

```powershell
# Edit SSH config
notepad C:\ProgramData\ssh\sshd_config
```

Ensure these lines are uncommented (remove # if present):
```
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
```

Restart SSH service:
```powershell
Restart-Service sshd
```

## Step 5: Test Connection

From your remote computer:
```bash
ssh fyp@10.119.178.85
```

You should connect **without entering a password**!

## Troubleshooting

### If it still asks for password:
1. Check permissions on `authorized_keys` file
2. Verify the public key was copied correctly (no extra spaces/lines)
3. Check SSH server logs: `Get-EventLog -LogName Application -Source OpenSSH -Newest 10`

### If connection fails:
1. Verify SSH service is running: `Get-Service sshd`
2. Check firewall: `Get-NetFirewallRule -Name sshd`

