# How to Connect Without Password Using SSH Keys

## How It Works

SSH key authentication uses a **key pair**:
- **Private Key** (stays on your remote computer - never share this!)
- **Public Key** (goes on Windows machine - safe to share)

When you connect, your private key automatically proves your identity - no password needed!

---

## Step-by-Step Setup

### Step 1: Generate SSH Key on Remote Computer

On your **REMOTE computer** (the one you'll use to connect FROM), open Terminal/PowerShell and run:

```bash
ssh-keygen -t rsa -b 4096
```

**What to do:**
- Press **Enter** to accept default location (`~/.ssh/id_rsa`)
- Press **Enter** for no passphrase (or set one if you want)
- Press **Enter** again to confirm

This creates:
- `~/.ssh/id_rsa` (private key - keep secret!)
- `~/.ssh/id_rsa.pub` (public key - we'll copy this)

### Step 2: Get Your Public Key

Display your public key:

```bash
cat ~/.ssh/id_rsa.pub
```

**Copy the entire output** - it looks like:
```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC... (long string) ... your_email@example.com
```

### Step 3: Add Public Key to Windows

**Option A: I can do it for you**
- Send me the public key output
- I'll add it to Windows automatically

**Option B: Do it manually**

On Windows (this machine), open PowerShell as Administrator:

```powershell
# Create authorized_keys file
notepad C:\Users\fyp\.ssh\authorized_keys
```

Paste your public key into the file, save and close.

Then set permissions:
```powershell
icacls C:\Users\fyp\.ssh\authorized_keys /inheritance:r
icacls C:\Users\fyp\.ssh\authorized_keys /grant "fyp:(R)"
```

### Step 4: Configure SSH Server (if needed)

Make sure SSH allows public key authentication:

```powershell
notepad C:\ProgramData\ssh\sshd_config
```

Ensure these lines are uncommented (no #):
```
PubkeyAuthentication yes
AuthorizedKeysFile .ssh/authorized_keys
```

Restart SSH:
```powershell
Restart-Service sshd
```

### Step 5: Connect!

From your remote computer:

```bash
ssh fyp@10.119.178.85
```

**No password needed!** It will use your private key automatically.

---

## Quick Test

After setup, try connecting:

```bash
ssh fyp@10.119.178.85
```

If it works without asking for a password, you're all set!

---

## Troubleshooting

### Still asks for password?
1. Check permissions: `icacls C:\Users\fyp\.ssh\authorized_keys`
2. Verify key was copied correctly (no extra spaces)
3. Check SSH config: `Get-Content C:\ProgramData\ssh\sshd_config | Select-String Pubkey`

### Connection refused?
1. Check SSH service: `Get-Service sshd`
2. Check firewall: `Get-NetFirewallRule -Name sshd`

---

## Using from Cursor/VS Code

Once SSH keys are set up:

1. Open Cursor/VS Code on remote computer
2. Press `F1` → "Remote-SSH: Connect to Host"
3. Enter: `ssh fyp@10.119.178.85`
4. **No password prompt!** It connects automatically.

