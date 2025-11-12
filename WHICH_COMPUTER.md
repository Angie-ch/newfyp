# Which Computer Should I Use?

## The Two Computers

1. **THIS Windows Machine** (IP: 10.119.178.85)
   - This is the TARGET (where you want to connect TO)
   - Username: `fyp`
   - SSH server is already set up here ✅

2. **YOUR Remote Computer** (the other computer)
   - This is where you CONNECT FROM
   - This is where you'll run `ssh-keygen`

---

## Step-by-Step: On Your Remote Computer

### Step 1: Open Terminal/PowerShell on Remote Computer

**If your remote computer is:**
- **Mac**: Open "Terminal" (Applications → Utilities → Terminal)
- **Linux**: Open "Terminal" (Ctrl+Alt+T or search for Terminal)
- **Windows**: Open "PowerShell" or "Command Prompt"

### Step 2: Generate SSH Key

Type this command and press Enter:
```bash
ssh-keygen -t rsa -b 4096
```

### Step 3: Press Enter 3 Times

You'll see prompts like:
```
Enter file in which to save the key (/Users/yourname/.ssh/id_rsa):
```
→ **Just press Enter** (don't type anything)

```
Enter passphrase (empty for no passphrase):
```
→ **Just press Enter** (no passphrase)

```
Enter same passphrase again:
```
→ **Just press Enter** (confirm)

### Step 4: Get Your Public Key

Type this command:
```bash
cat ~/.ssh/id_rsa.pub
```

### Step 5: Copy the Output

You'll see something like:
```
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQC... (long string) ... your_email@example.com
```

**Copy the ENTIRE line** (it's one long line)

### Step 6: Send It to Me

Paste the public key here, and I'll add it to the Windows machine.

---

## Visual Guide

```
┌─────────────────────┐         ┌─────────────────────┐
│  REMOTE COMPUTER    │         │  WINDOWS MACHINE    │
│  (Your laptop/etc)  │         │  (10.119.178.85)    │
│                     │         │                     │
│  Run ssh-keygen     │         │  SSH Server ✅      │
│  Get public key     │  ────>  │  (already set up)   │
│  Send to me         │         │                     │
└─────────────────────┘         └─────────────────────┘
```

---

## Quick Check

**Are you on the computer that will connect to Windows?**
- ✅ YES → You're in the right place! Continue with Step 2 above.
- ❌ NO → You need to switch to your remote computer first.

---

## After Setup

Once I add your public key to Windows, you can connect from your remote computer:

```bash
ssh fyp@10.119.178.85
```

No password needed! 🎉

