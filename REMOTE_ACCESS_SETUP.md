# Remote SSH Access Over Internet

Since you can't connect to the same Wi-Fi, you need to access Windows over the internet.

## Solution 1: Use Tailscale (EASIEST - Recommended)

Tailscale creates a secure VPN between your devices automatically.

### On Windows:
1. Download Tailscale: https://tailscale.com/download
2. Install and sign up/login
3. Note the Tailscale IP address (shown in app)

### On Mac:
1. Download Tailscale: https://tailscale.com/download
2. Install and login with same account
3. Connect using Tailscale IP:
   ```bash
   ssh -p 2222 fyp@[TAILSCALE_IP]
   ```

**Advantages:**
- No router configuration needed
- Works from anywhere
- Secure and encrypted
- Free for personal use

---

## Solution 2: Use ngrok (Quick Tunnel)

Creates a temporary tunnel to your Windows machine.

### On Windows:
1. Download ngrok: https://ngrok.com/download
2. Sign up for free account
3. Get your authtoken from dashboard
4. Run:
   ```powershell
   ngrok config add-authtoken YOUR_TOKEN
   ngrok tcp 2222
   ```
5. Copy the forwarding address (e.g., `tcp://0.tcp.ngrok.io:12345`)

### On Mac:
```bash
ssh -p [PORT] fyp@[HOST]
# Example: ssh -p 12345 fyp@0.tcp.ngrok.io
```

**Note:** Free ngrok gives you a new address each time you restart.

---

## Solution 3: Router Port Forwarding (If You Have Router Access)

If someone on-site can access the router:

1. **Access router admin panel** (usually `http://10.119.178.254` based on your gateway)
2. **Find "Port Forwarding" or "Virtual Server"**
3. **Add rule:**
   - External Port: 2222 (or any port)
   - Internal IP: 10.119.178.85
   - Internal Port: 2222
   - Protocol: TCP
4. **Find router's public IP:**
   - Visit: https://whatismyipaddress.com/ from Windows
5. **Connect from Mac:**
   ```bash
   ssh -p 2222 fyp@[ROUTER_PUBLIC_IP]
   ```

**Note:** You'll need the router's public IP, and it may change (dynamic IP).

---

## Solution 4: Use Remote Desktop Gateway

Windows has built-in remote access options, but SSH is better for development.

---

## Solution 5: Use Cloudflare Tunnel (Free)

Similar to ngrok but more permanent.

1. Sign up at Cloudflare
2. Install cloudflared on Windows
3. Create tunnel
4. Connect from Mac using tunnel address

---

## RECOMMENDED: Tailscale

**Why Tailscale:**
- ✅ No router configuration
- ✅ Works from anywhere
- ✅ Secure (encrypted)
- ✅ Free for personal use
- ✅ Easy setup (5 minutes)
- ✅ Permanent IP addresses

### Quick Setup:

**Windows:**
1. Go to https://tailscale.com/download
2. Download Windows version
3. Install and create account
4. Login
5. Note the IP address (e.g., `100.x.x.x`)

**Mac:**
1. Go to https://tailscale.com/download
2. Download macOS version
3. Install and login with same account
4. Connect:
   ```bash
   ssh -p 2222 fyp@[TAILSCALE_IP]
   ```

That's it! Both devices will be on the same virtual network.

---

## Alternative: Use TeamViewer/AnyDesk for Remote Desktop

If you just need remote access (not SSH):
- TeamViewer (free for personal use)
- AnyDesk (free)
- Chrome Remote Desktop (free)

But SSH is better for development work.

