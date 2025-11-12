# Mac Side Troubleshooting

## Windows is Configured Correctly ✅
- SSH listening on port 2222
- Firewall allows port 2222
- Everything set up properly

## The Issue: Network Connectivity

Since Windows is set up correctly, the problem is your Mac can't reach Windows over the network.

## Test Connectivity from Mac

### Step 1: Test if Mac can reach Windows

**On your Mac terminal, run:**
```bash
# Test basic connectivity
ping 10.119.178.85
```

**If ping works:**
- Network is reachable
- Problem is likely router blocking port 2222

**If ping fails:**
- Mac and Windows are on different networks
- Need to connect both to same Wi-Fi

### Step 2: Test if port 2222 is reachable

**On your Mac, run:**
```bash
# Test port connectivity
nc -zv 10.119.178.85 2222
```

**Or:**
```bash
telnet 10.119.178.85 2222
```

**If it says "Connection refused":**
- Port is blocked by router/firewall
- Try different port (see below)

**If it says "Connection timed out":**
- Router is blocking or devices on different networks

### Step 3: Check Mac's IP address

**On your Mac, run:**
```bash
ifconfig | grep "inet "
```

**Check:**
- Is your Mac's IP `10.119.178.x`? (same network as Windows)
- If different (like `192.168.x.x`), you're on different networks!

## Solutions

### Solution 1: Ensure Same Network
- Connect both Mac and Windows to **same Wi-Fi network**
- Not guest network (guest networks are often isolated)
- Check router settings for "AP Isolation" or "Client Isolation" - **DISABLE IT**

### Solution 2: Try Different Ports
If router blocks 2222, try:
- **22222** (less common)
- **22000** (high port, usually allowed)
- **443** (HTTPS port, often allowed)

### Solution 3: Use Mobile Hotspot
- Create hotspot on your phone
- Connect both Mac and Windows to phone hotspot
- This bypasses router restrictions

### Solution 4: Check Router Admin Panel
1. Access router: `http://192.168.1.1` (or check router label)
2. Look for:
   - **AP Isolation** → DISABLE
   - **Client Isolation** → DISABLE
   - **Wireless Isolation** → DISABLE
   - **Firewall** → Check if blocking ports
3. Save and restart router

## Quick Test Commands (Run on Mac)

```bash
# 1. Check Mac IP
ifconfig | grep "inet "

# 2. Ping Windows
ping 10.119.178.85

# 3. Test port 2222
nc -zv 10.119.178.85 2222

# 4. Try SSH with verbose output
ssh -v -p 2222 fyp@10.119.178.85
```

The `-v` flag shows detailed connection info - share the output!

