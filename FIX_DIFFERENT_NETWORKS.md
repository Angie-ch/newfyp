# Fix: Mac and Windows on Different Networks

## The Problem
- **Mac IP**: `192.168.100.13` (network: 192.168.100.x)
- **Windows IP**: `10.119.178.85` (network: 10.119.178.x)

They're on **completely different networks**, so they can't communicate!

## Solutions

### Solution 1: Connect Both to Same Wi-Fi Network (EASIEST)

**On your Mac:**
1. Click Wi-Fi icon in menu bar
2. Connect to the **same Wi-Fi network** that Windows is using
3. Check your IP again:
   ```bash
   ifconfig | grep "inet "
   ```
4. Should now show `10.119.178.x` (same network as Windows)
5. Try connecting:
   ```bash
   ssh -p 2222 fyp@10.119.178.85
   ```

### Solution 2: Connect Windows to Mac's Network

**On Windows:**
1. Open Settings → Network & Internet → Wi-Fi
2. Connect to the **same Wi-Fi network** your Mac is using (192.168.100.x network)
3. Check Windows IP:
   ```powershell
   ipconfig
   ```
4. Should now show `192.168.100.x`
5. Update connection from Mac:
   ```bash
   ssh -p 2222 fyp@[NEW_WINDOWS_IP]
   ```

### Solution 3: Use Mobile Hotspot

1. **Create hotspot on your phone**
2. **Connect both Mac and Windows to phone hotspot**
3. **Check IPs on both:**
   - Mac: `ifconfig | grep "inet "`
   - Windows: `ipconfig`
4. **They should now be on same network**
5. **Connect from Mac:**
   ```bash
   ssh -p 2222 fyp@[WINDOWS_IP]
   ```

### Solution 4: Use VPN (If Available)

If you have a VPN that both can connect to:
1. Connect both Mac and Windows to same VPN
2. They'll be on same virtual network
3. Use VPN IPs to connect

## Quick Check Commands

**On Mac:**
```bash
# Check current network
ifconfig | grep "inet "

# List available Wi-Fi networks
networksetup -listallhardwareports
```

**On Windows:**
```powershell
# Check current network
ipconfig

# List available Wi-Fi networks
netsh wlan show profiles
```

## After Connecting to Same Network

Once both are on same network:

1. **Check Windows IP** (it might have changed):
   ```powershell
   ipconfig
   ```
   Look for IPv4 Address under your Wi-Fi adapter

2. **Connect from Mac:**
   ```bash
   ssh -p 2222 fyp@[WINDOWS_IP]
   ```

## Most Common Fix

**Just connect both devices to the same Wi-Fi network!**

1. Find out which Wi-Fi network Windows is connected to
2. Connect Mac to that same network
3. Try connecting again

