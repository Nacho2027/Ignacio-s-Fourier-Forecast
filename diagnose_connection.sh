#!/bin/bash

echo "==================================="
echo "EC2 Connection Diagnostic"
echo "==================================="

EC2_IP="18.221.135.150"
KEY_PATH="$HOME/Personal/Ignacio-s-Fourier-Forecast/Fourier.pem"

echo "1. Checking key file permissions..."
if [ -f "$KEY_PATH" ]; then
    ls -la "$KEY_PATH"
    if [ "$(stat -f %A "$KEY_PATH" 2>/dev/null || stat -c %a "$KEY_PATH" 2>/dev/null)" != "400" ]; then
        echo "   ⚠️  Key permissions not 400. Fixing..."
        chmod 400 "$KEY_PATH"
    else
        echo "   ✅ Key permissions correct (400)"
    fi
else
    echo "   ❌ Key file not found at $KEY_PATH"
fi

echo ""
echo "2. Testing network connectivity..."
echo "   Pinging $EC2_IP (may fail if ICMP blocked)..."
ping -c 2 -W 2 $EC2_IP 2>/dev/null || echo "   ℹ️  Ping failed (often blocked by AWS)"

echo ""
echo "3. Testing port 22 (SSH)..."
nc -zv -w 5 $EC2_IP 22 2>&1 || echo "   ❌ Port 22 not reachable"

echo ""
echo "4. Testing port 443 (HTTPS - should work per security group)..."
nc -zv -w 5 $EC2_IP 443 2>&1 || echo "   ℹ️  Port 443 not open (expected)"

echo ""
echo "5. Checking DNS resolution..."
nslookup ec2-18-221-135-150.us-east-2.compute.amazonaws.com

echo ""
echo "6. Traceroute to instance (first 10 hops)..."
traceroute -m 10 -w 2 $EC2_IP 2>/dev/null | head -11 || echo "   ℹ️  Traceroute not available"

echo ""
echo "==================================="
echo "Diagnosis Complete"
echo "==================================="
echo ""
echo "If port 22 is not reachable, the issue is likely:"
echo "1. Security group not allowing your current IP"
echo "2. Network ACL blocking traffic"  
echo "3. Local firewall/ISP blocking outbound SSH"
echo "4. VPC routing issue (no internet gateway)"