"""Process-local DNS recovery; HTTPS hostname and certificate checks stay intact."""
import hashlib
import ipaddress
import json
import os
from pathlib import Path
import socket
import subprocess
import threading
import time

if os.environ.get('SCIENTIST_DNS_FALLBACK_ENABLED') == '1':
    source = Path(__file__).resolve()
    expected = os.environ.get('SCIENTIST_DNS_FALLBACK_SHA256')
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise RuntimeError('DNS recovery module hash mismatch')
    original_getaddrinfo = socket.getaddrinfo
    cache = {'until': 0, 'addresses': []}
    cache_lock = threading.Lock()

    def recovered_getaddrinfo(host, *args, **kwargs):
        try:
            return original_getaddrinfo(host, *args, **kwargs)
        except socket.gaierror:
            if host not in ('token4research.cn', b'token4research.cn'):
                raise
            with cache_lock:
                if time.monotonic() >= cache['until']:
                    result = subprocess.run(
                        ['dig', '+short', '+time=2', '+tries=1', '@1.1.1.1', 'token4research.cn', 'A'],
                        capture_output=True, text=True, timeout=5, check=True)
                    addresses = []
                    for line in result.stdout.splitlines():
                        try:
                            address = ipaddress.ip_address(line.strip())
                        except ValueError:
                            continue
                        if address.version == 4 and address.is_global:
                            addresses.append(str(address))
                    if not addresses:
                        raise socket.gaierror('Public DNS recovery returned no IPv4 addresses')
                    cache.update(until=time.monotonic() + 300, addresses=addresses)
                    with (source.parent.parent / 'dns_recovery.jsonl').open('a') as out:
                        out.write(json.dumps({'time': time.time(), 'pid': os.getpid(),
                            'host': 'token4research.cn', 'resolver': '1.1.1.1',
                            'addresses': addresses, 'TLS_hostname_unchanged': True}) + '\n')
                addresses = list(cache['addresses'])
            return sum((original_getaddrinfo(address, *args, **kwargs) for address in addresses), [])

    socket.getaddrinfo = recovered_getaddrinfo
