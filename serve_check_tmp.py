import urllib.request
for p in ['/dashboard', '/can', '/v2x', '/api/health']:
    try:
        r = urllib.request.urlopen('http://127.0.0.1:5000' + p, timeout=20)
        print(p, r.status, 'CSP:', r.headers.get('Content-Security-Policy'))
    except Exception as e:
        print(p, 'WAIT/RETRY:', str(e)[:120])
