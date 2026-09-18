import re
import math
from urllib.parse import urlparse
import tldextract

SUSPICIOUS_KEYWORDS = [
    'login', 'verify', 'account', 'update', 'banking', 'secure',
    'ebayisapi', 'webscr', 'signin', 'support', 'recover', 'wallet',
    'password', 'auth', 'confirm', 'security', 'alert'
]

def shannon_entropy(data: str) -> float:
    if not data:
        return 0.0
    entropy = 0.0
    for x in set(data):
        p_x = float(data.count(x)) / len(data)
        entropy -= p_x * math.log2(p_x)
    return entropy

def extract_features(url: str) -> dict:
    clean_url = url.strip()
    parsed = urlparse(clean_url if '://' in clean_url else f'http://{clean_url}')
    extracted = tldextract.extract(clean_url)

    hostname = (parsed.netloc or '').lower()
    path = (parsed.path or '').lower()
    full_url = clean_url.lower()

    has_ip = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}', hostname) else 0

    return {
        'url_length': len(full_url),
        'hostname_length': len(hostname),
        'path_length': len(path),
        'has_ip': has_ip,
        'count_dots': full_url.count('.'),
        'count_hyphens': full_url.count('-'),
        'count_at': full_url.count('@'),
        'count_question': full_url.count('?'),
        'count_equal': full_url.count('='),
        'count_slash': full_url.count('/'),
        'count_digits': sum(c.isdigit() for c in full_url),
        'subdomain_count': len(extracted.subdomain.split('.')) if extracted.subdomain else 0,
        'uses_https': 1 if parsed.scheme == 'https' else 0,
        'entropy': round(shannon_entropy(full_url), 4),
        'keyword_match_count': sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in full_url),
        'has_shortener': 1 if any(s in hostname for s in ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'is.gd']) else 0
    }