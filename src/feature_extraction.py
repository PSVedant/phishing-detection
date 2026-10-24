"""
Numerical feature extraction from raw RFC-822 emails.

Feature groups
--------------
URL        : num_links, num_ip_urls, num_http_links, all_links_https
Header     : sender_replyto_mismatch   (From domain != Reply-To domain)
Link       : num_mismatched_links, has_mismatched_link
             (displayed anchor text names one domain, href points to another)
Content    : kw_verify, kw_suspended, kw_password, kw_suspicious_total,
             urgency_score
Structure  : body_length, has_attachment

All extractors are pure functions and never raise on malformed emails: a
broken message yields a zero-ish feature vector instead of crashing the batch.
"""
from __future__ import annotations

import ipaddress
import logging
import re
from dataclasses import dataclass, field
from email import message_from_string, policy
from email.utils import parseaddr
from html.parser import HTMLParser
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_NAMES: List[str] = [
    # URL
    "num_links", "num_ip_urls", "num_http_links", "all_links_https",
    # Header
    "sender_replyto_mismatch",
    # Link
    "num_mismatched_links", "has_mismatched_link",
    # Content
    "kw_verify", "kw_suspended", "kw_password", "kw_suspicious_total",
    "urgency_score",
    # Structure
    "body_length", "has_attachment",
]

# ----------------------------------------------------------------------------
# Lexicons / regexes
# ----------------------------------------------------------------------------
URL_RE = re.compile(r"https?://[^\s<>\"'()\[\]]+", re.IGNORECASE)
DOMAIN_IN_TEXT_RE = re.compile(
    r"(?:https?://)?(?:www\.)?((?:[a-z0-9-]+\.)+[a-z]{2,})", re.IGNORECASE
)

# Required keywords (stem-matched so "verification", "suspension" also count)
KW_VERIFY_RE = re.compile(r"\bverif\w*", re.IGNORECASE)
KW_SUSPENDED_RE = re.compile(r"\bsuspen\w*", re.IGNORECASE)
KW_PASSWORD_RE = re.compile(r"\bpassword\w*", re.IGNORECASE)

# Broader list used for kw_suspicious_total (includes the three above)
SUSPICIOUS_RE = re.compile(
    r"\b(?:verif\w*|suspen\w*|password\w*|login|log\s?in|confirm\w*|"
    r"account|bank\w*|ssn|credential\w*|unauthori[sz]ed|locked|"
    r"security\s+alert|click\s+here|update\s+your)\b",
    re.IGNORECASE,
)

# Urgency lexicon: regex -> weight. Each term is capped at 3 hits.
URGENCY_TERMS: Dict[str, float] = {
    r"\burgent\w*": 2.0,
    r"\bimmediate\w*": 2.0,
    r"\bwithin\s+\d+\s+(?:hours?|days?)\b": 2.5,
    r"\b24\s*hours?\b": 1.5,
    r"\bact\s+now\b": 2.0,
    r"\basap\b": 1.5,
    r"\bfinal\s+(?:notice|warning)\b": 2.5,
    r"\bexpire[sd]?\b": 1.5,
    r"\bright\s+away\b": 1.5,
    r"\blimited\s+time\b": 1.0,
    r"\bfailure\s+to\b": 2.0,
    r"\bpermanently\b": 1.5,
}
_URGENCY_COMPILED = [(re.compile(p, re.IGNORECASE), w) for p, w in URGENCY_TERMS.items()]
_ALLCAPS_RE = re.compile(r"\b[A-Z]{4,}\b")

# Second-level labels that form a 3-label registrable domain (e.g. bbc.co.uk)
_SECOND_LEVEL = {"co", "com", "org", "net", "gov", "ac", "edu"}


# ----------------------------------------------------------------------------
# Email parsing
# ----------------------------------------------------------------------------
class _HTMLLinkParser(HTMLParser):
    """Collects (href, anchor_text) pairs and the visible text of an HTML body."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.anchors: List[Tuple[str, str]] = []
        self.text_parts: List[str] = []
        self._href: Optional[str] = None
        self._buf: List[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "a":
            self._href = dict(attrs).get("href") or ""
            self._buf = []

    def handle_endtag(self, tag):
        tag = tag.lower()
        if tag in ("script", "style") and self._skip:
            self._skip -= 1
        elif tag == "a" and self._href is not None:
            self.anchors.append((self._href, " ".join("".join(self._buf).split())))
            self._href = None

    def handle_data(self, data):
        if self._skip:
            return
        self.text_parts.append(data)
        if self._href is not None:
            self._buf.append(data)


@dataclass
class ParsedEmail:
    from_addr: str = ""
    reply_to: str = ""
    text: str = ""                                   # primary readable body
    anchors: List[Tuple[str, str]] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    has_attachment: bool = False


def _part_text(part) -> str:
    try:
        return part.get_content()
    except Exception:  # unknown charset etc.
        payload = part.get_payload(decode=True) or b""
        return payload.decode("utf-8", errors="ignore")


def parse_email(raw: str) -> ParsedEmail:
    """Parse raw email text into the pieces the feature extractors need."""
    parsed = ParsedEmail()
    try:
        msg = message_from_string(str(raw), policy=policy.default)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Failed to parse email: %s", exc)
        return parsed

    parsed.from_addr = parseaddr(str(msg.get("From", "")))[1]
    parsed.reply_to = parseaddr(str(msg.get("Reply-To", "")))[1]

    plain_parts, html_parts = [], []
    for part in msg.walk():
        if part.is_multipart():
            continue
        if part.get_content_disposition() == "attachment" or part.get_filename():
            parsed.has_attachment = True
            continue
        ctype = part.get_content_type()
        if ctype == "text/plain":
            plain_parts.append(_part_text(part))
        elif ctype == "text/html":
            html_parts.append(_part_text(part))

    html_parser = _HTMLLinkParser()
    for html in html_parts:
        try:
            html_parser.feed(html)
        except Exception:  # malformed HTML
            continue
    html_text = " ".join(" ".join(html_parser.text_parts).split())
    plain_text = "\n".join(plain_parts)

    parsed.text = plain_text if plain_text.strip() else html_text
    parsed.anchors = html_parser.anchors

    urls: List[str] = URL_RE.findall(plain_text + "\n" + html_text)
    urls += [h for h, _ in html_parser.anchors if h.lower().startswith(("http://", "https://"))]
    # de-duplicate, keep order
    parsed.urls = list(dict.fromkeys(u.rstrip(".,;:!?") for u in urls))
    return parsed


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _is_ip(host: str) -> bool:
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def registered_domain(host: str) -> str:
    """Approximate registrable domain (no external public-suffix list needed)."""
    host = (host or "").lower().strip(".")
    if not host or _is_ip(host):
        return host
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if len(labels[-1]) == 2 and labels[-2] in _SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _email_domain(addr: str) -> str:
    return registered_domain(addr.rsplit("@", 1)[1]) if "@" in addr else ""


def _host(url: str) -> str:
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


# ----------------------------------------------------------------------------
# Feature groups
# ----------------------------------------------------------------------------
def url_features(urls: Iterable[str]) -> Dict[str, float]:
    urls = list(urls)
    n_http = sum(1 for u in urls if u.lower().startswith("http://"))
    n_https = sum(1 for u in urls if u.lower().startswith("https://"))
    return {
        "num_links": len(urls),
        "num_ip_urls": sum(1 for u in urls if _is_ip(_host(u))),
        "num_http_links": n_http,
        "all_links_https": int(len(urls) > 0 and n_https == len(urls)),
    }


def header_features(from_addr: str, reply_to: str) -> Dict[str, float]:
    f_dom, r_dom = _email_domain(from_addr), _email_domain(reply_to)
    # No Reply-To header -> nothing to compare -> not a mismatch.
    return {"sender_replyto_mismatch": int(bool(f_dom and r_dom and f_dom != r_dom))}


def link_features(anchors: Iterable[Tuple[str, str]]) -> Dict[str, float]:
    mismatched = 0
    for href, text in anchors:
        if not href.lower().startswith(("http://", "https://")):
            continue
        m = DOMAIN_IN_TEXT_RE.search(text or "")
        if not m:
            continue  # text like "Click here" displays no domain to compare
        if registered_domain(m.group(1)) != registered_domain(_host(href)):
            mismatched += 1
    return {"num_mismatched_links": mismatched, "has_mismatched_link": int(mismatched > 0)}


def urgency_score(text: str) -> float:
    """Weighted urgency lexicon hits + exclamation marks + SHOUTING words."""
    score = sum(w * min(len(rx.findall(text)), 3) for rx, w in _URGENCY_COMPILED)
    score += 0.5 * min(text.count("!"), 5)
    score += 0.25 * min(len(_ALLCAPS_RE.findall(text)), 8)
    return round(score, 3)


def content_features(text: str) -> Dict[str, float]:
    return {
        "kw_verify": len(KW_VERIFY_RE.findall(text)),
        "kw_suspended": len(KW_SUSPENDED_RE.findall(text)),
        "kw_password": len(KW_PASSWORD_RE.findall(text)),
        "kw_suspicious_total": len(SUSPICIOUS_RE.findall(text)),
        "urgency_score": urgency_score(text),
    }


def structure_features(text: str, has_attachment: bool) -> Dict[str, float]:
    return {"body_length": len(text), "has_attachment": int(has_attachment)}


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------
def extract_features(raw_email: str) -> Dict[str, float]:
    """Return the full numeric feature dict for one raw email string."""
    p = parse_email(raw_email)
    feats: Dict[str, float] = {}
    feats.update(url_features(p.urls))
    feats.update(header_features(p.from_addr, p.reply_to))
    feats.update(link_features(p.anchors))
    feats.update(content_features(p.text))
    feats.update(structure_features(p.text, p.has_attachment))
    return {name: feats[name] for name in FEATURE_NAMES}


def extract_feature_frame(emails: Iterable[str]) -> pd.DataFrame:
    """Vectorised helper: iterable of raw emails -> DataFrame[FEATURE_NAMES]."""
    rows = [extract_features(e) for e in emails]
    return pd.DataFrame(rows, columns=FEATURE_NAMES)
