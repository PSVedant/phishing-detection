"""
Generate a balanced, *synthetic* labeled email dataset for demos and tests.

The generator injects realistic noise (stealthy phishing, legitimate emails that
mention passwords or use tracking redirects) so that classifiers do NOT score a
trivial 100%. Replace data/raw/emails.csv with a real corpus for real use.

    python -m src.generate_synthetic_data --n 2000
"""
from __future__ import annotations

import argparse
import logging
import random
from email.message import EmailMessage

import pandas as pd

from src import config

logger = logging.getLogger(__name__)

BRANDS = ["paypal", "amazon", "microsoft", "apple", "netflix", "chase", "dropbox"]
LEGIT_SENDERS = ["github.com", "linkedin.com", "medium.com", "stripe.com", "notion.so",
                 "university.edu", "acme-corp.com", "bbc.co.uk"]
FILLER = ("Thanks for being a valued member of our community. We regularly share product "
          "updates, tips and news about the services you use. Please review the details below "
          "and let us know if you have any questions. ")
LEGIT_SUBJECTS = ["Your monthly newsletter", "Receipt for your purchase", "Meeting notes",
                  "Weekly digest", "Your order has shipped", "Invitation: project kickoff"]
PHISH_SUBJECTS = ["Action required on your account", "Your account has been suspended",
                  "Security alert", "Unusual sign-in detected", "Payment failed", "Final notice"]


def _rand_ip(rng: random.Random) -> str:
    return ".".join(str(rng.randint(11, 223)) for _ in range(4))


def _build(rng: random.Random, phishing: bool) -> str:
    stealth = phishing and rng.random() < 0.15       # phishing that looks benign
    brand = rng.choice(BRANDS)

    if phishing:
        spoof = f"{brand}-{rng.choice(['secure', 'support', 'verify'])}.{rng.choice(['com', 'net', 'info'])}"
        from_dom = spoof if rng.random() < 0.7 else "gmail.com"
        mismatch_hdr = (not stealth) and rng.random() < 0.6
        reply_dom = "freemail-help.net" if mismatch_hdr else from_dom
        use_ip = (not stealth) and rng.random() < 0.25
        scheme_http = (not stealth) and rng.random() < 0.5
        text_mismatch = (not stealth) and rng.random() < 0.55
        n_links = rng.randint(1, 3)
        p_kw, p_urg = (0.25, 0.2) if stealth else (0.85, 0.8)
        p_attach = 0.2
        words = rng.randint(15, 60) if not stealth else rng.randint(40, 150)
        subject = rng.choice(PHISH_SUBJECTS)
    else:
        from_dom = rng.choice(LEGIT_SENDERS)
        reply_dom = "mailer-service.net" if rng.random() < 0.08 else from_dom
        use_ip, scheme_http = rng.random() < 0.01, rng.random() < 0.07
        text_mismatch = rng.random() < 0.07           # tracking redirects
        n_links = rng.randint(0, 4)
        p_kw, p_urg = 0.12, 0.08
        p_attach = 0.15
        words = rng.randint(30, 200)
        subject = rng.choice(LEGIT_SUBJECTS)

    sender_name = brand.title() if phishing else from_dom.split(".")[0].title()
    msg = EmailMessage()
    msg["From"] = f"{sender_name} <noreply@{from_dom}>"
    msg["To"] = "user@example.org"
    msg["Subject"] = subject
    if reply_dom != from_dom:
        msg["Reply-To"] = f"help@{reply_dom}"

    sentences = [FILLER * 1 for _ in range(max(1, words // 25))]
    if rng.random() < p_kw:
        sentences.append(rng.choice([
            "Please verify your account to continue.",
            "Your account has been suspended pending review.",
            "You must confirm your password now.",
            "We could not verify your password; sign in again.",
        ]) if phishing else rng.choice([
            "Use the link to reset your password.",
            "Please verify your email address.",
        ]))
    if rng.random() < p_urg:
        sentences.append(rng.choice([
            "URGENT: act now or your access will expire within 24 hours!",
            "Respond immediately - failure to comply will result in permanent closure!",
            "This is your final notice, please act right away.",
        ]) if phishing else "Offer expires soon, so don't miss out.")
    text = " ".join(sentences)

    anchors_html, plain_urls = [], []
    for _ in range(n_links):
        legit_host = f"www.{brand}.com" if phishing else f"www.{from_dom}"
        target = _rand_ip(rng) if use_ip else (from_dom if phishing else legit_host)
        url = f"{'http' if scheme_http else 'https'}://{target}/{rng.randint(1000, 99999)}"
        label = (f"www.{brand}.com" if phishing else f"www.{from_dom}") if text_mismatch \
            else rng.choice(["Click here", "View details", "Sign in", "Read more"])
        if not text_mismatch and not phishing and rng.random() < 0.3:
            label = f"www.{from_dom}"                  # consistent text/target
            url = f"https://www.{from_dom}/{rng.randint(1000, 99999)}"
        anchors_html.append(f'<a href="{url}">{label}</a>')
        plain_urls.append(url)

    msg.set_content(text + ("\n" + "\n".join(plain_urls) if plain_urls else ""))
    msg.add_alternative(f"<html><body><p>{text}</p><p>{' '.join(anchors_html)}</p></body></html>",
                        subtype="html")
    if rng.random() < p_attach:
        msg.add_attachment(b"dummy", maintype="application", subtype="octet-stream",
                           filename=rng.choice(["invoice.zip", "form.pdf", "report.docx"]))
    return msg.as_string()


def generate(n: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    half = n // 2
    rows = [(_build(rng, True), 1) for _ in range(half)] + \
           [(_build(rng, False), 0) for _ in range(n - half)]
    rng.shuffle(rows)
    return pd.DataFrame(rows, columns=[config.TEXT_COLUMN, config.LABEL_COLUMN])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=2000, help="total emails (balanced)")
    ap.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    ap.add_argument("--out", default=str(config.RAW_DATA_PATH))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = generate(args.n, args.seed)
    config.RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    logger.info("Wrote %d emails to %s (class balance: %s)", len(df), args.out,
                df[config.LABEL_COLUMN].value_counts().to_dict())


if __name__ == "__main__":
    main()
