from email.message import EmailMessage

from src.feature_extraction import FEATURE_NAMES, extract_features, registered_domain


def _mail(html="", plain="", from_="a@bank.com", reply=None, attach=False):
    m = EmailMessage()
    m["From"] = from_
    m["To"] = "u@x.org"
    if reply:
        m["Reply-To"] = reply
    m.set_content(plain or "hello")
    if html:
        m.add_alternative(html, subtype="html")
    if attach:
        m.add_attachment(b"x", maintype="application", subtype="octet-stream", filename="a.zip")
    return m.as_string()


def test_feature_vector_is_complete():
    assert list(extract_features(_mail()).keys()) == FEATURE_NAMES


def test_registered_domain():
    assert registered_domain("mail.bbc.co.uk") == "bbc.co.uk"
    assert registered_domain("a.b.paypal.com") == "paypal.com"
    assert registered_domain("192.168.0.1") == "192.168.0.1"


def test_url_features():
    f = extract_features(_mail(plain="go http://192.168.1.5/x and https://ok.com/y"))
    assert f["num_links"] == 2 and f["num_ip_urls"] == 1
    assert f["num_http_links"] == 1 and f["all_links_https"] == 0


def test_header_mismatch():
    assert extract_features(_mail(from_="a@paypal.com", reply="b@evil.net"))["sender_replyto_mismatch"] == 1
    assert extract_features(_mail(from_="a@paypal.com", reply="b@mail.paypal.com"))["sender_replyto_mismatch"] == 0
    assert extract_features(_mail())["sender_replyto_mismatch"] == 0


def test_link_text_mismatch():
    bad = _mail(html='<a href="http://evil.com/x">www.paypal.com</a>')
    ok = _mail(html='<a href="https://www.paypal.com/x">www.paypal.com</a>')
    generic = _mail(html='<a href="http://evil.com/x">Click here</a>')
    assert extract_features(bad)["has_mismatched_link"] == 1
    assert extract_features(ok)["has_mismatched_link"] == 0
    assert extract_features(generic)["has_mismatched_link"] == 0


def test_content_and_structure():
    f = extract_features(_mail(plain="URGENT! Verify your password or it is suspended within 24 hours!",
                               attach=True))
    assert f["kw_verify"] == 1 and f["kw_password"] == 1 and f["kw_suspended"] == 1
    assert f["urgency_score"] > 0 and f["has_attachment"] == 1 and f["body_length"] > 0


def test_malformed_input_does_not_crash():
    assert len(extract_features("")) == len(FEATURE_NAMES)
    assert len(extract_features("\x00garbage<<<")) == len(FEATURE_NAMES)
