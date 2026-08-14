import re
from urllib.parse import urlparse



# -----------------------------
# URL Security Rules
# -----------------------------

SHORTENERS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd"
}



SUSPICIOUS_TLDS = {
    ".xyz",
    ".top",
    ".click",
    ".tk",
    ".ml",
    ".ga",
    ".cf"
}



TRUSTED_DOMAINS = {
    "google.com",
    "microsoft.com",
    "apple.com",
    "amazon.com",
    "paypal.com",
    "gov.bd",
    "edu"
}



SUSPICIOUS_KEYWORDS = {
    "login",
    "verify",
    "verification",
    "password",
    "account",
    "update",
    "confirm",
    "secure",
    "suspended",
    "claim",
    "reward"
}





# -----------------------------
# Extract URLs
# -----------------------------

def extract_urls(text):
    """
    Extract URLs from email text.

    Detects:
    - http://example.com
    - https://example.com
    - www.example.com
    - example.com
    """


    pattern = r"""
    (
        https?://[^\s<>"]+
        |
        www\.[^\s<>"]+
        |
        \b[a-zA-Z0-9.-]+\.(?:com|org|net|gov|edu|bd|io|xyz|top|click|tk|ml|ga|cf)\b[^\s<>"]*
    )
    """


    urls = re.findall(
        pattern,
        text,
        re.VERBOSE
    )


    return urls






# -----------------------------
# Analyze single URL
# -----------------------------

def analyze_single_url(url):


    risk_score = 0

    reasons = []



    # Add protocol if missing

    original_url = url


    if not url.startswith(
        ("http://", "https://")
    ):

        url = "http://" + url





    parsed = urlparse(url)


    domain = parsed.netloc.lower()


    domain = domain.replace(
        "www.",
        ""
    )




    # -----------------------------
    # Trusted domains
    # -----------------------------

    for trusted in TRUSTED_DOMAINS:


        if domain.endswith(trusted):


            risk_score -= 3


            reasons.append(
                "Trusted domain"
            )


            break






    # -----------------------------
    # URL shorteners
    # -----------------------------

    if domain in SHORTENERS:


        risk_score += 3


        reasons.append(
            "URL shortener detected"
        )







    # -----------------------------
    # Suspicious TLD
    # -----------------------------

    for tld in SUSPICIOUS_TLDS:


        if domain.endswith(tld):


            risk_score += 2


            reasons.append(
                "Suspicious domain extension"
            )


            break






    # -----------------------------
    # Suspicious keywords
    # -----------------------------

    for keyword in SUSPICIOUS_KEYWORDS:


        if keyword in url.lower():


            risk_score += 1


            reasons.append(
                f"Contains keyword: {keyword}"
            )







    # -----------------------------
    # Long URL
    # -----------------------------

    if len(url) > 100:


        risk_score += 1


        reasons.append(
            "Very long URL"
        )







    # -----------------------------
    # IP address URL
    # -----------------------------

    if re.match(
        r"https?://\d+\.\d+\.\d+\.\d+",
        url
    ):


        risk_score += 3


        reasons.append(
            "IP address used instead of domain"
        )







    return {


        "url": original_url,


        "risk_score": risk_score,


        "reasons": reasons

    }







# -----------------------------
# Analyze all URLs
# -----------------------------

def analyze_urls(text):


    urls = extract_urls(text)


    results = []


    total_score = 0



    for url in urls:


        result = analyze_single_url(
            url
        )


        results.append(
            result
        )


        total_score += result["risk_score"]






    return {


        "urls_found": len(urls),


        "total_risk_score": total_score,


        "details": results

    }