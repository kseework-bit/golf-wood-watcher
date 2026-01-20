import os
import re
import json
import smtplib
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from dateutil.parser import parse as dtparse


UTC = timezone.utc


@dataclass
class Listing:
    source: str
    title: str
    url: str
    price: float
    currency: str = "USD"
    condition: str = ""
    location: str = ""
    posted_at: str = ""  # ISO8601
    club_type: str = ""  # 3W / 5W
    model: str = ""      # MAX / MAX D
    hand: str = ""
    flex: str = ""
    shafted: bool = True
    market: str = "national"  # local / national
    score_tag: str = ""       # Good / Great / Must-buy
    why: str = ""


def load_config() -> dict:
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_baseline() -> dict:
    with open("baseline.json", "r", encoding="utf-8") as f:
        return json.load(f)


def save_baseline(data: dict) -> None:
    with open("baseline.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def contains_any(text: str, terms: List[str]) -> bool:
    t = (text or "").lower()
    return any(term.lower() in t for term in terms)


def extract_price(text: str) -> Optional[float]:
    if not text:
        return None
    # find first currency-ish number
    m = re.search(r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)", text.replace("$", ""))
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def classify_variant(title: str) -> Tuple[str, str]:
    """Return (model, club_type). Model = 'MAX' or 'MAX D'. club_type = '3W'/'5W'."""
    t = title.lower()

    model = ""
    if "max d" in t or "maxd" in t:
        model = "MAX D"
    elif "max" in t:
        model = "MAX"

    club_type = ""
    if re.search(r"\b3w\b", t) or "3 wood" in t or "3-wood" in t:
        club_type = "3W"
    if re.search(r"\b5w\b", t) or "5 wood" in t or "5-wood" in t:
        # if both appear, keep the first match preference by order; but usually only one appears
        club_type = club_type or "5W"

    return model, club_type


def extract_hand_flex(title: str, extra_text: str = "") -> Tuple[str, str]:
    t = (title + " " + (extra_text or "")).lower()

    # Hand
    hand = ""
    if ("right" in t or re.search(r"\brh\b", t)) and not ("left" in t or re.search(r"\blh\b", t)):
        hand = "RH"

    # Flex (Regular)
    flex = ""
    regular_patterns = [
        r"\bregular\b",
        r"\breg\b",
        r"\br-flex\b",
        r"\br flex\b",
        r"\bflex:\s*r\b",
        r"\bflex\s*r\b",
    ]
    if any(re.search(p, t) for p in regular_patterns):
        flex = "Regular"

    # Explicitly reject Stiff/X if present (safety)
    if re.search(r"\bstiff\b|\bx-stiff\b|\bextra stiff\b|\bxs\b|\bx\b", t):
        flex = ""

    return hand, flex



def is_shafted_only_ok(title: str, reject_terms: List[str]) -> bool:
    # Hard reject head-only / shaft-only terms
    return not contains_any(title, reject_terms)


def is_target_listing(cfg: dict, listing: Listing) -> Tuple[bool, str]:
    """Apply hard filters; return (accepted, reason)."""
    cap = cfg["price_cap"]

    if listing.price is None or listing.price > cap:
        return False, f"Price over cap (${cap})"

    title_l = listing.title.lower()

    # must include Paradym + Ai Smoke
    if "paradym" not in title_l or ("ai smoke" not in title_l and "aismoke" not in title_l):
        return False, "Not Paradym Ai Smoke"

    # model and club type
    model, club_type = classify_variant(listing.title)
    if model not in ["MAX", "MAX D"]:
        return False, "Not MAX/MAX D"
    if club_type not in ["3W", "5W"]:
        return False, "Not 3W/5W"

    # shafted only & reject terms
    if not is_shafted_only_ok(listing.title, cfg["reject_terms"]):
        return False, "Rejected term (head-only/shaft-only/etc.)"

    # condition acceptance differs for CPO
    cond_l = (listing.condition or "").lower()
    if listing.source == "callaway_cpo":
        if not any(term in cond_l for term in cfg["cpo_accept_condition_terms"]):
            return False, "CPO condition not Like New/Very Good"
    else:
        if not any(term in cond_l for term in cfg["accept_condition_terms"]):
            return False, "Condition not Mint/Like New/Excellent/Used Demo"

    # hand + flex
    if listing.hand != "RH":
        return False, "Not RH"
    if listing.flex != "Regular":
        return False, "Not Regular flex"

    listing.model = model
    listing.club_type = club_type
    listing.shafted = True
    return True, "Accepted"


# ----------------------------
# Sources
# ----------------------------

def fetch_ebay_rss(cfg: dict) -> List[Listing]:
    """
    Uses eBay search RSS (reliable and lightweight).
    You should create saved searches in eBay and paste their RSS URLs below.
    """
    # Put your RSS URLs here (4 searches: MAX 3W, MAX 5W, MAX D 3W, MAX D 5W)
    rss_urls = [
  "https://www.ebay.com/sch/i.html?_nkw=callaway+paradigm+ai+smoke+max+3+wood&_sacat=0&_from=R40&_trksid=p2334524.m570.l1311&_odkw=Callaway+Paradym+Ai+Smoke+Max&_osacat=0&_rss=1",
  "https://www.ebay.com/sch/i.html?_nkw=callaway+paradigm+ai+smoke+max+5+wood&_sacat=0&_from=R40&_trksid=p2334524.m570.l1313&_odkw=callaway+paradigm+ai+smoke+max+3+wood&_osacat=0&_rss=1",
  "https://www.ebay.com/sch/i.html?_nkw=callaway+paradigm+ai+smoke+max+d+3+wood&_sacat=0&_from=R40&_trksid=p2334524.m570.l1313&_odkw=callaway+paradigm+ai+smoke+max+d+5+wood&_osacat=0&_rss=1",
  "https://www.ebay.com/sch/i.html?_nkw=callaway+paradigm+ai+smoke+max+d+5+wood&_sacat=0&_from=R40&_trksid=p2334524.m570.l1313&_odkw=callaway+paradigm+ai+smoke+max+5+wood&_osacat=0&_rss=1"
        # Example placeholder:
        # "https://www.ebay.com/sch/i.html?_nkw=...&rt=nc&_rss=1"
    ]

    out: List[Listing] = []
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; GolfWatcher/1.0)"})

    for url in rss_urls:
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "xml")
            for item in soup.find_all("item"):
                title = norm_text(item.title.text if item.title else "")
                link = norm_text(item.link.text if item.link else "")
                desc = item.description.text if item.description else ""

                # price often appears in description as "$123.45"
                price = extract_price(desc) or extract_price(title)
                if price is None:
                    continue

                # Basic extraction
                hand, flex = extract_hand_flex(title, desc)
                condition = "like new" if "like new" in desc.lower() else "excellent" if "excellent" in desc.lower() else ""

                out.append(Listing(
                    source="ebay",
                    title=title,
                    url=link,
                    price=float(price),
                    condition=condition or "unknown",
                    location="",
                    posted_at=now_iso(),
                    hand=hand,
                    flex=flex,
                    market="national"
                ))
        except Exception as e:
            print(f"[eBay RSS] Failed for {url}: {e}")

    return out


def fetch_callaway_cpo(cfg: dict) -> List[Listing]:
    """
    Callaway Pre-Owned:
    This adapter is intentionally conservative: it looks for listing tiles and parses title/price/condition.
    Markup can change; if it breaks, the run still emails you.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0 (compatible; GolfWatcher/1.0)"})

    # Start pages: you can refine these URLs to the exact Ai Smoke fairway category pages you prefer.
    urls = [
        "https://www.callawaygolfpreowned.com/fairway-woods/",
    ]

    out: List[Listing] = []
    for url in urls:
        try:
            r = session.get(url, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "lxml")

            # Heuristic: product tiles often include an <a> with product name and a price nearby
            for a in soup.select("a[href]"):
                title = norm_text(a.get_text(" "))
                href = a.get("href", "")
                if not title or "ai smoke" not in title.lower() or "paradym" not in title.lower():
                    continue

                # Try to find a nearby price
                parent_text = norm_text(a.parent.get_text(" ")) if a.parent else title
                price = extract_price(parent_text)
                if price is None:
                    continue

                full_url = href if href.startswith("http") else ("https://www.callawaygolfpreowned.com" + href)

                # Condition: try to infer from surrounding text (CPO usually shows it)
                cond = "very good" if "very good" in parent_text.lower() else "like new" if "like new" in parent_text.lower() else ""

                # Hand/flex sometimes not in tile; we enforce later — most CPO pages include details, but this keeps it lightweight
                hand, flex = extract_hand_flex(title, parent_text)

                out.append(Listing(
                    source="callaway_cpo",
                    title=title,
                    url=full_url,
                    price=float(price),
                    condition=cond or "unknown",
                    location="",
                    posted_at=now_iso(),
                    hand=hand,
                    flex=flex,
                    market="national"
                ))
        except Exception as e:
            print(f"[CPO] Failed for {url}: {e}")

    return out


def fetch_placeholder(name: str) -> List[Listing]:
    # Placeholder: adapters for DICK'S / Golf Galaxy / Play It Again can be added here.
    # Keeping this no-op makes the system stable even before those adapters exist.
    print(f"[{name}] Adapter not implemented yet (placeholder).")
    return []


# ----------------------------
# Baseline & scoring
# ----------------------------

def prune_history(history: List[dict], window_days: int) -> List[dict]:
    cutoff = datetime.now(tz=UTC) - timedelta(days=window_days)
    pruned = []
    for item in history:
        try:
            ts = dtparse(item.get("posted_at", "")).astimezone(UTC)
            if ts >= cutoff:
                pruned.append(item)
        except Exception:
            # If timestamp is bad, drop it
            continue
    return pruned


def compute_medians(history: List[dict]) -> Dict[str, Dict[str, float]]:
    """
    Returns dict:
      key = "<model>|<club_type>|<market>"
      values = { "median": x, "p25": y }
    """
    buckets: Dict[str, List[float]] = {}
    for h in history:
        key = f"{h.get('model','')}|{h.get('club_type','')}|{h.get('market','national')}"
        p = h.get("price")
        if isinstance(p, (int, float)):
            buckets.setdefault(key, []).append(float(p))

    stats_out: Dict[str, Dict[str, float]] = {}
    for k, prices in buckets.items():
        if len(prices) < 3:
            continue
        prices_sorted = sorted(prices)
        median = statistics.median(prices_sorted)
        # p25
        idx = int(0.25 * (len(prices_sorted) - 1))
        p25 = prices_sorted[idx]
        stats_out[k] = {"median": float(median), "p25": float(p25), "n": float(len(prices_sorted))}
    return stats_out


def pick_baseline(cfg: dict, stats: dict, model: str, club_type: str, market: str) -> Tuple[Optional[float], str]:
    """
    Returns (median, context_label).
    Local listings use local median if enough samples else fall back to national.
    National listings use national median.
    """
    local_key = f"{model}|{club_type}|local"
    nat_key = f"{model}|{club_type}|national"

    if market == "local":
        local = stats.get(local_key)
        if local and local.get("n", 0) >= cfg["min_local_samples"]:
            return local["median"], "Local baseline"
        nat = stats.get(nat_key)
        if nat:
            return nat["median"], "National baseline (fallback)"
        return None, "No baseline"
    else:
        nat = stats.get(nat_key)
        if nat:
            return nat["median"], "National baseline"
        # fallback to local if somehow only local exists
        local = stats.get(local_key)
        if local:
            return local["median"], "Local baseline (fallback)"
        return None, "No baseline"


def tag_deal(cfg: dict, listing: Listing, median: Optional[float]) -> Tuple[str, str]:
    """
    Returns (tag, why).
    Uses local adjustment: effective_price = P * 1.05 for local comparisons.
    """
    if median is None or median <= 0:
        return "Unscored", "Baseline unavailable"

    effective_price = listing.price
    if listing.market == "local":
        effective_price = listing.price * float(cfg["local_price_adjustment"])

    ratio = effective_price / median
    pct_under = (1 - ratio) * 100

    if ratio < 0.80:
        return "Must-buy", f"{pct_under:.0f}% under median ({listing.market})"
    if ratio < 0.90:
        return "Great", f"{pct_under:.0f}% under median ({listing.market})"
    if ratio <= 1.00 and ratio >= 0.90:
        return "Good", f"Within market (~{pct_under:.0f}% under) ({listing.market})"
    return "Over-market", f"{abs(pct_under):.0f}% over median ({listing.market})"


# ----------------------------
# Email
# ----------------------------

def send_email(subject: str, body_text: str) -> None:
    host = os.environ["SMTP_HOST"]
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ["SMTP_USER"]
    password = os.environ["SMTP_PASS"]
    to_addr = os.environ.get("EMAIL_TO") or user

    msg = MIMEMultipart()
    msg["From"] = user
    msg["To"] = to_addr
    msg["Subject"] = subject
    msg.attach(MIMEText(body_text, "plain"))

    with smtplib.SMTP(host, port) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(user, [to_addr], msg.as_string())


def format_email(listings: List[Listing], baseline_summary: Dict[str, float]) -> Tuple[str, str]:
    subject = "Daily Golf Club Finds — Paradym Ai Smoke MAX / MAX D (≤ $220)"
    lines = []
    lines.append("Daily Golf Club Monitoring Report")
    lines.append("Criteria: Paradym Ai Smoke MAX / MAX D • 3W / 5W • RH • Regular flex • Shafted only • ≤ $220")
    lines.append("")

    if not listings:
        lines.append("No qualifying listings found today.")
        lines.append("")
    else:
        # Sort by tag importance then price
        rank = {"Must-buy": 0, "Great": 1, "Good": 2, "Unscored": 3, "Over-market": 4}
        listings = sorted(listings, key=lambda x: (rank.get(x.score_tag, 9), x.price))

        sections = [("Must-buy", "🔥 MUST-BUY"), ("Great", "🔵 GREAT DEAL"), ("Good", "🟢 GOOD (Within Market)")]
        for tag, header in sections:
            chunk = [l for l in listings if l.score_tag == tag]
            if not chunk:
                continue
            lines.append(header)
            for l in chunk:
                lines.append(f"- {l.model} {l.club_type} — ${l.price:.0f} — {l.condition} — {l.source}")
                lines.append(f"  Shaft/Flex: {l.flex} | Hand: {l.hand} | Market: {l.market}")
                if l.location:
                    lines.append(f"  Location: {l.location}")
                lines.append(f"  Why: {l.why}")
                lines.append(f"  Link: {l.url}")
            lines.append("")

    # Baseline summary (optional but useful)
    if baseline_summary:
        lines.append("Current Market Baseline (Rolling Median)")
        for k, v in baseline_summary.items():
            lines.append(f"- {k}: ${v:.0f}")
        lines.append("")

    lines.append("—")
    lines.append("Generated automatically.")
    return subject, "\n".join(lines)


# ----------------------------
# Main
# ----------------------------

def main():
    cfg = load_config()
    base = load_baseline()
    history = base.get("accepted_listings", [])
    history = prune_history(history, cfg["window_days"])

    # Fetch listings
    candidates: List[Listing] = []
    candidates += fetch_ebay_rss(cfg)
    candidates += fetch_callaway_cpo(cfg)

    # Placeholders for future adapters
    candidates += fetch_placeholder("DICKS_USED_DEMO")
    candidates += fetch_placeholder("GOLF_GALAXY_USED")
    candidates += fetch_placeholder("PLAY_IT_AGAIN")
    # NOTE: FB Marketplace automation is often brittle; consider manual saved searches or a separate Playwright runner.

    accepted: List[Listing] = []
    for c in candidates:
        # Force local vs national classification (placeholder):
        # If you implement FB/PlayItAgain-local extraction later, set listing.market="local" and listing.location accordingly.
        c.model, c.club_type = classify_variant(c.title)

        ok, reason = is_target_listing(cfg, c)
        if not ok:
            continue

        # Record for baseline
        accepted.append(c)

    # Update history with accepted (only those that pass all filters)
    for a in accepted:
        history.append(asdict(a))

    # Recompute stats after update
    stats = compute_medians(history)

    # Score + tag accepted listings for today's email
    for a in accepted:
        median, ctx = pick_baseline(cfg, stats, a.model, a.club_type, a.market)
        tag, why = tag_deal(cfg, a, median)
        a.score_tag = tag
        a.why = f"{why}; {ctx}"

    # Build baseline summary (national medians only for quick view)
    baseline_summary = {}
    for model in ["MAX", "MAX D"]:
        for club_type in ["3W", "5W"]:
            key = f"{model} {club_type} (National)"
            stat_key = f"{model}|{club_type}|national"
            if stat_key in stats:
                baseline_summary[key] = stats[stat_key]["median"]

    # Email (always)
    subject, body = format_email(accepted, baseline_summary)

    # Save baseline
    base["accepted_listings"] = history
    save_baseline(base)

    # Send
    send_email(subject, body)
    print(f"Sent email to {os.environ.get('EMAIL_TO')} with {len(accepted)} matches.")


if __name__ == "__main__":
    main()
