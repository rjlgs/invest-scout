"""
Email alert module for invest-scout.

Sends an HTML digest of significant signals via SMTP. Disabled by default —
set EMAIL_ENABLED=true in .env along with SMTP credentials to activate.
"""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("invest-scout.alerts")

# A signal score of 3+ out of 5 is considered "significant" enough to alert
ALERT_THRESHOLD = 3


def _is_enabled() -> bool:
    """Check whether email alerts are turned on."""
    return os.getenv("EMAIL_ENABLED", "false").lower() in ("true", "1", "yes")


def _build_html(results: list[dict]) -> str:
    """Build an HTML email body from significant signal results."""
    significant = [
        r for r in results
        if r.get("on_ramp_score", 0) >= ALERT_THRESHOLD
        or r.get("off_ramp_score", 0) >= ALERT_THRESHOLD
    ]

    if not significant:
        return ""

    rows = ""
    for r in significant:
        color = "#4CAF50" if r["signal_type"] == "on-ramp" else "#F44336"
        signals = r.get("on_ramp_signals", []) + r.get("off_ramp_signals", [])
        signal_list = "<br>".join(s["name"] for s in signals)

        rows += f"""
        <tr>
            <td style="padding:8px;border:1px solid #ddd"><b>{r['ticker']}</b></td>
            <td style="padding:8px;border:1px solid #ddd">${r['price']}</td>
            <td style="padding:8px;border:1px solid #ddd;color:{color}">
                {r['signal_type'].upper()}
            </td>
            <td style="padding:8px;border:1px solid #ddd">
                {r['on_ramp_score']}/{r['off_ramp_score']}
            </td>
            <td style="padding:8px;border:1px solid #ddd;font-size:0.9em">
                {signal_list}
            </td>
        </tr>"""

    return f"""
    <html><body>
    <h2>Invest-Scout Signal Alert</h2>
    <p>The following tickers have significant signal activity:</p>
    <table style="border-collapse:collapse;width:100%">
        <tr style="background:#f5f5f5">
            <th style="padding:8px;border:1px solid #ddd">Ticker</th>
            <th style="padding:8px;border:1px solid #ddd">Price</th>
            <th style="padding:8px;border:1px solid #ddd">Signal</th>
            <th style="padding:8px;border:1px solid #ddd">Score (On/Off)</th>
            <th style="padding:8px;border:1px solid #ddd">Active Signals</th>
        </tr>
        {rows}
    </table>
    <p style="color:#999;font-size:0.8em">
        This is for informational purposes only — not financial advice.
    </p>
    </body></html>
    """


def send_email_alert(results: list[dict]) -> None:
    """
    Send an email digest if alerts are enabled and significant signals exist.

    Does nothing if EMAIL_ENABLED is false or no tickers have scores >= 3.
    """
    if not _is_enabled():
        return

    html = _build_html(results)
    if not html:
        logger.info("No significant signals to alert on.")
        return

    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "")
    smtp_pass = os.getenv("SMTP_PASS", "")
    email_to = os.getenv("EMAIL_TO", "")

    if not all([smtp_user, smtp_pass, email_to]):
        logger.warning(
            "Email enabled but SMTP credentials incomplete. "
            "Set SMTP_USER, SMTP_PASS, and EMAIL_TO in .env."
        )
        return

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "Invest-Scout Signal Alert"
    msg["From"] = smtp_user
    msg["To"] = email_to
    msg.attach(MIMEText(html, "html"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.send_message(msg)
        logger.info("Alert email sent to %s", email_to)
    except Exception as e:
        logger.error("Failed to send alert email: %s", e)
        raise
