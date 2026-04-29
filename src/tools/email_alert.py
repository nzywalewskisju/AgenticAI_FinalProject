import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

ALERT_EMAIL = os.getenv("ALERT_EMAIL", "daniel.smith36797@gmail.com")
GMAIL_APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD", "")
FROM_EMAIL = ALERT_EMAIL


def send_alert_email(
    subject: str,
    username: str,
    query: str,
    reason: str,
    alert_type: str,
    timestamp: str = None
) -> bool:
    """
    Sends an alert email to the HR/security inbox.
    alert_type: "security" or "escalation"
    Returns True if sent successfully, False otherwise.
    """
    if not GMAIL_APP_PASSWORD:
        print(f"[EMAIL] No Gmail app password configured — skipping email alert")
        return False

    if timestamp is None:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    type_label = "Security Alert" if alert_type == "security" else "HR Escalation"
    color = "#c0392b" if alert_type == "security" else "#e67e22"

    body_html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
        <div style="background: {color}; padding: 16px 24px; border-radius: 6px 6px 0 0;">
            <h2 style="color: white; margin: 0; font-size: 18px;">
                {type_label} — Nexarion HR Assistant
            </h2>
        </div>
        <div style="border: 1px solid #ddd; border-top: none; padding: 24px; border-radius: 0 0 6px 6px;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <td style="padding: 8px 0; color: #666; width: 140px; vertical-align: top;">
                        <strong>Timestamp</strong>
                    </td>
                    <td style="padding: 8px 0; color: #333;">
                        {timestamp}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #666; vertical-align: top;">
                        <strong>User</strong>
                    </td>
                    <td style="padding: 8px 0; color: #333;">
                        {username}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #666; vertical-align: top;">
                        <strong>Query</strong>
                    </td>
                    <td style="padding: 8px 0; color: #333;">
                        {query}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #666; vertical-align: top;">
                        <strong>Reason</strong>
                    </td>
                    <td style="padding: 8px 0; color: #333;">
                        {reason}
                    </td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: #666; vertical-align: top;">
                        <strong>Alert type</strong>
                    </td>
                    <td style="padding: 8px 0; color: #333;">
                        {type_label}
                    </td>
                </tr>
            </table>
            <hr style="border: none; border-top: 1px solid #eee; margin: 20px 0;">
            <p style="color: #999; font-size: 12px; margin: 0;">
                This alert was generated automatically by the Nexarion HR Policy Assistant.
            </p>
        </div>
    </div>
    """

    body_text = (
        f"{type_label} — Nexarion HR Assistant\n\n"
        f"Timestamp: {timestamp}\n"
        f"User: {username}\n"
        f"Query: {query}\n"
        f"Reason: {reason}\n"
        f"Alert type: {type_label}\n"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[{type_label}] Nexarion HR Assistant — {username}"
    msg["From"] = FROM_EMAIL
    msg["To"] = ALERT_EMAIL
    msg.attach(MIMEText(body_text, "plain"))
    msg.attach(MIMEText(body_html, "html"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(FROM_EMAIL, GMAIL_APP_PASSWORD)
            server.sendmail(FROM_EMAIL, ALERT_EMAIL, msg.as_string())
        print(f"[EMAIL] Alert sent — type={alert_type} user={username}")
        return True
    except Exception as e:
        print(f"[EMAIL] Failed to send alert: {e}")
        return False