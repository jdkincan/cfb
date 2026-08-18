"""Send the readout over SMTP.

Defaults target Gmail with an app password, which needs no third-party service
and no billing relationship. Any SMTP host works.

Required environment (set these as GitHub Actions repository secrets):

    SMTP_HOST      smtp.gmail.com
    SMTP_PORT      587
    SMTP_USER      you@gmail.com
    SMTP_PASSWORD  a Google *app password*, not your account password
    EMAIL_TO       recipient; defaults to SMTP_USER
    EMAIL_FROM     optional, defaults to SMTP_USER
"""

from __future__ import annotations

import logging
import os
import smtplib
import ssl
from dataclasses import dataclass
from email.message import EmailMessage
from email.utils import formataddr, formatdate, make_msgid
from typing import List, Optional

log = logging.getLogger(__name__)


class EmailConfigError(RuntimeError):
    pass


@dataclass
class SMTPSettings:
    host: str
    port: int
    user: str
    password: str
    sender: str
    recipients: List[str]
    use_tls: bool = True
    sender_name: str = "CFB Meta Forecast"

    @classmethod
    def from_env(cls) -> "SMTPSettings":
        host = os.getenv("SMTP_HOST", "smtp.gmail.com").strip()
        user = os.getenv("SMTP_USER", "").strip()
        password = os.getenv("SMTP_PASSWORD", "")
        if not user or not password:
            raise EmailConfigError(
                "SMTP_USER and SMTP_PASSWORD must be set. For Gmail, create an app "
                "password at https://myaccount.google.com/apppasswords — a normal "
                "account password will be rejected."
            )

        recipients = [
            addr.strip()
            for addr in (os.getenv("EMAIL_TO") or user).replace(";", ",").split(",")
            if addr.strip()
        ]
        if not recipients:
            raise EmailConfigError("EMAIL_TO resolved to no addresses.")

        port = int(os.getenv("SMTP_PORT", "587"))
        return cls(
            host=host,
            port=port,
            user=user,
            password=password,
            sender=os.getenv("EMAIL_FROM", user).strip(),
            recipients=recipients,
            # Port 465 is implicit TLS; 587 is STARTTLS.
            use_tls=port != 465,
        )


def build_message(
    subject: str,
    html: str,
    text: str,
    settings: SMTPSettings,
) -> EmailMessage:
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = formataddr((settings.sender_name, settings.sender))
    message["To"] = ", ".join(settings.recipients)
    message["Date"] = formatdate(localtime=True)
    message["Message-ID"] = make_msgid(domain=settings.sender.split("@")[-1] or "localhost")
    # Text first: the last part added is what a client prefers to show.
    message.set_content(text)
    message.add_alternative(html, subtype="html")
    return message


def send_email(
    subject: str,
    html: str,
    text: str,
    settings: Optional[SMTPSettings] = None,
    dry_run: bool = False,
) -> bool:
    if settings is None:
        try:
            settings = SMTPSettings.from_env()
        except EmailConfigError:
            # A dry run's whole point is to exercise the pipeline without
            # sending, so missing SMTP credentials are not a failure — the
            # readout still has to come out the other end.
            if not dry_run:
                raise
            log.info("dry run: SMTP is not configured, so nothing would be sent")
            return False
    message = build_message(subject, html, text, settings)

    if dry_run:
        log.info("dry run: would send %r to %s", subject, ", ".join(settings.recipients))
        return False

    context = ssl.create_default_context()
    if settings.use_tls:
        with smtplib.SMTP(settings.host, settings.port, timeout=45) as server:
            server.ehlo()
            server.starttls(context=context)
            server.ehlo()
            server.login(settings.user, settings.password)
            server.send_message(message)
    else:
        with smtplib.SMTP_SSL(settings.host, settings.port, timeout=45, context=context) as server:
            server.login(settings.user, settings.password)
            server.send_message(message)

    log.info("sent %r to %s", subject, ", ".join(settings.recipients))
    return True
