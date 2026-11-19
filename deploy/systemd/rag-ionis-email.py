#!/usr/bin/env python3
"""Send a monitoring message from stdin through the VPS SMTP relay."""

import argparse
import configparser
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage
from pathlib import Path


def send(config: configparser.SectionProxy, subject: str, body: str) -> None:
    security = config["security"]
    if security not in ("starttls", "tls"):
        raise ValueError("security must be starttls or tls")

    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = config["from"]
    message["To"] = config["to"]
    message.set_content(body)
    if not message["From"].addresses or not message["To"].addresses:
        raise ValueError("from and to are required")

    context = ssl.create_default_context()
    host = config["host"]
    if security == "tls":
        connection = smtplib.SMTP_SSL(host, config.getint("port", 465), timeout=20, context=context)
    else:
        connection = smtplib.SMTP(host, config.getint("port", 587), timeout=20)
    with connection as smtp:
        if security == "starttls":
            smtp.starttls(context=context)
        if config.get("username"):
            smtp.login(config["username"], config["password"])
        if smtp.send_message(message):
            raise smtplib.SMTPException("one or more recipients were refused")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("subject")
    args = parser.parse_args()
    path = Path(os.getenv("RAG_IONIS_SMTP_CONFIG", "/etc/rag-ionis/smtp.ini"))
    config = configparser.ConfigParser(interpolation=None)
    try:
        with path.open(encoding="utf-8") as source:
            config.read_file(source)
        body = sys.stdin.buffer.read().decode("utf-8", errors="replace")
        send(config["smtp"], args.subject, body)
    except (configparser.Error, KeyError, OSError, ValueError) as exc:
        # SMTP replies and malformed config lines can contain private data.
        print(f"Email delivery failed ({type(exc).__name__}); check {path}.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
