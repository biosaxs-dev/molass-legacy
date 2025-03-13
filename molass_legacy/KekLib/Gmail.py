"""
PythonでGmailを送信する
https://qiita.com/eito_2/items/ef77e44955e43f31ba78

Gmail（Google）アカウント – アプリパスワードの生成（作成）
https://pc-karuma.net/google-account-generate-app-password/
"""
import os
import smtplib
from email.mime.text import MIMEText
from email.utils import formatdate

def send_notify_gmail(subject, body):
    sendAddress = os.environ.get("MY_GMAIL_ADDRESS")
    if sendAddress is None:
        return

    password = os.environ.get("MY_GMAIL_PASSWORD")
    fromAddress = sendAddress
    toAddress = sendAddress

    # SMTPサーバに接続
    smtpobj = smtplib.SMTP('smtp.gmail.com', 587)
    smtpobj.starttls()
    smtpobj.login(sendAddress, password)

    # メール作成
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = fromAddress
    msg['To'] = toAddress
    msg['Date'] = formatdate()

    # 作成したメールを送信
    smtpobj.send_message(msg)
    smtpobj.close()

if __name__ == '__main__':
    subject = "Test"
    body = "test body"
    send_notify_gmail(subject, body)
