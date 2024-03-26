import smtplib
from flask_mail import Message
from database import get_logger

logger = get_logger()
def sendVerificationEmail(user_email, verification_token, mail, app):
    with app.app_context():
        try:
            server = smtplib.SMTP(app.config['MAIL_SERVER'], app.config['MAIL_PORT'])
            server.ehlo()
            server.starttls()
            server.login(app.config['MAIL_USERNAME'], app.config['MAIL_PASSWORD'])
            message = Message('Verify your email', sender=app.config['MAIL_USERNAME'], recipients=[user_email])
            message.body = f'Verify your email by clicking on the link: http://localhost:5000/verify/{verification_token}'
            server.sendmail(app.config['MAIL_USERNAME'], user_email, message.as_string())
            logger.info('Email sent successfully')
        except Exception as e:
            logger.error(f'Failed to send email: {e}', exc_info=True)
            return False