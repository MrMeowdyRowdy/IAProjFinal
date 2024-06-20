import imaplib
import email
from email.header import decode_header
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from ucimlrepo import fetch_ucirepo 

import os

os.environ['OMP_NUM_THREADS'] = '11'

# Configuración del correo
username = "info@expertweb.com.ec"
password = "k.bDHW8#T$9Q"
imap_server = "mail.expertweb.com.ec"
imap_port = 993

# Función para obtener correos electrónicos no leídos
def get_emails(limit=100):
    try:
        mail = imaplib.IMAP4_SSL(imap_server, imap_port)
        mail.login(username, password)
        mail.select("inbox")
        status, messages = mail.search(None, 'UNSEEN')
        email_ids = messages[0].split()
        
        # Limitamos la cantidad de correos a procesar a los últimos 'limit' correos no leídos
        email_ids = email_ids[-limit:]

        emails = []
        for email_id in email_ids:
            status, msg_data = mail.fetch(email_id, '(RFC822)')
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject, encoding = decode_header(msg["Subject"])[0] if msg["Subject"] else (None, None)
                    if isinstance(subject, bytes):
                        try:
                            subject = subject.decode(encoding if encoding else "latin-1")
                        except (UnicodeDecodeError, LookupError):
                            subject = subject.decode("latin1")
                    from_ = msg.get("From")
                    body = None
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                charset = part.get_content_charset() or "latin-1"
                                body = part.get_payload(decode=True).decode(charset)
                                break
                    else:
                        charset = msg.get_content_charset() or "latin-1"
                        body = msg.get_payload(decode=True).decode(charset)
                    emails.append({"subject": subject, "from": from_, "body": body})
        return emails
    except Exception as e:
        print(f"Error al obtener correos: {e}")
        return []

# Cargar y preprocesar el conjunto de datos desde UCI Repository
def load_and_preprocess_data():
    # fetch dataset
    phiusiil_phishing_url_website = fetch_ucirepo(id=967)

    # data (as pandas dataframes)
    X = phiusiil_phishing_url_website.data.features
    y = phiusiil_phishing_url_website.data.targets

    # Combine the necessary fields for the text processing
    data = pd.DataFrame(X, columns=phiusiil_phishing_url_website.variables)
    data['label'] = y
    data['combined'] = data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

    return data

# Entrenar el modelo
def train_model(data):
    X = data['combined']
    y = data['label']
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred))
    return model, vectorizer

# Predecir si un correo es spam
def predict_spam(model, vectorizer, sender, subject, email):
    combined = [sender.lower().replace(r'[^\w\s]', '') + ' ' + subject.lower().replace(r'[^\w\s]', '') + ' ' + email.lower().replace(r'[^\w\s]', '')]
    combined_vec = vectorizer.transform(combined)
    prediction = model.predict(combined_vec)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'

def main():
    emails = get_emails()
    print(f"Se importaron {len(emails)} correos no leídos.")
    data = load_and_preprocess_data()
    model, vectorizer = train_model(data)
    for mail in emails:
        sender = mail['from']
        subject = mail['subject']
        body = mail['body']
        result = predict_spam(model, vectorizer, sender, subject, body)
        print(f"From: {sender}\nSubject: {subject}\nPrediction: {result}\n")

if __name__ == "__main__":
    main()
