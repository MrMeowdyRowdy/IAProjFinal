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
os.environ['OMP_NUM_THREADS'] = '6'

# Configuración del correo
username = "info@expertweb.com.ec"
password = "k.bDHW8#T$9Q"
imap_server = "mail.expertweb.com.ec"
imap_port = 993

# fetch dataset 
phiusiil_phishing_url_website = fetch_ucirepo(id=967) 
  
# data (as pandas dataframes) 
# metadata 
print(phiusiil_phishing_url_website.metadata) 
  
# variable information 
print(phiusiil_phishing_url_website.variables)

# Función para obtener correos electrónicos no leídos
def get_emails():
    mail = imaplib.IMAP4_SSL(imap_server, imap_port)
    mail.login(username, password)
    mail.select("inbox")
    status, messages = mail.search(None, 'UNSEEN')
    email_ids = messages[0].split()
    emails = []
    for email_id in email_ids:
        status, msg_data = mail.fetch(email_id, '(RFC822)')
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8")
                from_ = msg.get("From")
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == "text/plain":
                            charset = part.get_content_charset() or "utf-8"
                            body = part.get_payload(decode=True).decode(charset)
                            break
                else:
                    charset = msg.get_content_charset() or "utf-8"
                    body = msg.get_payload(decode=True).decode(charset)
                emails.append({"subject": subject, "from": from_, "body": body})
    return emails

# Cargar y preprocesar el conjunto de datos
# def load_and_preprocess_data():
#     data = pd.read_csv('COLOCAR AQUÍ LA RUTA DE LOS DATOS')
#     data['text'] = data['text'].str.lower().str.replace(r'[^\w\s]', '')
#     data['sender'] = data['sender'].str.lower().str.replace(r'[^\w\s]', '')
#     data['subject'] = data['subject'].str.lower().str.replace(r'[^\w\s]', '')
#     data['combined'] = data['sender'] + ' ' + data['subject'] + ' ' + data['text']
#     return data

# Entrenar el modelo
def train_model():
    X = phiusiil_phishing_url_website.data.features 
    y = phiusiil_phishing_url_website.data.targets 
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
    #data = load_and_preprocess_data()
    model, vectorizer = train_model()
    for mail in emails:
        sender = mail['from']
        subject = mail['subject']
        body = mail['body']
        result = predict_spam(model, vectorizer, sender, subject, body)
        print(f"From: {sender}\nSubject: {subject}\nPrediction: {result}\n")

if __name__ == "__main__":
    main()
