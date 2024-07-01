import imaplib
import email
from email.header import decode_header
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.under_sampling import RandomUnderSampler
import pickle
import os
from ucimlrepo import fetch_ucirepo

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
    phiusiil_phishing_url_website = fetch_ucirepo(id=967)
    X = phiusiil_phishing_url_website.data.features
    y = phiusiil_phishing_url_website.data.targets
    data = pd.DataFrame(X, columns=phiusiil_phishing_url_website.variables)
    data['label'] = y
    data['combined'] = data.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    return data

# Balancear los datos
def balance_data(X, y):
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X, y)
    print(f"Class distribution: {dict(pd.Series(y_res).value_counts())}")
    return X_res, y_res

# Guardar el modelo y el vectorizador
def save_model_and_vectorizer(model, vectorizer):
    with open('spam_model.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    with open('vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(vectorizer, vectorizer_file)

# Cargar el modelo y el vectorizador
def load_model_and_vectorizer():
    with open('spam_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)
    return model, vectorizer

# Entrenar el modelo con ajuste de hiperparámetros
def train_model(data):
    X = data['combined']
    y = data['label']
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(X)
    
    # Balancear los datos
    X, y = balance_data(X, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Parámetros para la búsqueda en cuadrícula
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters found: {grid_search.best_params_}")
    
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Guardar el modelo y el vectorizador
    save_model_and_vectorizer(best_rf, vectorizer)
    
    return best_rf, vectorizer

# Predecir si un correo es spam
def predict_spam(model, vectorizer, sender, subject, email):
    combined = [sender.lower().replace(r'[^\w\s]', '') + ' ' + subject.lower().replace(r'[^\w\s]', '') + ' ' + (email.lower().replace(r'[^\w\s]', '') if email else '')]
    combined_vec = vectorizer.transform(combined)
    prediction = model.predict(combined_vec)
    return 'Spam' if prediction[0] == 1 else 'Not Spam'

def main():
    emails = get_emails()
    print(f"Se importaron {len(emails)} correos no leídos.")
    
    if os.path.exists('spam_model.pkl') and os.path.exists('vectorizer.pkl'):
        model, vectorizer = load_model_and_vectorizer()
    else:
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




