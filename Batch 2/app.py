# importing required libraries
from flask import Flask, request, render_template, redirect, url_for, session, flash, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
import pickle
import re
import tldextract
import urllib.parse
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import whois
import os
import hashlib
from flask_bcrypt import Bcrypt
from functools import wraps
import base64
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import tempfile
import time
import matplotlib.pyplot as plt
import io

warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = os.urandom(24)


# More comprehensive feature extraction
class FeatureExtraction:
    def __init__(self, url):
        self.url = url
        self.domain = ""
        self.whois_features = {}
        self.extracted_url = tldextract.extract(url)
        self.features = {}

    def extract_domain_info(self):
        try:
            self.domain = self.extracted_url.domain + '.' + self.extracted_url.suffix
            try:
                self.whois_features = whois.whois(self.domain)
            except:
                self.whois_features = {}
        except:
            pass

    def url_length(self):
        return len(self.url)

    def domain_age(self):
        if not self.whois_features or not self.whois_features.get('creation_date'):
            return -1

        creation_date = self.whois_features['creation_date']
        if isinstance(creation_date, list):
            creation_date = creation_date[0]

        try:
            days_since_creation = (datetime.now() - creation_date).days
            return days_since_creation
        except:
            return -1

    def has_suspicious_tld(self):
        suspicious_tlds = ['.xyz', '.top', '.club', '.online', '.site']
        return any(self.url.endswith(tld) for tld in suspicious_tlds)

    def has_suspicious_words(self):
        suspicious_words = ['secure', 'account', 'banking', 'login', 'verify', 'update', 'confirm']
        return any(word in self.url.lower() for word in suspicious_words)

    def count_dots(self):
        return self.url.count('.')

    def count_special_chars(self):
        special_chars = ['@', '!', '#', '$', '%', '^', '*', '(', ')', '-', '+', '=', '{', '}', '[', ']']
        return sum(self.url.count(char) for char in special_chars)

    def has_ip_address(self):
        pattern = re.compile(
            r'(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])')
        return 1 if pattern.search(self.url) else 0

    def has_https(self):
        return 1 if self.url.startswith('https://') else 0

    def url_shortening_service(self):
        shortening_services = ['bit.ly', 'goo.gl', 'tinyurl.com', 't.co', 'tiny.cc']
        return any(service in self.url for service in shortening_services)

    def has_at_symbol(self):
        return '@' in self.url

    def redirect_using_double_slash(self):
        return self.url.count('//') > 1

    def get_features(self):
        self.extract_domain_info()

        self.features = {
            'url_length': self.url_length(),
            'domain_age': self.domain_age(),
            'has_suspicious_tld': 1 if self.has_suspicious_tld() else 0,
            'has_suspicious_words': 1 if self.has_suspicious_words() else 0,
            'count_dots': self.count_dots(),
            'count_special_chars': self.count_special_chars(),
            'has_ip_address': self.has_ip_address(),
            'has_https': self.has_https(),
            'url_shortening_service': 1 if self.url_shortening_service() else 0,
            'has_at_symbol': 1 if self.has_at_symbol() else 0,
            'redirect_double_slash': 1 if self.redirect_using_double_slash() else 0
        }

        return list(self.features.values())


class ModelTrainer:
    def __init__(self, csv_path='phishing_dataset.csv'):
        """
        Initialize the ModelTrainer with a path to a CSV file containing training data

        Parameters:
        -----------
        csv_path : str
            Path to the CSV file containing the training data
        """
        # Only using Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        self.csv_path = csv_path

        # Performance metrics for the model
        self.model_performance = {}

        # Load and split data from CSV
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_split_data()

        # Train the model with data
        self.train_model()

        # Evaluate model performance
        self.evaluate_model()

    def load_and_split_data(self):
        """
        Load data from CSV file and split into training and testing sets

        The CSV file should have the following columns:
        - url_length
        - domain_age
        - has_suspicious_tld
        - has_suspicious_words
        - count_dots
        - count_special_chars
        - has_ip_address
        - has_https
        - url_shortening_service
        - has_at_symbol
        - redirect_double_slash
        - label (0 for legitimate, 1 for phishing)
        """
        try:
            print(f"Loading data from {self.csv_path}...")
            data = pd.read_csv(self.csv_path)

            # Check if the required columns are present
            required_columns = [
                'url_length', 'domain_age', 'has_suspicious_tld', 'has_suspicious_words',
                'count_dots', 'count_special_chars', 'has_ip_address', 'has_https',
                'url_shortening_service', 'has_at_symbol', 'redirect_double_slash', 'label'
            ]

            for col in required_columns:
                if col not in data.columns:
                    raise ValueError(f"Required column '{col}' not found in CSV file")

            # Extract features and labels
            X = data.drop('label', axis=1).values
            y = data['label'].values

            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

            print(f"Data loaded successfully: {len(X_train)} training samples, {len(X_test)} testing samples")
            return X_train, X_test, y_train, y_test

        except FileNotFoundError:
            print(f"Warning: CSV file '{self.csv_path}' not found. Falling back to sample data.")
            return self.create_and_split_data()
        except Exception as e:
            print(f"Error loading data from CSV: {str(e)}. Falling back to sample data.")
            return self.create_and_split_data()

    def create_and_split_data(self):
        """
        Create a small sample dataset as fallback when CSV loading fails
        """
        print("Using sample data for training...")
        # In a real-world scenario, you would load this from a larger dataset
        # This is an expanded training set from the original example
        X = np.array([
            # URL length, domain age, suspicious TLD, suspicious words, dots, special chars,
            # IP address, HTTPS, shortening service, @ symbol, double slash
            [75, 500, 0, 0, 2, 1, 0, 1, 0, 0, 0],  # Legitimate
            [65, 400, 0, 0, 2, 0, 0, 1, 0, 0, 0],  # Legitimate
            [50, 800, 0, 0, 3, 0, 0, 1, 0, 0, 0],  # Legitimate
            [60, 700, 0, 0, 2, 0, 0, 1, 0, 0, 0],  # Legitimate
            [55, 600, 0, 0, 2, 1, 0, 1, 0, 0, 0],  # Legitimate
            [70, 450, 0, 0, 3, 0, 0, 1, 0, 0, 0],  # Legitimate
            [80, 300, 0, 0, 2, 0, 0, 1, 0, 0, 0],  # Legitimate
            [90, 10, 1, 1, 4, 3, 0, 0, 0, 0, 0],  # Phishing
            [120, 5, 1, 1, 5, 4, 1, 0, 1, 1, 0],  # Phishing
            [85, 3, 1, 1, 3, 2, 0, 0, 1, 0, 1],  # Phishing
            [100, 15, 1, 1, 4, 3, 0, 0, 0, 1, 1],  # Phishing
            [110, 8, 1, 1, 5, 2, 1, 0, 0, 1, 0],  # Phishing
            [95, 12, 1, 1, 4, 3, 0, 0, 1, 0, 0],  # Phishing
            [105, 7, 1, 1, 3, 2, 0, 0, 0, 1, 1],  # Phishing
        ])
        y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # 0 for legitimate, 1 for phishing

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return X_train, X_test, y_train, y_test

    def train_model(self):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        self.model.fit(self.X_train, self.y_train)
        print("Random Forest model trained.")

    def evaluate_model(self):
        """Evaluate model performance on test data"""
        # Basic prediction on test set
        y_pred = self.model.predict(self.X_test)

        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)

        # Cross-validation for more robust accuracy assessment
        cv_scores = cross_val_score(self.model, np.vstack((self.X_train, self.X_test)),
                                    np.hstack((self.y_train, self.y_test)),
                                    cv=5, scoring='accuracy')

        # Store performance metrics
        self.model_performance = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cross_val_accuracy': cv_scores.mean(),
            'cross_val_std': cv_scores.std()
        }

        print("Random Forest model evaluation:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  Cross-val Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

        return self.model_performance

    def generate_performance_chart(self):
        """Generate performance chart for Random Forest model"""
        try:
            # Prepare data for plotting
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
            values = [
                self.model_performance['accuracy'],
                self.model_performance['precision'],
                self.model_performance['recall'],
                self.model_performance['f1_score']
            ]

            # Create figure
            fig, ax = plt.subplots(figsize=(8, 6))

            # Create bars
            bars = ax.bar(metrics, values, color='skyblue')

            # Add labels, title and values
            ax.set_ylabel('Score')
            ax.set_title('Random Forest Model Performance')
            ax.set_ylim(0, 1.1)

            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{height:.2f}', ha='center', fontsize=10)

            # Save to memory buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)

            # Convert to base64 for embedding in HTML
            chart_img = base64.b64encode(buf.getvalue()).decode('utf-8')
            plt.close()

            return chart_img
        except Exception as e:
            print(f"Error generating chart: {str(e)}")
            return None

    def predict(self, features):
        """
        Make a prediction using the trained model

        Parameters:
        -----------
        features : list
            List of feature values in the same order as the training data

        Returns:
        --------
        prediction : array
            0 for legitimate, 1 for phishing
        probability : array
            Probability estimates for each class
        """
        features_array = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features_array)
        probability = self.model.predict_proba(features_array)

        return prediction, probability

    def save_model(self, filepath='phishing_detector_model.pkl'):
        """
        Save the trained model to a file

        Parameters:
        -----------
        filepath : str
            Path to save the model to
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='phishing_detector_model.pkl'):
        """
        Load a trained model from a file

        Parameters:
        -----------
        filepath : str
            Path to load the model from
        """
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            print(f"Model loaded from {filepath}")
            return True
        except FileNotFoundError:
            print(f"Model file {filepath} not found")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False


def create_csv_dataset(output_path='phishing_dataset.csv'):
    """
    Create a CSV dataset from the sample data (useful for testing)

    Parameters:
    -----------
    output_path : str
        Path to save the CSV file to
    """
    # Sample data
    X = np.array([
        # URL length, domain age, suspicious TLD, suspicious words, dots, special chars,
        # IP address, HTTPS, shortening service, @ symbol, double slash
        [75, 500, 0, 0, 2, 1, 0, 1, 0, 0, 0],  # Legitimate
        [65, 400, 0, 0, 2, 0, 0, 1, 0, 0, 0],  # Legitimate
        [50, 800, 0, 0, 3, 0, 0, 1, 0, 0, 0],  # Legitimate
        [60, 700, 0, 0, 2, 0, 0, 1, 0, 0, 0],  # Legitimate
        [55, 600, 0, 0, 2, 1, 0, 1, 0, 0, 0],  # Legitimate
        [70, 450, 0, 0, 3, 0, 0, 1, 0, 0, 0],  # Legitimate
        [80, 300, 0, 0, 2, 0, 0, 1, 0, 0, 0],  # Legitimate
        [90, 10, 1, 1, 4, 3, 0, 0, 0, 0, 0],  # Phishing
        [120, 5, 1, 1, 5, 4, 1, 0, 1, 1, 0],  # Phishing
        [85, 3, 1, 1, 3, 2, 0, 0, 1, 0, 1],  # Phishing
        [100, 15, 1, 1, 4, 3, 0, 0, 0, 1, 1],  # Phishing
        [110, 8, 1, 1, 5, 2, 1, 0, 0, 1, 0],  # Phishing
        [95, 12, 1, 1, 4, 3, 0, 0, 1, 0, 0],  # Phishing
        [105, 7, 1, 1, 3, 2, 0, 0, 0, 1, 1],  # Phishing
    ])
    y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])  # 0 for legitimate, 1 for phishing

    # Create DataFrame
    columns = [
        'url_length', 'domain_age', 'has_suspicious_tld', 'has_suspicious_words',
        'count_dots', 'count_special_chars', 'has_ip_address', 'has_https',
        'url_shortening_service', 'has_at_symbol', 'redirect_double_slash'
    ]

    df = pd.DataFrame(X, columns=columns)
    df['label'] = y

    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"Sample dataset created and saved to {output_path}")
    return df


def capture_screenshot(url):
    try:
        # Set up headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1280,800")
        chrome_options.add_argument("--no-sandbox")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(url)
        time.sleep(2)  # Allow time for page to load

        # Capture screenshot using a more robust temporary file approach
        try:
            # Create temporary file in a directory we have write access to
            temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp')
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_path = os.path.join(temp_dir, f"screenshot_{hashlib.md5(url.encode()).hexdigest()}.png")

            driver.save_screenshot(temp_file_path)

            # Convert to base64 for embedding in HTML
            with open(temp_file_path, "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

            # Clean up
            os.remove(temp_file_path)
        except Exception as file_error:
            print(f"File operation error: {file_error}")
            # Alternative approach using BytesIO if file system access fails
            import io
            from PIL import Image

            # Take screenshot and save to in-memory buffer
            png = driver.get_screenshot_as_png()
            im = Image.open(io.BytesIO(png))
            buffered = io.BytesIO()
            im.save(buffered, format="PNG")
            encoded_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        driver.quit()
        return encoded_image
    except Exception as e:
        print(f"Screenshot error: {str(e)}")
        return None


# Create a CSV dataset when needed
if not os.path.exists('phishing_dataset.csv'):
    create_csv_dataset()

# Initialize model trainer
model_trainer = ModelTrainer('phishing_dataset.csv')


# Add the home route
@app.route('/')
def home():
    return render_template('index.html')


# Add the result route to handle URL scanning
@app.route('/result', methods=['GET', 'POST'])
def result():
    try:
        if request.method == 'POST':
            url = request.form['name']
        else:
            url = request.args.get('url')

        if not url:
            return redirect(url_for('home'))

        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url

        # Extract features
        fe = FeatureExtraction(url)
        features = fe.get_features()

        # Make prediction using Random Forest model
        prediction, probability = model_trainer.predict(features)

        # Get risk percentage
        risk_percentage = round(probability[0][1] * 100, 2)

        # Capture screenshot if safe or medium risk
        screenshot = None
        if prediction[0] == 0 or (prediction[0] == 1 and risk_percentage < 75):
            screenshot = capture_screenshot(url)

        # Store scan history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result_hash = hashlib.md5((url + timestamp).encode()).hexdigest()

        # Get model performance data
        model_accuracy = model_trainer.model_performance['accuracy'] * 100
        model_precision = model_trainer.model_performance['precision'] * 100

        scan_result = {
            'id': result_hash,
            'url': url,
            'timestamp': timestamp,
            'risk_percentage': risk_percentage,
            'features': fe.features,
            'model_used': 'random_forest',
            'model_accuracy': model_accuracy,
            'model_precision': model_precision
        }

        # Store in session
        if 'scan_history' not in session:
            session['scan_history'] = []

        session['scan_history'] = [scan_result] + session['scan_history'][:9]  # Keep last 10
        session.modified = True

        # Determine result message
        if prediction[0] == 1:
            status = "UNSAFE"
            warning = "Visit at your own risk"
            is_safe = False
            risk_level = "high" if risk_percentage > 75 else "medium"
        else:
            status = "SAFE"
            warning = "Safe to visit"
            is_safe = True
            risk_level = "low"

        return render_template('result.html',
                               result={
                                   'url': url,
                                   'status': status,
                                   'warning': warning,
                                   'is_safe': is_safe,
                                   'risk_percentage': risk_percentage,
                                   'risk_level': risk_level,
                                   'features': fe.features,
                                   'screenshot': screenshot,
                                   'model_used': 'random_forest',
                                   'model_accuracy': model_accuracy,
                                   'model_precision': model_precision
                               })

    except Exception as e:
        return render_template('index.html', error="Error processing URL: " + str(e))


@app.route('/bulk-scan', methods=['GET', 'POST'])
def bulk_scan():
    results = []

    if request.method == 'POST':
        urls = request.form.get('urls', '').splitlines()
        for url in urls:
            if url.strip():
                try:
                    # Basic URL validation
                    if not url.startswith(('http://', 'https://')):
                        url = 'http://' + url

                    # Extract features
                    fe = FeatureExtraction(url)
                    features = fe.get_features()

                    # Make prediction using Random Forest model
                    prediction, probability = model_trainer.predict(features)

                    # Get risk percentage
                    risk_percentage = round(probability[0][1] * 100, 2)

                    # Determine result
                    if prediction[0] == 1:
                        status = "UNSAFE"
                        is_safe = False
                        risk_level = "high" if risk_percentage > 75 else "medium"
                    else:
                        status = "SAFE"
                        is_safe = True
                        risk_level = "low"

                    # Get model accuracy
                    model_accuracy = model_trainer.model_performance['accuracy'] * 100

                    results.append({
                        'url': url,
                        'status': status,
                        'is_safe': is_safe,
                        'risk_percentage': risk_percentage,
                        'risk_level': risk_level,
                        'model_used': 'random_forest',
                        'model_accuracy': model_accuracy
                    })

                except Exception as e:
                    results.append({
                        'url': url,
                        'status': 'ERROR',
                        'error': str(e)
                    })

    return render_template('bulk_scan.html', results=results)


@app.route('/model-performance')
def model_performance():
    # Generate performance chart for the Random Forest model
    performance_chart = model_trainer.generate_performance_chart()

    return render_template('model_performance.html',
                           model_performance=model_trainer.model_performance,
                           performance_chart=performance_chart)


# Add the history route
@app.route('/history')
def history():
    scan_history = session.get('scan_history', [])
    return render_template('history.html', history=scan_history)


# Add the usecases route
@app.route('/usecases')
def usecases():
    return render_template('usecases.html')


# Add education route
@app.route('/education')
def education():
    return render_template('education.html')


# User authentication
bcrypt = Bcrypt(app)

# User model (simplified)
users = {}  # In production, use a proper database


# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)

    return decorated_function


# User routes
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users:
            flash('Username already exists', 'danger')
            return render_template('register.html')

        hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')
        users[username] = {
            'password': hashed_password,
            'scan_history': []
        }

        flash('Account created! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username in users and bcrypt.check_password_hash(users[username]['password'], password):
            session['user_id'] = username
            session['username'] = username

            next_page = request.args.get('next')
            flash('Login successful!', 'success')
            return redirect(next_page or url_for('home'))
        else:
            flash('Login failed. Please check your credentials.', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('home'))


@app.route('/api/scan', methods=['POST'])
def api_scan():
    try:
        data = request.get_json()
        url = data.get('url', '')
        api_key = data.get('api_key', '')

        # Simple API key validation (in production, use a proper authentication system)
        valid_api_keys = ['phishguard-demo-key']  # Store securely in production

        if not api_key or api_key not in valid_api_keys:
            return jsonify({'error': 'Invalid API key'}), 401

        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # Extract features
        fe = FeatureExtraction(url)
        features = fe.get_features()

        # Make prediction using Random Forest model
        prediction, probability = model_trainer.predict(features)
        risk_percentage = round(float(probability[0][1]) * 100, 2)

        # Get model performance metrics
        model_accuracy = model_trainer.model_performance['accuracy'] * 100
        model_precision = model_trainer.model_performance['precision'] * 100
        model_recall = model_trainer.model_performance['recall'] * 100
        model_f1 = model_trainer.model_performance['f1_score'] * 100

        return jsonify({
            'url': url,
            'is_phishing': bool(prediction[0]),
            'probability': float(probability[0][1]),
            'risk_percentage': risk_percentage,
            'status': 'UNSAFE' if prediction[0] == 1 else 'SAFE',
            'risk_level': "high" if risk_percentage > 75 else "medium" if prediction[0] == 1 else "low",
            'features': fe.features,
            'model_performance': {
                'accuracy': model_accuracy,
                'precision': model_precision,
                'recall': model_recall,
                'f1_score': model_f1
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/bulk-scan', methods=['POST'])
def api_bulk_scan():
    try:
        data = request.get_json()
        urls = data.get('urls', [])
        api_key = data.get('api_key', '')

        # Simple API key validation
        valid_api_keys = ['phishguard-demo-key']  # Store securely in production

        if not api_key or api_key not in valid_api_keys:
            return jsonify({'error': 'Invalid API key'}), 401

        if not urls or not isinstance(urls, list):
            return jsonify({'error': 'URLs must be provided as a list'}), 400

        results = []
        for url in urls:
            try:
                # Extract features
                fe = FeatureExtraction(url)
                features = fe.get_features()

                # Make prediction
                prediction, probability = model_trainer.predict(features)
                risk_percentage = round(float(probability[0][1]) * 100, 2)

                results.append({
                    'url': url,
                    'is_phishing': bool(prediction[0]),
                    'risk_percentage': risk_percentage,
                    'status': 'UNSAFE' if prediction[0] == 1 else 'SAFE',
                    'risk_level': "high" if risk_percentage > 75 else "medium" if prediction[0] == 1 else "low",
                })
            except Exception as e:
                results.append({
                    'url': url,
                    'error': str(e)
                })

        return jsonify({
            'results': results,
            'count': len(results)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/model-performance', methods=['GET'])
def api_model_performance():
    try:
        api_key = request.args.get('api_key', '')

        # Simple API key validation
        valid_api_keys = ['phishguard-demo-key']  # Store securely in production

        if not api_key or api_key not in valid_api_keys:
            return jsonify({'error': 'Invalid API key'}), 401

        return jsonify({
            'model': 'random_forest',
            'accuracy': model_trainer.model_performance['accuracy'],
            'precision': model_trainer.model_performance['precision'],
            'recall': model_trainer.model_performance['recall'],
            'f1_score': model_trainer.model_performance['f1_score'],
            'cross_val_accuracy': model_trainer.model_performance['cross_val_accuracy'],
            'cross_val_std': model_trainer.model_performance['cross_val_std']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/docs')
def api_docs():
    return render_template('api_docs.html')


@app.route('/dashboard')
@login_required
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # Get the user's scan history
    scan_history = session.get('scan_history', [])

    # Prepare stats
    total_scans = len(scan_history)
    safe_sites = sum(1 for scan in scan_history if scan.get('risk_percentage', 0) < 50)
    unsafe_sites = total_scans - safe_sites

    # Chart data
    safe_percentage = (safe_sites / total_scans * 100) if total_scans > 0 else 0
    unsafe_percentage = (unsafe_sites / total_scans * 100) if total_scans > 0 else 0

    return render_template('dashboard.html',
                           scan_history=scan_history,
                           total_scans=total_scans,
                           safe_sites=safe_sites,
                           unsafe_sites=unsafe_sites,
                           safe_percentage=safe_percentage,
                           unsafe_percentage=unsafe_percentage)


@app.route('/screenshots/<path:url>')
def get_screenshot(url):
    try:
        screenshot = capture_screenshot(url)
        if screenshot:
            return render_template('screenshot.html', url=url, screenshot=screenshot)
        else:
            return "Screenshot not available", 404
    except Exception as e:
        return str(e), 500


@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        url = data.get('url')
        is_accurate = data.get('is_accurate')
        comments = data.get('comments')

        # In a real app, you'd save this feedback to a database
        # For this demo, just print it
        print(f"Feedback received for {url}: Accurate={is_accurate}, Comments={comments}")

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Train and save the model if not already saved
    if not os.path.exists('models/phishing_detector_model.pkl'):
        model_trainer.save_model('models/phishing_detector_model.pkl')
    else:
        # Load the model if it exists
        model_trainer.load_model('models/phishing_detector_model.pkl')

    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)