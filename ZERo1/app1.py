from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import io
import base64

app = Flask(__name__)

# Load and preprocess dataset
df = pd.read_csv('updated_dataset_Multi_healthy.csv')

# Preprocess data
numeric_columns = df.select_dtypes(include=['number']).columns
df.fillna(df[numeric_columns].mean(), inplace=True)

non_numeric_columns = df.select_dtypes(exclude=['number']).columns
for col in non_numeric_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

label_encoder = LabelEncoder()
df['Disease_Encoded'] = label_encoder.fit_transform(df['Disease'])

# Define features and target
features = [
    'Glucose', 'Cholesterol', 'Hemoglobin', 'Platelets', 'White Blood Cells',
    'Mean Corpuscular Volume', 'Mean Corpuscular Hemoglobin',
    'Mean Corpuscular Hemoglobin Concentration', 'HbA1c', 'Creatinine', 'Heart Rate', 'BMI','Age'
    
]
X = df[features]
y = df['Disease_Encoded']

# Train RandomForest model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Handle file upload
            if 'user_file' in request.files and request.files['user_file'].filename != '':
                uploaded_file = request.files['user_file']
                user_input = {}
                data = uploaded_file.read().decode('utf-8')
                for line in data.split('\n'):
                    if ':' in line:
                        key, value = map(str.strip, line.split(':'))
                        if key in features:
                            user_input[key] = float(value)
                user_input['Age'] = user_input.get('Age', 30)  # Default age if missing
            else:
                # Handle manual form input
                user_input = {
                    'Glucose': float(request.form['glucose']),
                    'Cholesterol': float(request.form['cholesterol']),
                    'Hemoglobin': float(request.form['hemoglobin']),
                    'Platelets': float(request.form['platelets']),
                    'White Blood Cells': float(request.form['wbc']),
                    'Mean Corpuscular Volume': float(request.form['mcv']),
                    'Mean Corpuscular Hemoglobin': float(request.form['mch']),
                    'Mean Corpuscular Hemoglobin Concentration': float(request.form['mchc']),
                    'Heart Rate': float(request.form['heart_rate']),
                    'HbA1c': float(request.form['hba1c']),
                    'Creatinine': float(request.form['creatinine']),
                    'BMI': float(request.form['bmi']),
                    'Age': int(request.form['age'])
                    
                }

            # Predict probabilities
            user_data = pd.DataFrame([user_input], columns=features)
            risk_probabilities = rf.predict_proba(user_data)[0]
            disease_probabilities = dict(zip(label_encoder.classes_, risk_probabilities))

            # Plot probabilities
            plt.figure(figsize=(8, 6))
            plt.bar(disease_probabilities.keys(), disease_probabilities.values(), color='skyblue')
            plt.title("Disease Risk Probabilities")
            plt.xlabel("Disease")
            plt.ylabel("Probability")
            plt.xticks(rotation=45)
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode('utf8')
            plt.close()

            # Compare user data to healthy population
            age_group_lower = (user_input['Age'] // 10) * 10
            age_group_upper = age_group_lower + 10
            age_filtered_df = df[
                (df['Age'] >= age_group_lower) & (df['Age'] < age_group_upper) & (df['Disease'] == 'Healthy')
            ]

            comparisons = {}
            for param in features[:-1]:  # Exclude Age and BMI
                plt.figure()
                sns.kdeplot(age_filtered_df[param], fill=True, color='skyblue', label=f'Healthy {age_group_lower}-{age_group_upper}')
                plt.axvline(user_input[param], color='red', linestyle='--', label='User Input')
                plt.title(f'{param} Comparison')
                plt.legend()
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                comparisons[param] = base64.b64encode(img.getvalue()).decode('utf8')
                plt.close()

            return render_template('results.html', plot_url=plot_url, comparisons=comparisons)
        except Exception as e:
            return f"Error: {e}", 400
    return render_template('predict.html')

@app.route('/results')
def results():
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
