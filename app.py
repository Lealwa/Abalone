from flask import Flask, render_template, request, jsonify
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG)

classifier3 = None
classifier5 = None
meta_classifier = None
label_encoder = None
X_train, X_test, y_train, y_test = None, None, None, None
XTrain1, XTrain2, XTest1, XTest2 = None, None, None, None

def load_data():
    # Assuming fetch_ucirepo is installed and used correctly
    from ucimlrepo import fetch_ucirepo
    abalone = fetch_ucirepo(id=1)
    abalone_features = abalone.data.features
    abalone_class = abalone.data.targets
    df_abalone = abalone_features.join(abalone_class)
    feature_columns = ["Length", "Diameter", "Height", "Whole_weight", "Shucked_weight", "Viscera_weight", "Shell_weight"]
    X = df_abalone[feature_columns].values
    y = df_abalone['Sex'].values

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    return X, y_encoded, label_encoder

def train_classifiers():
    global classifier3, classifier5, label_encoder
    global X_train, X_test, y_train, y_test
    global XTrain1, XTrain2, XTest1, XTest2

    X, y_encoded, label_encoder = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=0)

    classifier3 = KNeighborsClassifier(n_neighbors=3)
    classifier3.fit(X_train, y_train)
    XTrain1 = classifier3.predict(X_train)
    XTest1 = classifier3.predict(X_test)
    train_accuracy3 = accuracy_score(y_train, XTrain1)
    test_accuracy3 = accuracy_score(y_test, XTest1)

    classifier5 = KNeighborsClassifier(n_neighbors=5)
    classifier5.fit(X_train, y_train)
    XTrain2 = classifier5.predict(X_train)
    XTest2 = classifier5.predict(X_test)
    train_accuracy5 = accuracy_score(y_train, XTrain2)
    test_accuracy5 = accuracy_score(y_test, XTest2)

    combined_train_df = pd.DataFrame({'P1': XTrain1, 'P2': XTrain2, 'Y': y_train})
    combined_train_df.to_csv('combined_train.csv', index=False)
    combined_test_df = pd.DataFrame({'P1': XTest1, 'P2': XTest2, 'Y': y_test})
    combined_test_df.to_csv('combined_test.csv', index=False)

    return {'train_accuracy3': train_accuracy3, 'test_accuracy3': test_accuracy3, 'train_accuracy5': train_accuracy5, 'test_accuracy5': test_accuracy5}


def train_meta_classifier():
    global meta_classifier, XTrain1, XTrain2, y_train
    f_meta = np.column_stack((XTrain1, XTrain2))
    meta_classifier = GaussianNB()
    meta_classifier.fit(f_meta, y_train)

app.route('/train', methods=['POST'])
def train_model():
    train = train_classifiers()
    train_meta_classifier()
    return jsonify(train)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    accuracies = train_classifiers()
    train_meta_classifier()
    return jsonify(accuracies)

@app.route('/predict', methods=['POST'])
def predict():
    global classifier3, classifier5, meta_classifier, label_encoder
    try:
        if classifier3 is None or classifier5 is None or meta_classifier is None or label_encoder is None:
            raise Exception("Models not trained yet")

        input_data = request.json['data']
        logging.debug(f"Received input data: {input_data}")
        
        new_data = [input_data]

        knn_pred1 = classifier3.predict(new_data)
        knn_pred2 = classifier5.predict(new_data)
        combined_features = np.column_stack((knn_pred1, knn_pred2))
        prediction = meta_classifier.predict(combined_features)


        logging.debug(f"Prediction : {prediction}")

        return jsonify({'prediction': label_encoder.inverse_transform(prediction)[0]})
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)