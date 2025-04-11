# Training Source Code

import mlflow
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

logging.basicConfig(
    level=logging.INFO,
    format= '%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("mlops.log"),
        logging.StreamHandler()
    ]
)

logging.info("Starting model training process....")

logging.info('Loading data...')
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)

logging.info("Data loaded and split into training and test sets.")

with mlflow.start_run():
    logging.info("Training the RandomForest model...")
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    logging.info("Model Training completed")

    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)

    mlflow.log_metric("accuracy", accuracy)
    logging.info(f"Model Accuracy: {accuracy:.4f}")
