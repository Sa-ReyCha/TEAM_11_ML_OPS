#!/usr/bin/env python
# coding: utf-8

# In[18]:


import sys
sys.path.append('team_11_ml_ops_module')  # Reemplaza 'team_11' con la ruta correcta de tu carpeta

from load_data import load_data
from explore_data import explore_data
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[16]:


datapath = 'DATA/divorce.csv'
data = load_data(datapath)


# In[19]:


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment(f"/team11/prediccion_divorsio")

with mlflow.start_run() as run:
    # Load the diabetes dataset.
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Explorar datos
    explore_data(data)

    # Log data split information
    mlflow.log_param("train_data_size", len(X_train))
    mlflow.log_param("test_data_size", len(X_test))

    # 2. Create a Pipeline for Preprocessing and Model Training
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.1, max_iter=1000, penalty='l1', solver='saga'))
    ])

    # 3. Train the Model
    pipeline.fit(X_train, y_train)

    # 4. Evaluate the Model
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log model performance metrics
    mlflow.log_metric("accuracy", accuracy)

    # Generate and print classification report
    report = classification_report(y_test, y_pred)
    print(report)
    mlflow.log_text(report, "classification_report.txt")  # Save report to artifacts

    # Generate and print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    mlflow.log_text(str(cm), "confusion_matrix.txt")  # Save matrix to artifacts

    # 5. Log Model with MLflow
    mlflow.sklearn.log_model(pipeline, "Divorce_Prediction_Model")

    # 6. Add any relevant tags or information
    mlflow.set_tag("developer", "Team 11")  # For example
    mlflow.set_tag("model_type", "LogisticRegression")

print("MLflow experiment completed. Check mlruns folder for results.")


# In[ ]:





# In[ ]:





# In[ ]:




