from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import pandas as pd
import io
from holisticai.security.mitigation import Anonymize
from sklearn.preprocessing import LabelEncoder
import time

app = FastAPI()

def ensure_correct_types(df):
    """
    Ensures that numeric columns stay as numbers (int/float) and non-numeric columns
    are explicitly converted to strings after the anonymization process.
    """
    for col in df.columns:
        # If the column is numeric, make sure it stays numeric (int or float)
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors='ignore')
        # If the column is non-numeric, convert it to string
        else:
            df[col] = df[col].astype(str)
    return df

@app.post("/anonymize")
async def anonymize_csv(
    file: UploadFile = File(...),
    k: int = Form(...),
    quasi_identifiers: str = Form(...),
):
    try:
        start_time = time.time()  # Start timing

        print(f"Received quasi_identifiers: {quasi_identifiers}")

        if not quasi_identifiers:
            raise HTTPException(status_code=400, detail="Quasi-identifiers must be provided")

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Clean column names to remove extra spaces and newline characters
        df.columns = df.columns.str.replace("\n", "").str.strip()
        print(f"Column names after cleaning: {df.columns.tolist()}")

        quasi_identifiers_list = [qi.strip() for qi in quasi_identifiers.split(",")]

        # Validate quasi_identifiers
        invalid_qis = [qi for qi in quasi_identifiers_list if qi not in df.columns]
        if invalid_qis:
            raise HTTPException(status_code=400, detail=f"Invalid quasi-identifiers: {', '.join(invalid_qis)}")

        target_column = 'income'
        if target_column in quasi_identifiers_list:
            quasi_identifiers_list.remove(target_column)

        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data")

        # Encode categorical columns as numeric using LabelEncoder
        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        # Prepare features and target
        X_train = df.drop(columns=[target_column])
        y_train = df[target_column].values

        # Anonymize using Holistic AI
        anonymizer_start_time = time.time()
        anonymizer = Anonymize(k=k, quasi_identifiers=quasi_identifiers_list, features_names=list(X_train.columns))
        anonymized_df = anonymizer.anonymize(X_train, y_train)
        anonymizer_duration = time.time() - anonymizer_start_time
        print(f"Anonymization took {anonymizer_duration:.2f} seconds")

        # Ensure numeric columns remain numeric, and non-numeric columns are converted to string
        anonymized_df = ensure_correct_types(anonymized_df)

        # Total response time
        total_duration = time.time() - start_time
        print(f"Total response time: {total_duration:.2f} seconds")

        return anonymized_df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
