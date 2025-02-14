from fastapi import FastAPI, File, UploadFile, Form, HTTPException
import pandas as pd
import io
from holisticai.security.mitigation import Anonymize
from sklearn.preprocessing import LabelEncoder
app = FastAPI()


@app.post("/anonymize")
async def anonymize_csv(
    file: UploadFile = File(...),
    k: int = Form(...),  # Use Form() to correctly parse form-data
    quasi_identifiers: str = Form(...),  # Use Form() to correctly parse form-data
):
    try:
        print(f"Received quasi_identifiers: {quasi_identifiers}")  # Debugging line

        if not quasi_identifiers:
            raise HTTPException(status_code=400, detail="Quasi-identifiers must be provided")

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Clean column names to remove extra spaces and newline characters
        df.columns = df.columns.str.replace("\n", "").str.strip()  # Explicitly remove newlines and extra spaces

        print(f"Column names after cleaning: {df.columns.tolist()}")  # Debugging line

        quasi_identifiers_list = quasi_identifiers.split(",")

        # Ensure that 'target' is not part of quasi_identifiers
        target_column = 'target'
        if target_column in quasi_identifiers_list:
            quasi_identifiers_list.remove(target_column)

        # Ensure the target column exists in the dataframe
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"Target column '{target_column}' not found in the data")

        # Encode categorical columns as numeric using LabelEncoder
        label_encoders = {}
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])
            label_encoders[column] = le

        # Now that we have encoded the categorical variables, drop the target column from the features
        X_train = df.drop(columns=[target_column])
        y_train = df[target_column].values  # This is the target variable

        anonymizer = Anonymize(k=k, quasi_identifiers=quasi_identifiers_list, features_names=list(X_train.columns))
        anonymized_df = anonymizer.anonymize(X_train, y_train)

        return anonymized_df.to_dict(orient="records")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
