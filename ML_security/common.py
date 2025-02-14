from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
import pandas as pd
from enum import Enum
import numpy as np
from holisticai.utils import BinaryClassificationProxy
from holisticai.security.commons.data_minimization._modificators import ModifierHandler
from holisticai.security.commons.data_minimization._selectors import SelectorsHandler, SelectorType
from holisticai.security.commons.data_minimization._core import DataMinimizer
from typing import List, Optional
from sklearn.linear_model import LogisticRegression
from holisticai.security.commons._black_box_attack import BlackBoxAttack
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectPercentile, VarianceThreshold, SelectFromModel
import io
from fastapi import Request as request
import json
import logging


app = FastAPI()


# Define your data input model
class PredictionRequest(BaseModel):
    X: List[List[float]]  # Example input, modify based on your actual data
    Y: Optional[List[float]] = None  # Target variable (optional for prediction)


# DummyModel for testing
class DummyModel:
    def fit(self, X: pd.DataFrame, Y: pd.Series):
        pass


    def predict(self, X: pd.DataFrame) -> List[float]:
        return [1 if x[0] > 0.5 else 0 for x in X.values]


    def predict_proba(self, X: pd.DataFrame) -> List[List[float]]:
        return [[0.5, 0.5] for _ in X.values]


class DummyBinaryClassificationProxy(BinaryClassificationProxy):
    def __init__(self, model: DummyModel):
        self.model = model
        self.classes = [0, 1]


    def fit(self, X: pd.DataFrame, Y: pd.Series):
        self.model.fit(X, Y)


    def predict(self, X: pd.DataFrame) -> List[float]:
        return self.model.predict(X)


    def predict_proba(self, X: pd.DataFrame) -> List[List[float]]:
        return self.model.predict_proba(X)


class DataInput(BaseModel):
    important_features: List[str]


class ModifiedDataResponse(BaseModel):
    method: str
    modified_data: List[List[float]]
    updated_features: List[str]


class SelectorType(str, Enum):
    percentile = "percentile"
    variance = "variance"
    model = "model"




class SelectorRequest(BaseModel):
    X: List[List[float]]
    y: Optional[List[float]] = None
    selector_types: List[SelectorType]


class FeatureSelectionResponse(BaseModel):
    selector_type: str
    selected_features: List[int]


class AttackRequest(BaseModel):
    attack_feature: str


class BlackBoxAttack:
    def __init__(self, attacker_estimator, attack_feature: str):
        self.attacker_estimator = attacker_estimator
        self.attack_feature = attack_feature


    def fit(self, X, Y):
        # Fit the model using the specified estimator
        self.attacker_estimator.fit(X, Y)


    def transform(self, X, Y):
        # Attack the feature by performing prediction (dummy attack for illustration)
        Y_pred = self.attacker_estimator.predict(X)
        return Y, Y_pred  # Returning true values and predicted values


def create_preprocessor(X):
    categorical_features = X.select_dtypes(include=["object", "category"]).columns
    numerical_features = X.select_dtypes(exclude=["object", "category"]).columns


    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))])


    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
# Initialize model and proxy
dummy_model = DummyModel()
proxy_model = DummyBinaryClassificationProxy(model=dummy_model)


# Initialize DataMinimizer
data_minimizer = DataMinimizer(
    proxy=proxy_model,
    selector_types=["FImportance", "Percentile"],
    modifier_types=["Average", "Permutation"]
)


# Initialize the feature selection handler (this could be a dictionary of selectors or another structure)
selectors_handler = {
    "percentile": SelectPercentile(),
    "variance": VarianceThreshold(),
    "model": SelectFromModel(RandomForestClassifier())  # Example: Using RandomForestClassifier for model-based feature selection
}
class PredictionRequest(BaseModel):
    X: list  # List of lists for features
    Y: list  # List for target variable (optional for prediction)




@app.post("/predict/")
async def predict(
    file: UploadFile = File(...),  # File upload
    X: str = Form(...),  # Comma-separated feature column names
    Y: str = Form(...),  # Target column name (single string)
):
    try:
        logging.info(f"Received X: {X}")
        logging.info(f"Received Y: {Y}")

        # Split the comma-separated values into lists
        X_columns = X.split(",")  # Example: "feature1,feature2" → ["feature1", "feature2"]
        Y_column = Y.strip()  # Example: "target"

        # Read the CSV file into a Pandas DataFrame
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        # Validate if columns exist in the uploaded file
        for col in X_columns + [Y_column]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found in CSV")

        # Extract feature (X) and target (Y) values from CSV
        X_data = df[X_columns].values.tolist()  # Convert DataFrame to list of lists
        Y_data = df[Y_column].tolist()  # Convert Series to list

        logging.info(f"Extracted X_data: {X_data[:5]}")  # Log first 5 rows
        logging.info(f"Extracted Y_data: {Y_data[:5]}")  # Log first 5 labels

        # Example prediction logic (replace this with actual model inference)
        predictions = [1 if sum(row) > 5 else 0 for row in X_data]  # Dummy logic

        return {"predictions": predictions}
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))






# Define the modify_data endpoint
@app.post("/modify_data/")
async def modify_data(file: UploadFile = File(...), important_features: str = Form(...)):
    try:
        # Convert important_features from string to a list
        request = DataInput(important_features=important_features.split(","))
       
        # Read the CSV file and convert to DataFrame
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))


        # Extract features (X)
        X_df = df.drop(columns=['target'])  # Assuming 'target' is the target column


        # Initialize ModifierHandler with methods ("Average" and "Permutation")
        modifier_handler = ModifierHandler(methods=["Average", "Permutation"])


        # Apply the data modification methods
        modified_results = modifier_handler(X_df, request.important_features)


        # Prepare the response by collecting modified data for each method
        response = []
        for method, result in modified_results.items():
            updated_feature_names = [str(feature) for feature in result["updated_features"]]
            response.append(
                ModifiedDataResponse(
                    method=method,
                    modified_data=result["x"].values.tolist(),
                    updated_features=updated_feature_names
                )
            )


        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))




@app.post("/select_features/")
async def select_features(file: UploadFile = File(...), selector_types: str = Form(...)):
    try:
        # Read the CSV file and convert to DataFrame
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))


        # Extract features (X) and target (y) from the DataFrame
        X_df = df.drop(columns=['target'])  # Assuming 'target' is the target column
        y_series = df['target'] if 'target' in df else None  # Target is optional


        # Convert DataFrame to list of lists for X and list for y
        X = X_df.values.tolist()  # List of lists for features
        y = y_series.values.tolist() if y_series is not None else None  # List for target


        # Construct the SelectorRequest object
        selector_request = SelectorRequest(X=X, y=y, selector_types=[SelectorType(x) for x in selector_types.split(',')])


        # List to store selected features for each selector type
        selected_features = []


        # Loop through the requested selector types
        for selector_type in selector_request.selector_types:
            if selector_type == SelectorType.percentile:
                selector = selectors_handler["percentile"]
                selector.fit(X_df, y_series)
                selected_features_indexes = selector.get_support()
                selected_features_list = list(selected_features_indexes.nonzero()[0])  # Get selected indices


            elif selector_type == SelectorType.variance:
                selector = selectors_handler["variance"]
                selector.fit(X_df)
                selected_features_indexes = selector.get_support()
                selected_features_list = list(selected_features_indexes.nonzero()[0])  # Get selected indices


            elif selector_type == SelectorType.model:
                selector = selectors_handler["model"]
                selector.fit(X_df, y_series)
                selected_features_list = selector.get_support(indices=True)  # Get indices of selected features


            # Add selected features for this selector type to the response list
            selected_features.append(
                FeatureSelectionResponse(
                    selector_type=selector_type,
                    selected_features=selected_features_list
                )
            )


        # Ensure the response is a proper JSON-serializable format (list of Pydantic models)
        return {"selected_features": selected_features}


    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
   


@app.post("/blackbox-attack")
async def blackbox_attack(file: UploadFile = File(...), attack_feature: str = "sensitive_column"):
    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))


        if attack_feature not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{attack_feature}' not found in dataset")


        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]


        for col in X.select_dtypes(include=["object"]).columns:
            X[col] = X[col].astype("category")


        preprocessor = create_preprocessor(X)
       
        attacker = BlackBoxAttack(attacker_estimator=Pipeline([
            ("preprocessor", preprocessor),
            ("estimator", RandomForestClassifier())
        ]), attack_feature=attack_feature)
       
        attacker.fit(X, y)
       
        y_attack, y_pred_attack = attacker.transform(X, y)
       
        accuracy = np.mean(y_attack == y_pred_attack)


        return {"attack_feature": attack_feature, "attack_success_rate": accuracy}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


