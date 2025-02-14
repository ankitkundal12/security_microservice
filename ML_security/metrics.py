from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from typing import List, Dict, Any, Union, Callable
import pandas as pd
import io
from sklearn.linear_model import LinearRegression, LogisticRegression
from holisticai.security.commons import BlackBoxAttack
from holisticai.security.metrics._utils import check_valid_output_type
from io import StringIO
from typing import List
from io import StringIO
from holisticai.security.metrics import shapr_score
from holisticai.security.metrics import privacy_risk_score
import ast
from holisticai.security.metrics import data_minimization_score  # Import your functions here
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error, r2_score
from holisticai.security.metrics._attribute_attack import attribute_attack_score


app = FastAPI()

def k_anonymity(df, qi):
    """
    Computes k-Anonymity metric.
    """
    return df[qi].value_counts()

def l_diversity(df, qi, sa):
    """
    Computes l-Diversity metric.
    """
    df_grouped = df.groupby(qi, as_index=False)
    return {s: [len(row["unique"]) for _, row in df_grouped[s].agg(["unique"]).dropna().iterrows()] for s in sa}


@app.post("/compute_metrics/")
async def compute_metrics(
    file: UploadFile = File(...),
    qi: List[str] = Form(...),
    sa: List[str] = Form(...)
) -> Dict[str, Any]:
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    k_anonymity_metric = k_anonymity(df, qi).to_dict()
    l_diversity_metric = {key: tuple(value) for key, value in l_diversity(df, qi, sa).items()}
    
    return {
        "k_anonymity": k_anonymity_metric,
        "l_diversity": l_diversity_metric
    }



@app.post("/attribute_attack_score/")
async def calculate_attack_score(
    attribute_attack: str = Form(...),  # The feature to attack
    file: UploadFile = File(...),  # The uploaded CSV file
    attack_train_ratio: Optional[float] = Form(0.5),  # Attack training ratio (default to 0.5)
    metric_fn: Optional[str] = Form("accuracy"),  # Metric function, default to "accuracy"
):
    # Read the uploaded CSV file into a pandas DataFrame
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode('utf-8')))  # Use StringIO to convert byte data to a file-like object

    # Separate features and target variable from the CSV file
    X = df.iloc[:, :-1]  # All columns except the last one (features)
    y = df.iloc[:, -1]   # The last column (target)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-attack_train_ratio, random_state=42)

    # Check if the target variable is continuous or categorical
    is_continuous = y_train.dtype.kind in ['f', 'i'] and len(y_train.unique()) > 2

    # Adjust the metric_fn based on whether the task is classification or regression
    if is_continuous:
        # For regression, we can use metrics like MSE, MAE, or R2
        if metric_fn == "mean_squared_error":
            metric_fn = mean_squared_error
        elif metric_fn == "mean_absolute_error":
            metric_fn = mean_absolute_error
        elif metric_fn == "r2_score":
            metric_fn = r2_score
        else:
            return {"error": "For continuous target variables, use 'mean_squared_error', 'mean_absolute_error', or 'r2_score'."}
    else:
        # For classification, use accuracy or F1 score
        if metric_fn == "accuracy":
            metric_fn = accuracy_score
        elif metric_fn == "f1":
            metric_fn = f1_score
        else:
            return {"error": "For categorical targets, only 'accuracy' or 'f1' are supported."}

    # Now call the attribute_attack_score function to calculate the attack score
    try:
        score = attribute_attack_score(
            x_train=X_train,  # Pass the train data
            x_test=X_test,    # Pass the test data
            y_train=y_train,  # Pass the train target
            y_test=y_test,    # Pass the test target
            attribute_attack=attribute_attack,  # Feature to attack
            attack_train_ratio=attack_train_ratio,  # Ratio of training data for attack
            metric_fn=metric_fn  # Metric function passed from the form
        )
        return {"message": "Score calculated successfully", "score": score}
    except Exception as e:
        return {"error": str(e)}



# Route to handle CSV file upload and data minimization calculation
@app.post("/data-minimization-score/")
async def calculate_data_minimization_score(file: UploadFile = File(...)):
    # Read the uploaded CSV file
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    # Check if the CSV has necessary columns
    if not all(col in df.columns for col in ['y_true', 'y_pred']):
        return {"error": "CSV must include 'y_true' and 'y_pred' columns."}

    y_true = df['y_true']
    y_pred = df['y_pred']
    
    # Assuming the rest of the columns are different data minimization methods
    y_pred_dm = []
    for col in df.columns:
        if col not in ['y_true', 'y_pred']:
            y_pred_dm.append({
                "selector_type": col,  # Example: using column name as selector_type
                "modifier_type": "Base",  # You can customize this depending on your needs
                "n_feats": len(df[col].dropna()),  # Example: feature count, customize as necessary
                "feats": df.columns.tolist(),  # List of all feature names
                "predictions": df[col].tolist()  # Predictions for this minimization method
            })
    
    # Calculate the data minimization score
    score = data_minimization_score(y_true, y_pred, y_pred_dm)
    return {"data_minimization_score": score}



@app.post("/calculate_risk_score/")
async def calculate_risk_score(file: UploadFile = File(...)):
    try:
        # Read the CSV file content
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Convert string representation of lists into actual lists
        df["shadow_train_probs"] = df["shadow_train_probs"].apply(ast.literal_eval)
        df["shadow_test_probs"] = df["shadow_test_probs"].apply(ast.literal_eval)
        df["target_train_probs"] = df["target_train_probs"].apply(ast.literal_eval)
        
        # Extract the necessary data for risk score calculation
        shadow_train = (df["shadow_train_probs"].tolist(), df["shadow_train_labels"].tolist())
        shadow_test = (df["shadow_test_probs"].tolist(), df["shadow_test_labels"].tolist())
        target_train = (df["target_train_probs"].tolist(), df["target_train_labels"].tolist())
        
        # Call the function to calculate the privacy risk score
        risk_scores = privacy_risk_score(shadow_train, shadow_test, target_train)
        
        return {"risk_scores": risk_scores.tolist()}
    
    except Exception as e:
        return {"detail": f"Error calculating risk score: {str(e)}"}



@app.post("/compute_shapr_score_from_csv/")
async def compute_shapr_score_from_csv(
    file: UploadFile = File(...), 
    batch_size: int = 500, 
    train_size: float = 1.0, 
    aggregated: bool = True  # This will control whether to aggregate the results
):
    # Read the uploaded CSV file
    content = await file.read()
    csv_data = StringIO(content.decode("utf-8"))
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_data)

    # Ensure the CSV contains the necessary columns (for example: y_train, y_test, y_pred_train, y_pred_test)
    if not all(col in df.columns for col in ["y_train", "y_test", "y_pred_train", "y_pred_test"]):
        return {"error": "CSV must contain 'y_train', 'y_test', 'y_pred_train', and 'y_pred_test' columns."}

    # Convert the relevant columns into pandas Series
    y_train = df["y_train"]
    y_test = df["y_test"]
    y_pred_train = df["y_pred_train"]
    y_pred_test = df["y_pred_test"]
    
    # Compute SHAPr score
    score = shapr_score(y_train, y_test, y_pred_train, y_pred_test, batch_size, train_size)

    # If aggregated is True, return the average of the scores
    if aggregated:
        # Return aggregated score (mean of the score across all samples)
        return {"shapr_score": score.tolist()}
    else:
        # Return non-aggregated score (raw score for each sample)
        return {"shapr_score": score.tolist()}




@app.post("/check_output_type_from_csv/")
async def check_output_type_from_csv(file: UploadFile = File(...)):
    # Read the uploaded CSV file
    content = await file.read()
    csv_data = StringIO(content.decode("utf-8"))
    
    # Load the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_data)

    # Ensure the CSV contains the necessary target column 'y_true'
    if "y_true" not in df.columns:
        return {"error": "CSV must contain 'y_true' column."}

    # Convert the relevant column into pandas Series
    y_true = df["y_true"]

    try:
        # Check the output type using the imported function
        output_type = check_valid_output_type(y_true)
        return {"output_type": output_type}
    except ValueError as e:
        # If the check fails, return the error message
        return {"error": str(e)}

