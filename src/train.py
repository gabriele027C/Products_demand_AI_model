from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLUMNS, TARGET

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "restaurant_demand_dataset_raw.csv"


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    data = df[FEATURE_COLUMNS + [TARGET]].dropna()

    X = data[FEATURE_COLUMNS]
    y = data[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("estimator", LinearRegression()),
        ]
    )

    # Evaluate generalisation with cross-validation on the training split.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "mae": "neg_mean_absolute_error",
        "mse": "neg_mean_squared_error",
        "r2": "r2",
    }
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=None,
        error_score="raise",
    )

    cv_mae = -cv_results["test_mae"]
    cv_mse = -cv_results["test_mse"]
    cv_rmse = cv_mse ** 0.5
    cv_r2 = cv_results["test_r2"]

    print("Cross-validation (5-fold) metrics on training split:")
    print(f"  MAE  mean={cv_mae.mean():.2f} std={cv_mae.std():.2f}")
    print(f"  RMSE mean={cv_rmse.mean():.2f} std={cv_rmse.std():.2f}")
    print(f"  R2   mean={cv_r2.mean():.3f} std={cv_r2.std():.3f}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("Hold-out test metrics:")
    print(f"  MAE : {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  R2  : {r2:.3f}")


if __name__ == "__main__":
    main()
