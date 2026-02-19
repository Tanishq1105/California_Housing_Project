from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def load_split_data(seed=42):
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame

    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
