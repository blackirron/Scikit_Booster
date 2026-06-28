#Lower error using XGBoost
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor

df = pd.read_csv('/content/swag_gamer_data.csv')

df = df[['Kills','Swag_Score','Hours_Played','Skill_Level','Game']].dropna()

X = df[['Kills','Hours_Played','Swag_Score','Game']]
Y = df['Skill_Level']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

numeric_features = ['Kills','Hours_Played','Swag_Score']
categorical_features = ['Game']

preprocessor = ColumnTransformer(
    transformers=[
        ('num',StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

boosting_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model',LGBMRegressor(random_state=42, verbose=-1))
])

param_grid = {
    'model__n_estimators': [50,100],
    'model__learning_rate': [0.01, 0.1],
    'model__max_depth': [3, 5, -1]
}

grid_search = GridSearchCV(
    estimator=boosting_pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

grid_search.fit(X_train, Y_train)

print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best Cross-Validation MAE: {-grid_search.best_score_:.2f}")

final_predictions = grid_search.predict(X_test)
final_mae = mean_absolute_error(Y_test, final_predictions)

print(f"Final Test MAE: {final_test_mae:.2f}")
