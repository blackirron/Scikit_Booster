#using OnhHotEncoding for categorical features
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('/content/swag_gamer_data.csv')

df = df[['Kills', 'Hours_Played', 'Swag_Score', 'Game', 'Skill_Level']].dropna()

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

full_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model',RandomForestRegressor(random_state=42))
])

full_pipeline.fit(X_train, Y_train)
predictions = full_pipeline.predict(X_test)
mae = mean_absolute_error(Y_test, predictions)

print(f"Mean Absolute Error = {mae:.2f}")
