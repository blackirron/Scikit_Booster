#try to include the game column for different skill level according to the game
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer

df = pd.read_csv('/content/swag_gamer_data.csv')

df = df[['Game','Kills','Swag_Score','Hours_Played','Skill_Level']].dropna()

X = df[['Game','Kills','Hours_Played','Swag_Score']]
Y = df['Skill_Level']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Kills','Hours_Played','Swag_Score']),
        ('cat',OneHotEncoder(handle_unknown='ignore'),['Game'])
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

final_model = RandomForestRegressor(max_depth=5,n_estimators=100,random_state=42)
final_model.fit(X_train_processed, Y_train)

predictions = final_model.predict(X_test_processed)
mae = mean_absolute_error(Y_test, predictions)

print(f"Mean Absolute Error = {mae:.2f}")
