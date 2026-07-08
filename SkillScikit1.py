#Base level model architecture
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('/content/swag_gamer_data.csv')

df = df[['Kills','Swag_Score','Hours_Played','Skill_Level']].dropna()

X = df[['Kills','Hours_Played','Swag_Score']]
Y = df['Skill_Level']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

Scaler = StandardScaler()
X_train_scaled = Scaler.fit_transform(X_train)
X_test_scaled = Scaler.transform(X_test)

model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, Y_train)

predictions = model.predict(X_test_scaled)
mae = mean_absolute_error(Y_test, predictions)

print(f"Mean Absolute Error in Skill_Level = {mae:.2f}")
