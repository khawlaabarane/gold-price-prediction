# test_arima_direct.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

print("🧪 TEST DIRECT ARIMA")
print("=" * 50)

# 1. Charger tes données
df = pd.read_csv('gold_prices_perg.csv', parse_dates=['Date'], index_col='Date')
data = df['Price_per_gram']
data = data.asfreq('D').ffill()
data = data.dropna()

print(f"📊 Données: {len(data)} points")
print(f"📅 Période: {data.index[0].date()} → {data.index[-1].date()}")
print(f"💰 Prix actuel: {data.iloc[-1]:.2f}$")

# 2. Afficher les données
plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title("Évolution du prix de l'or")
plt.show()

# 3. Tester différents ordres ARIMA
print("\n🔍 Test de différents ordres ARIMA:")
orders_to_test = [(1,1,1), (1,1,0), (0,1,1), (2,1,2), (0,1,0)]

for order in orders_to_test:
    print(f"\nTesting ARIMA{order}:")
    try:
        # Diviser train/test
        train = data.iloc[:-100]  # Tout sauf 100 derniers jours
        test = data.iloc[-100:]   # 100 derniers jours
        
        # Entraîner
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        
        # Prédire
        forecast = model_fit.get_forecast(steps=100)
        predictions = forecast.predicted_mean
        
        # Calculer erreur
        error = np.mean(np.abs(predictions.values - test.values))
        
        print(f"   MAE: {error:.2f}$")
        print(f"   Prix prédit demain: {predictions.iloc[0]:.2f}$")
        print(f"   Prix prédit dans 100j: {predictions.iloc[-1]:.2f}$")
        
        # Vérifier si prédiction varie
        if predictions.iloc[0] != predictions.iloc[-1]:
            print(f"   ✅ Prédictions varient!")
        else:
            print(f"   ❌ Prédictions plates!")
            
    except Exception as e:
        print(f"   ❌ Erreur: {str(e)[:50]}...")

# 4. Tester prédiction longue
print("\n🔮 Test prédiction 5 ans:")
best_order = (1,1,1)  # À changer selon résultats ci-dessus
model = ARIMA(data, order=best_order)
model_fit = model.fit()

# Prédire 5 ans (365*5 jours)
forecast = model_fit.get_forecast(steps=365*5)
predictions = forecast.predicted_mean

print(f"\nPrédictions avec ARIMA{best_order}:")
print(f"Demain: {predictions.iloc[0]:.2f}$")
print(f"Dans 1 an: {predictions.iloc[365]:.2f}$")  
print(f"Dans 2 ans: {predictions.iloc[730]:.2f}$")
print(f"Dans 5 ans: {predictions.iloc[1825]:.2f}$")

# Vérifier si ça varie
if len(set(round(p, 2) for p in predictions.iloc[::365])) > 1:
    print("✅ Les prédictions varient dans le temps!")
else:
    print("❌ Toutes les prédictions sont identiques!")
    
# Afficher graphique
plt.figure(figsize=(12, 6))
plt.plot(data.index[-500:], data.values[-500:], label='Historique (500 derniers jours)')
plt.plot(predictions.index, predictions.values, 'r--', label='Prédiction 5 ans')
plt.title("Prédiction ARIMA - Vérification")
plt.legend()
plt.show()