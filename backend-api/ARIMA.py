import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet


# 1. Chargement des données
df = pd.read_csv('gold_prices_perg.csv', parse_dates=['Date'], index_col='Date')

data = df['Price_per_gram']

data = data.asfreq('D').ffill()

df = data.dropna()
print(f"Données nettoyées : {len(df)} lignes. Fréquence : {df.index.freq}")

plt.figure(figsize=(12, 6))
plt.plot(data)
plt.title("Évolution du prix de l'or (2010 - 2025)")
plt.xlabel("Date")
plt.ylabel("Price_per_gram ($)")
plt.show()


# 2.Analyse de la Tendance,la Saisonnalité et autocorrelation
#decoposition
result = seasonal_decompose(df, model='multiplicative', period=252)
result.plot()
plt.show()

#elimination de variance
data_log = np.log(df)

#elimination de tendance(differenciation)
data_diff = df.diff(1)
plt.figure(figsize=(10, 5))
data_diff.plot(title="differenciation")
plt.show()

#Test de Stationnarité (Augmented Dickey-Fuller)
result = adfuller(data_diff.dropna())
print("statistiques du Test ADF:", result[0])
print("p-value:", result[1])
print("Valeurs critiques:", result[4])
if result[1] > 0.05:
    print("la serie n'est pas stationnaire")
else:
    print("la serie est stationnaire")

#Test PACF
plt.figure(figsize=(12, 6))
plot_pacf(data_diff.dropna(), lags=40, method='ywm', zero=False, auto_ylims=True) # method='ywm' est recommandé pour les données financières
plt.title("PACF (Autocorrélation Partielle) ")
plt.show()

#Test d'autocorrelation
data_diff_clean = data_diff.dropna() # On utilise data_diff car la série doit être stationnaire pour ce test et dropna pour supprimer les nan creer par la differenciation

plt.figure(figsize=(12, 6))
plot_acf(data_diff_clean, lags=40, zero=False, auto_ylims=True) #40 derniers jours
plt.title("Fonction d'Autocorrélation (ACF) sur les données différenciées")
plt.xlabel("Lags (Retards en jours)")
plt.ylabel("Corrélation")
plt.grid(True)
plt.show()


# 3. Chois du modele
# Séparation Train / Test
train_size = int(len(df) * 0.95)
train, test = df[0:train_size], df[train_size:]

# Entraînement du modèle
model = ARIMA(train, order=(1, 1, 1))
model_fit = model.fit()

# Prédictions
forecast_result = model_fit.get_forecast(steps=len(test))
preds = forecast_result.predicted_mean

preds.index = test.index

# Calcul des erreurs
mae = mean_absolute_error(test, preds)
rmse = np.sqrt(mean_squared_error(test, preds))

print(model_fit.summary())
print(f"\n>>> RÉSULTATS : MAE = {mae:.2f} $ | RMSE = {rmse:.2f} $")

# Visualisation
plt.figure(figsize=(12, 6))
plt.plot(train.index[-100:], train[-100:], label='Train (Fin)')
plt.plot(test.index, test, label='Test (Réel)')
plt.plot(test.index, preds, label='Prédiction', color='red', linestyle='--')
plt.legend()
plt.title("Comparaison Réalité vs Prédiction ARIMA")
plt.show()

# 4. PRÉDICTION FUTURE (DATE CIBLE)
print("\n" + "=" * 50)
print("SYSTÈME DE PRÉDICTION PAR DATE")
print("=" * 50)

# Ré-entraînement du modèle sur TOUT l'historique (Train + Test), C'est nécessaire pour prédire à partir de la toute dernière date connue
print("Mise à jour du modèle avec l'ensemble des données...")
full_model = ARIMA(df, order=(1, 1, 1))
full_model_fit = full_model.fit()

last_known_date = df.index[-1]  # Récupération de la dernière date disponible dans vos données
print(f"Dernière date historique connue : {last_known_date.date()}")

while True:
    # Entrée utilisateur
    user_input = input("\nEntrez une date (format YYYY-MM-DD) ou 'q' pour quitter : ")

    if user_input.lower() == 'q':
        print("Fermeture du programme.")
        break

    try:
        # Conversion de l'entrée en date
        target_date = pd.to_datetime(user_input)

        # Vérification : on ne peut prédire que le futur
        if target_date <= last_known_date:
            print(f"Erreur : Veuillez entrer une date postérieure au {last_known_date.date()}")
            continue

        # Calcul du nombre de jours à prédire (Steps)
        delta = target_date - last_known_date
        steps_to_predict = delta.days

        # Génération de la prévision
        forecast_result = full_model_fit.get_forecast(steps=steps_to_predict)
        predicted_mean = forecast_result.predicted_mean

        # Récupération du prix spécifique à la date entrée
        # predicted_mean est une Série Pandas avec les dates en index
        target_price = predicted_mean.loc[target_date]

        # Intervalle de confiance (Optionnel, pour info)
        conf_int = forecast_result.conf_int().loc[target_date]

        # Affichage du résultat
        print("-" * 40)
        print(f"PRÉDICTION POUR LE : {target_date.date()}")
        print("-" * 40)
        print(f"Prix estimé de l'or : {target_price:.2f} $ / gramme")
        print(f"Intervalle probable : entre {conf_int.iloc[0]:.2f} $ et {conf_int.iloc[1]:.2f} $")
        print("-" * 40)

    except ValueError:
        print("Erreur de format ! Assurez-vous d'écrire la date comme ceci : 2025-12-31")
    except Exception as e:
        print(f"Une erreur inattendue est survenue : {e}")
    # ===== AJOUTE CE CODE À LA FIN DE TON ARIMA.py =====

class GoldPredictorARIMA:
    def __init__(self, data_path='gold_prices_perg.csv'):
        self.data_path = data_path
        self.model_fit = None
        self.last_known_date = None
        self.df = None
        
    def load_and_prepare_data(self):
        df = pd.read_csv(self.data_path, parse_dates=['Date'], index_col='Date')
        data = df['Price_per_gram']
        data = data.asfreq('D').ffill()
        self.df = data.dropna()
        self.last_known_date = self.df.index[-1]
        return self.df
    
    def train_model(self, order=(1, 1, 1)):
        if self.df is None:
            self.load_and_prepare_data()
        
        train_size = int(len(self.df) * 0.95)
        train = self.df[0:train_size]
        
        model = ARIMA(train, order=order)
        self.model_fit = model.fit()
        
        return self.model_fit
    
    def predict_for_date(self, target_date_str):
        try:
            target_date = pd.to_datetime(target_date_str)
            
            if target_date <= self.last_known_date:
                return {"error": True, "message": f"Date après {self.last_known_date.date()}"}
            
            days_ahead = (target_date - self.last_known_date).days
            
            full_model = ARIMA(self.df, order=(1, 1, 1))
            full_model_fit = full_model.fit()
            
            forecast = full_model_fit.get_forecast(steps=days_ahead)
            predicted_price = forecast.predicted_mean.loc[target_date]
            conf_int = forecast.conf_int().loc[target_date]
            
            return {
                "error": False,
                "date": target_date.strftime("%d/%m/%Y"),
                "predicted_price": round(float(predicted_price), 2),
                "confidence_interval": {
                    "lower": round(float(conf_int.iloc[0]), 2),
                    "upper": round(float(conf_int.iloc[1]), 2)
                },
                "last_training_date": self.last_known_date.strftime("%d/%m/%Y"),
                "days_ahead": days_ahead,
                "model_used": "ARIMA(1,1,1)"
            }
            
        except Exception as e:
            return {"error": True, "message": str(e)}

# Instance globale pour l'API
predictor = GoldPredictorARIMA()