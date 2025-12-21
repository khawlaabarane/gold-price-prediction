# model.py - VERSION CORRIGÉE
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

class GoldPredictor:
    def __init__(self, data_path='gold_prices_perg.csv'):
        self.data_path = data_path
        self.model = None  # Stocke le modèle, pas model_fit
        self.last_known_date = None
        self.df = None
        self.full_model_fit = None  # Modèle entraîné sur TOUTES les données
        
    def load_and_prepare_data(self):
        """Charge les données"""
        df = pd.read_csv(self.data_path, parse_dates=['Date'], index_col='Date')
        data = df['Price_per_gram']
        data = data.asfreq('D').ffill()
        self.df = data.dropna()
        self.last_known_date = self.df.index[-1]
        print(f"✅ Données chargées : {len(self.df)} lignes")
        print(f"📅 Dernière date : {self.last_known_date.date()}")
        return self.df
    
    def train_model(self, order=(1, 1, 1)):
        """Entraîne sur TOUTES les données"""
        if self.df is None:
            self.load_and_prepare_data()
        
        print(f"⚙️  Entraînement ARIMA{order} sur TOUTES les données...")
        
        # Entraîner une seule fois sur toutes les données
        self.full_model_fit = ARIMA(self.df, order=order).fit()
        
        print(f"✅ Modèle entraîné sur {len(self.df)} points")
        return self.full_model_fit
    
    def predict_for_date(self, target_date_str):
        """PRÉDIT CORRECTEMENT - Version simplifiée"""
        try:
            target_date = pd.to_datetime(target_date_str)
            
            # Vérifier date future
            if target_date <= self.last_known_date:
                return {
                    "error": True,
                    "message": f"⚠️ Date doit être après le {self.last_known_date.date()}"
                }
            
            # Calculer jours à prédire
            days_ahead = (target_date - self.last_known_date).days
            
            if days_ahead <= 0:
                return {"error": True, "message": "Date invalide"}
            
            # Utiliser le modèle déjà entraîné
            forecast = self.full_model_fit.get_forecast(steps=days_ahead)
            predicted_series = forecast.predicted_mean
            
            # Récupérer le prix à la date exacte
            predicted_price = predicted_series.iloc[-1]  # Dernière prédiction
            conf_int = forecast.conf_int().iloc[-1]  # Dernier intervalle
            
            # DEBUG - Affiche dans le terminal
            print(f"🔍 {target_date.date()} → {predicted_price:.2f}$ (+{days_ahead}j)")
            
            return {
                "error": False,
                "date": target_date.strftime("%d/%m/%Y"),
                "predicted_price": round(float(predicted_price), 2),
                "confidence_interval": {
                    "lower": round(float(conf_int[0]), 2),
                    "upper": round(float(conf_int[1]), 2)
                },
                "last_training_date": self.last_known_date.strftime("%d/%m/%Y"),
                "days_ahead": days_ahead,
                "model_used": "ARIMA(1,1,1)"
            }
            
        except Exception as e:
            print(f"❌ ERREUR: {str(e)}")
            return {
                "error": True,
                "message": f"❌ Erreur : {str(e)}"
            }

# Instance globale
predictor = GoldPredictor()

# Test
if __name__ == "__main__":
    print("🧪 Test des prédictions...")
    predictor.load_and_prepare_data()
    predictor.train_model()
    
    # Test avec différentes dates
    test_dates = ["2025-12-31", "2026-06-30", "2027-12-31", "2028-12-28", "2030-12-30"]
    
    for date in test_dates:
        result = predictor.predict_for_date(date)
        if not result["error"]:
            print(f"📅 {date}: {result['predicted_price']}$")
            print(f"   📊 Intervalle: {result['confidence_interval']['lower']}$ - {result['confidence_interval']['upper']}$")
            print(f"   ⏱️ {result['days_ahead']} jours")
            print()