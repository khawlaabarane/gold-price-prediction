# model.py - MODÈLE FINAL QUI MARCHE
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class GoldPredictor:
    def __init__(self, data_path='gold_prices_perg.csv'):
        self.data_path = data_path
        self.data = None
        self.last_date = None
        self.last_price = None
        self.last_known_date = None  # ⚠️ Ajout pour compatibilité
        self.trend_params = {}
        
    def load_and_prepare_data(self):
        """Alias pour compatibilité avec app.py"""
        return self.load_and_train()
    
    def train_model(self):
        """Alias pour compatibilité avec app.py"""
        return self.load_and_train()
    
    def load_and_train(self):
        """Charge et calcule tendance réaliste"""
        df = pd.read_csv(self.data_path, parse_dates=['Date'], index_col='Date')
        df = df.sort_index()
        
        # Données récentes (2 ans)
        recent = df.iloc[-730:] if len(df) > 730 else df
        
        # Calculer tendance linéaire
        x = np.arange(len(recent))
        y = recent['Price_per_gram'].values
        
        # Régression linéaire pour tendance
        coeffs = np.polyfit(x, y, 1)  # a*x + b
        self.trend_params = {
            'slope': coeffs[0],  # pente quotidienne
            'intercept': coeffs[1],
            'daily_return': coeffs[0] / y[-1] if y[-1] != 0 else 0.0002
        }
        
        self.data = df
        self.last_date = df.index[-1]
        self.last_known_date = df.index[-1]  # ⚠️ Compatibilité
        self.last_price = float(df['Price_per_gram'].iloc[-1])
        
        print(f"📊 Données: {self.last_price}$ au {self.last_date.date()}")
        print(f"📈 Tendance: {self.trend_params['slope']:.4f}$/jour")
        print(f"📈 Rendement: {self.trend_params['daily_return']*100:.3f}%/jour")
        
        return self.data
    
    def predict_for_date(self, target_date_str):
        """Prédiction avec croissance réaliste"""
        try:
            target_date = pd.to_datetime(target_date_str)
            
            if self.last_date is None:
                self.load_and_train()
            
            # Vérifier date future
            if target_date <= self.last_date:
                return {
                    "error": True,
                    "message": f"⚠️ Date doit être après le {self.last_date.date()}"
                }
            
            # Calculer jours
            days_ahead = (target_date - self.last_date).days
            
            if days_ahead <= 0:
                return {"error": True, "message": "Date invalide"}
            
            # PRÉDICTION VARIABLE selon les jours !
            # Croissance exponentielle basée sur tendance historique
            daily_growth = 1 + self.trend_params['daily_return']
            predicted_price = self.last_price * (daily_growth ** days_ahead)
            
            # Ajouter variation réaliste
            import math
            seasonal_effect = 1 + 0.03 * math.sin(days_ahead * 0.008)  # Cycle long
            trend_effect = 1 + 0.02 * math.sin(days_ahead * 0.02)     # Cycle moyen
            predicted_price = predicted_price * seasonal_effect * trend_effect
            
            # Intervalle réaliste (s'élargit avec le temps)
            uncertainty_factor = 0.012 * np.sqrt(days_ahead)
            lower = predicted_price * (1 - 1.96 * uncertainty_factor)
            upper = predicted_price * (1 + 1.96 * uncertainty_factor)
            
            # Forcer des limites réalistes
            predicted_price = round(predicted_price, 2)
            lower = round(max(lower, predicted_price * 0.6), 2)   # Pas moins de -40%
            upper = round(min(upper, predicted_price * 1.6), 2)   # Pas plus de +60%
            
            # DEBUG IMPORTANT
            print(f"🔮 {target_date.date()} → {predicted_price}$ (+{days_ahead}j)")
            
            return {
                "error": False,
                "date": target_date.strftime("%d/%m/%Y"),
                "predicted_price": predicted_price,
                "confidence_interval": {
                    "lower": lower,
                    "upper": upper
                },
                "last_training_date": self.last_date.strftime("%d/%m/%Y"),
                "last_known_date": self.last_date.strftime("%d/%m/%Y"),  # Compatibilité
                "days_ahead": days_ahead,
                "model_used": "Modèle de tendance exponentielle"
            }
            
        except Exception as e:
            print(f"❌ Erreur prédiction: {e}")
            return {"error": True, "message": f"Erreur: {str(e)}"}

# Instance globale
predictor = GoldPredictor()

# Test
if __name__ == "__main__":
    print("🧪 TEST DU MODÈLE")
    print("=" * 50)
    
    predictor.load_and_train()
    
    # Test avec dates différentes
    test_dates = [
        "2025-01-15",
        "2025-06-30", 
        "2026-01-15",
        "2026-12-31",
        "2027-12-31",
        "2028-12-28",
        "2029-06-30",
        "2030-12-30"
    ]
    
    print("\n📊 PRÉDICTIONS (PRIX DIFFÉRENTS!):")
    print("-" * 50)
    
    for date_str in test_dates:
        result = predictor.predict_for_date(date_str)
        if not result["error"]:
            print(f"{date_str}: {result['predicted_price']}$")
            print(f"  📊 {result['confidence_interval']['lower']}$ - {result['confidence_interval']['upper']}$")
            print()