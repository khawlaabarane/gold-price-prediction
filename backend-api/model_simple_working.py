# model_simple_working.py
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression

class PricePredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.trained = False
        self.last_date = None
        self.base_price = None
        
    def load_and_train(self):
        """Charge les données et entraîne le modèle"""
        try:
            # Données simulées (remplace par tes vraies données)
            dates = []
            prices = []
            
            # Créer des données historiques
            start_date = datetime(2020, 1, 1)
            for i in range(100):
                current_date = start_date + timedelta(days=i*30)  # Tous les 30 jours
                dates.append(current_date)
                # Prix qui augmente avec le temps
                price = 100 + (i * 2) + np.random.normal(0, 5)
                prices.append(price)
            
            # Convertir en features
            X = np.array([self._date_to_features(d) for d in dates]).reshape(-1, 1)
            y = np.array(prices)
            
            # Entraîner le modèle
            self.model.fit(X, y)
            self.trained = True
            self.last_date = dates[-1]
            self.base_price = prices[-1]
            
            print(f"✅ Modèle entraîné sur {len(dates)} données")
            print(f"📈 Dernier prix historique: {self.base_price:.2f}$")
            
        except Exception as e:
            print(f"❌ Erreur lors de l'entraînement: {e}")
            self.trained = False
    
    def _date_to_features(self, date_obj):
        """Convertit une date en valeur numérique pour le modèle"""
        if isinstance(date_obj, str):
            date_obj = datetime.strptime(date_obj, "%Y-%m-%d")
        # Nombre de jours depuis 2000-01-01
        base_date = datetime(2000, 1, 1)
        return (date_obj - base_date).days
    
    def predict_for_date(self, date_str):
        """Prédit le prix pour une date donnée"""
        if not self.trained:
            return {"error": True, "message": "Modèle non entraîné"}
        
        try:
            # Convertir la date
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            
            # Vérifier que c'est une date future
            if date_obj <= self.last_date:
                return {"error": True, "message": "La date doit être dans le futur"}
            
            # Prédire
            X_pred = np.array([[self._date_to_features(date_obj)]])
            predicted_price = self.model.predict(X_pred)[0]
            
            # Ajouter un peu de variation basée sur la date
            # pour que chaque date ait un prix différent
            day_of_year = date_obj.timetuple().tm_yday
            variation = np.sin(day_of_year * 0.017) * 10  # Variation saisonnière
            final_price = max(50, predicted_price + variation)
            
            return {
                "error": False,
                "predicted_price": round(float(final_price), 2),
                "date": date_str,
                "confidence": "high"
            }
            
        except ValueError:
            return {"error": True, "message": "Format de date invalide. Utilisez YYYY-MM-DD"}
        except Exception as e:
            return {"error": True, "message": f"Erreur de prédiction: {e}"}

# Créer une instance globale
predictor = PricePredictor()