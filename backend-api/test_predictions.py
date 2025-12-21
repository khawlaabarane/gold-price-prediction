# test_predictions.py - Vérifie que les prix sont différents
from model_simple_working import predictor
import sys

print("🧪 VÉRIFICATION : Les prix sont-ils différents ?")
print("=" * 60)

# Initialiser
predictor.load_and_train()

# Liste de dates à tester
dates_to_test = [
    "2025-01-15",
    "2025-06-15", 
    "2026-01-15",
    "2026-12-15",
    "2027-12-15",
    "2028-12-15",
    "2029-12-15",
    "2030-12-15"
]

print("\n📅 Test de différentes dates futures :")
print("-" * 60)

results = []
for date_str in dates_to_test:
    result = predictor.predict_for_date(date_str)
    if not result["error"]:
        price = result["predicted_price"]
        results.append((date_str, price))
        print(f"{date_str}: {price}$")

# Vérifier si tous les prix sont différents
prices = [p for _, p in results]
unique_prices = set(prices)

print("\n" + "=" * 60)
print(f"📊 Résultat : {len(results)} prédictions")
print(f"🎯 Prix uniques : {len(unique_prices)} sur {len(results)}")

if len(unique_prices) == len(results):
    print("✅ SUCCÈS : Tous les prix sont différents !")
    sys.exit(0)
else:
    print("❌ ÉCHEC : Certains prix sont identiques !")
    print(f"Prix identiques trouvés :")
    for i in range(len(prices)):
        for j in range(i+1, len(prices)):
            if prices[i] == prices[j]:
                print(f"  - {results[i][0]} = {results[j][0]} = {prices[i]}$")
    sys.exit(1)