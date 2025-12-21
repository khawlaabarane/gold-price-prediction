
# app.py - VERSION COMPLÈTE CORRIGÉE
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from model import predictor

app = FastAPI(title="Prédiction Prix de l'Or", version="1.0")
templates = Jinja2Templates(directory="templates")

@app.on_event("startup")
async def startup_event():
    print("🚀 Démarrage de l'API...")
    predictor.load_and_train()
    print(f"📅 Dernière date: {predictor.last_date.strftime('%d/%m/%Y')}")
    print("✅ API prête → http://127.0.0.1:8000")

# Page d'accueil
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    last = predictor.last_date.strftime("%d/%m/%Y") if predictor.last_date else "N/A"
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "last_date": last,
            "last_price": predictor.last_price
        }
    )

# Redirection pour /predict en GET
@app.get("/predict")
async def redirect_predict():
    """Redirige vers l'accueil si on essaie d'accéder à /predict en GET"""
    return RedirectResponse(url="/", status_code=302)

# Prédiction (POST uniquement)
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, date: str = Form(...)):
    # Faire la prédiction
    result = predictor.predict_for_date(date)
    
    # Préparer le contexte
    context = {
        "request": request, 
        "last_date": predictor.last_date.strftime("%d/%m/%Y"),
        "last_price": predictor.last_price,
        "date_entered": date
    }
    
    if result.get("error"):
        context["error"] = result["message"]
    else:
        context.update({
            "prediction": True,
            "date": result["date"],
            "price": result["predicted_price"],
            "lower": result["confidence_interval"]["lower"],
            "upper": result["confidence_interval"]["upper"],
            "days_ahead": result["days_ahead"],
            "model": result["model_used"]
        })
    
    return templates.TemplateResponse("index.html", context)

# API REST
@app.get("/api/predict")
async def api_predict(date: str):
    return predictor.predict_for_date(date)

# Santé de l'API
@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": "GoldPredictor",
        "last_date": predictor.last_date.strftime("%Y-%m-%d") if predictor.last_date else None,
        "last_price": predictor.last_price
    }

# Documentation
@app.get("/docs")
async def custom_docs():
    """Redirige vers la documentation FastAPI"""
    return RedirectResponse(url="/docs")