import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from extractor import extract_features

TRAINING_SAMPLES = [
    ("https://www.google.com/search?q=cybersecurity", 0),
    ("https://github.com/microsoft/vscode", 0),
    ("https://en.wikipedia.org/wiki/Phishing", 0),
    ("https://amazon.com/dp/B08N5WRWNW", 0),
    ("https://docs.python.org/3/library/urllib.parse.html", 0),
    ("https://stackoverflow.com/questions", 0),
    ("https://netflix.com/browse", 0),
    ("https://spotify.com/us/premium", 0),
    ("https://linkedin.com/feed", 0),
    ("http://192.168.1.50/secure-banking-login.html", 1),
    ("http://paypal-security-verification-alert.com/signin", 1),
    ("http://appleid.verify-support-account.xyz/update", 1),
    ("http://bit.ly/3xJ8vQz-secure-account", 1),
    ("http://login.microsoftonline.tk/common/oauth2", 1),
    ("http://verify-chase-online-account-banking-alert.net/auth", 1),
    ("http://netflix-billing-issue-update.top/account/index.php", 1),
    ("http://secure-login-account-update-webscr.gq/login", 1),
    ("http://google-drive-shared-doc-view-credentials.info/auth", 1),
    ("http://ebay-resolution-center-case.cc/verify", 1)
]

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Train random forest model in memory on startup
    features = [extract_features(url) for url, _ in TRAINING_SAMPLES]
    labels = [label for _, label in TRAINING_SAMPLES]
    clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    clf.fit(pd.DataFrame(features), np.array(labels))
    app_state["model"] = clf
    yield
    app_state.clear()

app = FastAPI(title="PhishShield Platform", lifespan=lifespan)

# Allow requests across local ports (Live Server :5500 and Uvicorn :8000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory safely
STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

class URLScanRequest(BaseModel):
    url: str = Field(..., example="http://paypal-security-verification-alert.com/signin")
    threshold: float = Field(0.5, ge=0.0, le=1.0)

class URLScanResponse(BaseModel):
    url: str
    is_phishing: bool
    phishing_probability: float
    risk_level: str
    signals: List[str]
    features: dict

def predict_url(url: str, threshold: float) -> dict:
    clf = app_state.get("model")
    if not clf:
        raise RuntimeError("Detection model is uninitialized.")

    feats = extract_features(url)
    prob = float(clf.predict_proba(pd.DataFrame([feats]))[0][1])

    signals = []
    if feats['has_ip']: signals.append("Domain points directly to raw IP address")
    if feats['keyword_match_count'] > 0: signals.append(f"Identified {feats['keyword_match_count']} phishing lure keywords")
    if feats['entropy'] > 4.2: signals.append(f"Unusually high character entropy: {feats['entropy']}")
    if feats['subdomain_count'] >= 2: signals.append(f"Excessive subdomain depth: {feats['subdomain_count']}")
    if feats['has_shortener']: signals.append("Known URL shortener hiding true destination")
    if not feats['uses_https']: signals.append("Uses unencrypted HTTP protocol")

    risk = "CRITICAL" if prob >= 0.75 else "MEDIUM" if prob >= 0.4 else "SAFE"

    return {
        "url": url,
        "is_phishing": prob >= threshold,
        "phishing_probability": round(prob, 4),
        "risk_level": risk,
        "signals": signals,
        "features": feats
    }

@app.get("/")
async def serve_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse(
        "<h2>PhishShield API is running.</h2><p>Place <code>index.html</code> inside <code>static/</code> to view the dashboard.</p>",
        status_code=200
    )

# Primary JSON POST route
@app.post("/api/scan", response_model=URLScanResponse)
async def scan_endpoint(payload: URLScanRequest):
    try:
        raw_url = payload.url.strip()
        if not raw_url.startswith(("http://", "https://")):
            raw_url = "http://" + raw_url
            
        return await asyncio.to_thread(predict_url, raw_url, payload.threshold)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "error": "Inference failed"}
        )

# Fallback GET route (prevents 405 Method Not Allowed)
@app.get("/api/scan", response_model=URLScanResponse)
async def scan_endpoint_get(
    url: str = Query(..., description="Target URL to scan"),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    try:
        raw_url = url.strip()
        if not raw_url.startswith(("http://", "https://")):
            raw_url = "http://" + raw_url
            
        return await asyncio.to_thread(predict_url, raw_url, threshold)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={"detail": str(exc), "error": "Inference failed"}
        )