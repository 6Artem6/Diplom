from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from routes.graph import router

app = FastAPI(title="Site-GPT Element Graph API")

app.include_router(router, prefix="/api", tags=["graph"])


@app.get("/")
def index():
    return {"message": "SiteGPT is running"}
