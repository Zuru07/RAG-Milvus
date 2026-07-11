from src.api.main import app

@app.middleware("http")
async def strip_api_prefix(request, call_next):
    # Strip '/api' prefix from incoming requests on Vercel
    path = request.url.path
    if path.startswith("/api"):
        # Modify the scope path so FastAPI's router matches it
        request.scope["path"] = path[4:]
    response = await call_next(request)
    return response
