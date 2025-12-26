from fastapi import FastAPI
from app.routers import users, items
from app.internal import admin

app = FastAPI(title="Vero Backend API", version="0.1.0")

app.include_router(users.router)
app.include_router(items.router)
app.include_router(
    admin.router,
    prefix="/admin",
    tags=["admin"],
)


@app.get("/")
async def root():
    return {"message": "Welcome to Vero Backend API"}
