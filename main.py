import os
import shutil
import uuid
from datetime import datetime, timedelta
from typing import Optional

# FastAPI and dependencies
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from jose import JWTError, jwt

# Database (SQLAlchemy)
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# Import ML components
from ml_engine import FeatureExtractor, SearchIndex

# --- CONFIGURATION ---
SECRET_KEY = "YOUR_SUPER_SECRET_KEY_CHANGE_THIS_IN_PROD"  # IMPORTANT: Change this!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
UPLOAD_DIR = "static/uploads"
# Ensure the uploads folder exists
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- DATABASE SETUP (SQLite) ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./app.db"
# connect_args={"check_same_thread": False} is required for SQLite with FastAPI
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class ImageLog(Base):
    __tablename__ = "image_logs"
    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    username = Column(String)

# Create tables in the database file (app.db) if they don't exist
Base.metadata.create_all(bind=engine)

# --- SECURITY UTILITIES ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# --- APP INIT ---
app = FastAPI(title="Similar Images Finder")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize AI Engine and Index
# These objects load the model and existing index data when the app starts
ml_model = FeatureExtractor()
search_index = SearchIndex()

# --- DEPENDENCIES ---
def get_db():
    """Dependency to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(request: Request):
    """Dependency to extract user identity from JWT cookie."""
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        # Check for 'Bearer ' prefix and strip it if necessary
        if token.startswith("Bearer "):
            token = token.split(" ")[1]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return None
        return username
    except JWTError:
        return None

# --- WEB PAGE ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, user: Optional[str] = Depends(get_current_user)):
    """Serves the main search page, redirects to login if unauthenticated."""
    if not user:
        return RedirectResponse(url="/login")
    return templates.TemplateResponse("index.html", {"request": request, "username": user})

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serves the login/registration page."""
    return templates.TemplateResponse("login.html", {"request": request})

# --- AUTHENTICATION ENDPOINTS ---

@app.post("/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    """Handles user login and issues a JWT token as a cookie."""
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    
    # Set the token as a secure cookie and redirect to home
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@app.post("/register")
async def register(username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    """Handles user registration by hashing the password and storing in the DB."""
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    new_user = User(username=username, hashed_password=get_password_hash(password))
    db.add(new_user)
    db.commit()
    return RedirectResponse(url="/login", status_code=303)

@app.get("/logout")
async def logout():
    """Clears the access token cookie and redirects to login."""
    response = RedirectResponse(url="/login")
    response.delete_cookie("access_token")
    return response

# --- CORE LOGIC: IMAGE ENDPOINTS ---

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), 
                       user: Optional[str] = Depends(get_current_user),
                       db: Session = Depends(get_db)):
    """
    Saves an image, extracts features, adds the vector to the FAISS index,
    and logs the action to the database.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # 1. Save File
    file_ext = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, unique_filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Extract Features using ResNet-50
    features = ml_model.extract_features(file_path)
    
    # 3. Add to FAISS Index
    if features is not None:
        search_index.add_image(unique_filename, features)
        
        # 4. Log to DB
        log = ImageLog(filename=unique_filename, username=user)
        db.add(log)
        db.commit()
        
    return {"message": "Image indexed successfully", "filename": unique_filename}

@app.post("/search")
async def search_image(file: UploadFile = File(...), 
                       user: Optional[str] = Depends(get_current_user)):
    """
    Processes a query image, searches the FAISS index for similar vectors,
    and returns the top results.
    """
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # 1. Save Query Image Temporarily
    temp_filename = f"temp_{uuid.uuid4()}.jpg"
    temp_path = os.path.join(UPLOAD_DIR, temp_filename)
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 2. Extract Features from query image
    query_features = ml_model.extract_features(temp_path)
    
    # 3. Search FAISS index for matches
    results = search_index.search(query_features, k=5)
    
    # Cleanup temp file (important!)
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    return {"results": results}
