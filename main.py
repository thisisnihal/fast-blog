#####################
# Imports
#####################
from fastapi import FastAPI, HTTPException, Depends, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import jwt
import bcrypt
import os
import markdown
from utils.markdown_utils import convert_markdown
import uuid
from pathlib import Path
import math
from collections import defaultdict
from dotenv import load_dotenv

# MongoDB
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import DESCENDING, ASCENDING
import pymongo.errors


#####################
# Configuration
#####################
load_dotenv()
SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 72
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = "fastblog"

# Directory for storing markdown files
POSTS_DIR = Path("posts")
POSTS_DIR.mkdir(exist_ok=True)


#####################
# FastAPI App Setup
#####################
app = FastAPI(title="FastBlog", description="Multi-user blogging platform")

# Mount static files (CSS, JS, images)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 templates (HTML rendering)
templates = Jinja2Templates(directory="templates")


#####################
# Security
#####################
security = HTTPBearer(auto_error=False)


#####################
# Database
#####################
client = AsyncIOMotorClient(MONGODB_URL)
db = client[DATABASE_NAME]


#####################
# Pydantic Models
#####################
class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class PostCreate(BaseModel):
    title: str
    content: str
    tags: List[str] = []
    private: bool = False

class PostUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    tags: Optional[List[str]] = None
    private: Optional[bool] = None


#####################
# Utility Functions
#####################
def create_access_token(data: dict):
    """Generate JWT access token with expiration"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Decode and verify JWT token"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.PyJWTError:
        return None

def hash_password(password: str):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str):
    """Check if given password matches the hash"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

def generate_slug(title: str):
    """Generate a URL-friendly slug from a title"""
    import re
    slug = re.sub(r'[^a-zA-Z0-9\s]', '', title).strip()
    slug = re.sub(r'\s+', '-', slug).lower()
    return slug

def calculate_post_score(post: dict):
    """Calculate ranking score based on freshness, upvotes, and engagement"""
    now = datetime.utcnow()
    created_at = post.get('created_at', now)

    # Time decay (newer posts get higher scores)
    time_diff = (now - created_at).total_seconds() / 3600  # hours
    freshness_score = max(0, 100 - time_diff * 0.5)  # Decay over time

    # Engagement score
    upvotes = post.get('upvotes', 0)
    downvotes = post.get('downvotes', 0)
    views = post.get('views', 0)

    engagement_score = (upvotes * 2) - downvotes + (views * 0.1)
    return freshness_score + engagement_score


#####################
# Dependency Helpers
#####################
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Get user from JWT token if available (returns None if not valid)"""
    if not credentials:
        return None
    payload = verify_token(credentials.credentials)
    if not payload:
        return None
    return await db.users.find_one({"user_id": payload.get("user_id")})

async def get_current_user_required(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Require valid user authentication"""
    user = await get_current_user(credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

def get_logged_in_status(request: Request):
    """Check if user is logged in via cookie"""
    token = request.cookies.get("access_token")
    if token and verify_token(token):
        return True
    return False


#####################
# Event Handlers
#####################
@app.on_event("startup")
async def startup_event():
    """Create MongoDB indexes on startup"""
    await db.users.create_index("username", unique=True)
    await db.users.create_index("email", unique=True)
    await db.posts.create_index("slug", unique=True)
    await db.posts.create_index([("created_at", DESCENDING)])
    await db.posts.create_index([("author", ASCENDING)])


#####################
# Routes - Public Pages
#####################
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page showing top-ranked public posts"""
    posts_cursor = db.posts.find({"private": False})
    posts = await posts_cursor.to_list(None)

    # Rank posts
    for post in posts:
        post['score'] = calculate_post_score(post)
    posts.sort(key=lambda x: x['score'], reverse=True)
    posts = posts[:20]

    # Personalized tag recommendations
    user_tags = []
    token = request.cookies.get("access_token")
    if token:
        payload = verify_token(token)
        if payload:
            user = await db.users.find_one({"user_id": payload.get("user_id")})
            if user and user.get("liked_tags"):
                sorted_tags = sorted(user["liked_tags"].items(), key=lambda x: x[1], reverse=True)
                user_tags = [tag for tag, _ in sorted_tags[:3]]

    return templates.TemplateResponse("home.html", {
        "request": request,
        "title": "Fast Blog - Home",
        "posts": posts,
        "user_tags": user_tags,
        "logged_in": get_logged_in_status(request)
    })


#####################
# Routes - Authentication
#####################
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Render login page"""
    if get_logged_in_status(request):
        return RedirectResponse("/dashboard", status_code=302)
    
    return templates.TemplateResponse("login.html", {
        "request": request,
        "title": "Login",
        "logged_in": False
    })


@app.post("/login")
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    """Handle login form submission"""
    user = await db.users.find_one({"username": username})
    
    if not user or not verify_password(password, user["password"]):
        return templates.TemplateResponse("login.html", {
            "request": request,
            "title": "Login",
            "error": "Invalid username or password",
            "logged_in": False
        })
    
    # Generate token and store in cookie
    token = create_access_token({"user_id": user["user_id"]})
    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie("access_token", token, httponly=True)
    return response


@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    """Render signup page"""
    if get_logged_in_status(request):
        return RedirectResponse("/dashboard", status_code=302)
    
    return templates.TemplateResponse("signup.html", {
        "request": request,
        "title": "Sign Up",
        "logged_in": False
    })


@app.post("/signup")
async def signup(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    """Handle signup form submission"""
    # Check if username or email already exists
    existing_user = await db.users.find_one({"$or": [{"username": username}, {"email": email}]})
    if existing_user:
        error_msg = "Username already exists" if existing_user["username"] == username else "Email already exists"
        return templates.TemplateResponse("signup.html", {
            "request": request,
            "title": "Sign Up",
            "error": error_msg,
            "logged_in": False
        })
    
    # Create user
    user_id = str(uuid.uuid4())
    hashed_password = hash_password(password)
    
    user_doc = {
        "user_id": user_id,
        "username": username,
        "email": email,
        "password": hashed_password,
        "total_reader": 0,
        "total_upvotes": 0,
        "total_downvotes": 0,
        "liked_tags": {},
        "created_at": datetime.utcnow()
    }
    
    await db.users.insert_one(user_doc)
    
    # Auto-login new user
    token = create_access_token({"user_id": user_id})
    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie("access_token", token, httponly=True)
    return response


@app.get("/logout")
async def logout():
    """Logout user by clearing cookie"""
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("access_token")
    return response


#####################
# Routes - Dashboard
#####################
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, page: int = 1):
    """User dashboard showing their own posts"""
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    
    payload = verify_token(token)
    if not payload:
        return RedirectResponse("/login", status_code=302)
    
    user = await db.users.find_one({"user_id": payload.get("user_id")})
    if not user:
        return RedirectResponse("/login", status_code=302)
    
    # Pagination setup
    posts_per_page = 10
    skip = (page - 1) * posts_per_page
    
    # Get user's posts
    posts_cursor = db.posts.find({"author": user["user_id"]}).sort("created_at", -1).skip(skip).limit(posts_per_page)
    posts = await posts_cursor.to_list(None)
    
    # Total pages
    total_posts = await db.posts.count_documents({"author": user["user_id"]})
    total_pages = math.ceil(total_posts / posts_per_page)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Dashboard",
        "posts": posts,
        "current_page": page,
        "total_pages": total_pages,
        "logged_in": True,
        "user": user
    })


#####################
# Routes - Post CRUD
#####################
@app.get("/create", response_class=HTMLResponse)
async def create_post_page(request: Request):
    """Render new post creation page"""
    token = request.cookies.get("access_token")
    if not token or not verify_token(token):
        return RedirectResponse("/login", status_code=302)
    
    return templates.TemplateResponse("create.html", {
        "request": request,
        "title": "Create Post",
        "logged_in": True
    })


@app.post("/create")
async def create_post(
    request: Request,
    title: str = Form(...),
    content: str = Form(...),
    tags: str = Form(""),
    private: bool = Form(False)
):
    """Handle new post creation"""
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    
    payload = verify_token(token)
    if not payload:
        return RedirectResponse("/login", status_code=302)
    
    # Generate unique slug
    base_slug = generate_slug(title)
    slug = base_slug
    counter = 1
    while await db.posts.find_one({"slug": slug}):
        slug = f"{base_slug}-{counter}"
        counter += 1
    
    # Process tags
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    
    # Insert post in DB
    post_id = str(uuid.uuid4())
    post_doc = {
        "post_id": post_id,
        "title": title,
        "slug": slug,
        "author": payload["user_id"],
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "upvotes": 0,
        "downvotes": 0,
        "views": 0,
        "tags": tag_list,
        "private": private
    }
    await db.posts.insert_one(post_doc)
    
    # Save markdown to file
    post_file_path = POSTS_DIR / f"{post_id}.md"
    with open(post_file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return RedirectResponse(f"/post/{slug}", status_code=302)


@app.get("/edit/{post_id}", response_class=HTMLResponse)
async def edit_post_page(request: Request, post_id: str):
    """Render edit page for a specific post"""
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    
    payload = verify_token(token)
    if not payload:
        return RedirectResponse("/login", status_code=302)
    
    post = await db.posts.find_one({"post_id": post_id, "author": payload["user_id"]})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Load existing markdown
    post_file_path = POSTS_DIR / f"{post_id}.md"
    content = ""
    if post_file_path.exists():
        with open(post_file_path, "r", encoding="utf-8") as f:
            content = f.read()
    
    return templates.TemplateResponse("edit.html", {
        "request": request,
        "title": f"Edit: {post['title']}",
        "post": post,
        "content": content,
        "tags_str": ", ".join(post.get("tags", [])),
        "logged_in": True
    })


@app.post("/edit/{post_id}")
async def edit_post(
    request: Request,
    post_id: str,
    title: str = Form(...),
    content: str = Form(...),
    tags: str = Form(""),
    private: bool = Form(False)
):
    """Handle post edit submission"""
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    
    payload = verify_token(token)
    if not payload:
        return RedirectResponse("/login", status_code=302)
    
    post = await db.posts.find_one({"post_id": post_id, "author": payload["user_id"]})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Update tags
    tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    
    # Update post in DB
    update_data = {
        "title": title,
        "updated_at": datetime.utcnow(),
        "tags": tag_list,
        "private": private
    }
    await db.posts.update_one({"post_id": post_id}, {"$set": update_data})
    
    # Update markdown file
    post_file_path = POSTS_DIR / f"{post_id}.md"
    with open(post_file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return RedirectResponse(f"/post/{post['slug']}", status_code=302)


@app.post("/delete/{post_id}")
async def delete_post(request: Request, post_id: str):
    """Delete a post (DB + markdown file)"""
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    
    payload = verify_token(token)
    if not payload:
        return RedirectResponse("/login", status_code=302)
    
    post = await db.posts.find_one({"post_id": post_id, "author": payload["user_id"]})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Delete DB entry
    await db.posts.delete_one({"post_id": post_id})
    
    # Delete markdown file
    post_file_path = POSTS_DIR / f"{post_id}.md"
    if post_file_path.exists():
        post_file_path.unlink()
    
    return RedirectResponse("/dashboard", status_code=302)


#####################
# Routes - Post Viewing
#####################
@app.get("/post/{slug}", response_class=HTMLResponse)
async def view_post(request: Request, slug: str):
    """Render a post (public or private if owner)"""
    post = await db.posts.find_one({"slug": slug})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Check private visibility
    if post.get("private", False):
        token = request.cookies.get("access_token")
        if not token:
            raise HTTPException(status_code=404, detail="Post not found")
        payload = verify_token(token)
        if not payload or payload["user_id"] != post["author"]:
            raise HTTPException(status_code=404, detail="Post not found")
    
    # Track engagement
    await db.posts.update_one({"slug": slug}, {"$inc": {"views": 1}})
    await db.users.update_one({"user_id": post["author"]}, {"$inc": {"total_reader": 1}})
    
    # Load markdown and convert to HTML
    post_file_path = POSTS_DIR / f"{post['post_id']}.md"
    content = ""
    if post_file_path.exists():
        with open(post_file_path, "r", encoding="utf-8") as f:
            content = f.read()
    html_content = convert_markdown(content)
    
    # Fetch author details
    author = await db.users.find_one({"user_id": post["author"]})
    
    return templates.TemplateResponse("post.html", {
        "request": request,
        "title": post["title"],
        "post": post,
        "content": html_content,
        "author": author,
        "logged_in": get_logged_in_status(request)
    })


#####################
# Routes - Voting
#####################
@app.post("/vote/{post_id}")
async def vote_post(request: Request, post_id: str, vote_type: str = Form(...)):
    """Handle upvote/downvote on a post"""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    post = await db.posts.find_one({"post_id": post_id})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    if vote_type == "upvote":
        # Increment counters
        await db.posts.update_one({"post_id": post_id}, {"$inc": {"upvotes": 1}})
        await db.users.update_one({"user_id": post["author"]}, {"$inc": {"total_upvotes": 1}})
        
        # Track user liked tags
        user = await db.users.find_one({"user_id": payload["user_id"]})
        liked_tags = user.get("liked_tags", {})
        for tag in post.get("tags", []):
            liked_tags[tag] = liked_tags.get(tag, 0) + 1
        await db.users.update_one({"user_id": payload["user_id"]}, {"$set": {"liked_tags": liked_tags}})
    
    elif vote_type == "downvote":
        await db.posts.update_one({"post_id": post_id}, {"$inc": {"downvotes": 1}})
        await db.users.update_one({"user_id": post["author"]}, {"$inc": {"total_downvotes": 1}})
    
    return {"status": "success"}


#####################
# Main Entry Point
#####################
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
