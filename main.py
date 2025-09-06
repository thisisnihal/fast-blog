##########
# Imports
##########
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


#################
# Configuration
#################
# Secret key and JWT config
load_dotenv()
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-this")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# MongoDB config
MONGODB_URL = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DATABASE_NAME = "fastblog"

# Local folder for markdown posts
POSTS_DIR = Path("posts")
POSTS_DIR.mkdir(exist_ok=True)


#####################
# FastAPI App Setup
#####################
app = FastAPI(title="FastBlog", description="Multi-user blogging platform")

# Mount static files for CSS/JS/images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Jinja2 template setup
templates = Jinja2Templates(directory="templates")


##########
# Security
##########
# HTTP Bearer token dependency
security = HTTPBearer(auto_error=False)


######################
# Database Connection
######################
client = AsyncIOMotorClient(MONGODB_URL)
db = client[DATABASE_NAME]


#################
# Pydantic Models
#################
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

class UserTagPreference(BaseModel):
    tag: str
    weight: float
    user_chosen: bool = True


######################
# Utility Functions
######################

##########
# JWT Token
##########
def create_access_token(data: dict):
    """Create JWT token with expiration"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str):
    """Verify JWT token, return payload or None if invalid"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.PyJWTError:
        return None

##########
# Passwords
##########
def hash_password(password: str):
    """Hash password using bcrypt"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def verify_password(password: str, hashed: str):
    """Verify password against hashed value"""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

##########
# Slugs
##########
def generate_slug(title: str):
    """Generate URL-friendly slug from post title"""
    import re
    slug = re.sub(r'[^a-zA-Z0-9\s]', '', title).strip()
    slug = re.sub(r'\s+', '-', slug).lower()
    return slug

##########
# Post Scoring
##########
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

##########
# Current User
##########
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Return current user if token is valid, else None"""
    if not credentials:
        return None
    
    payload = verify_token(credentials.credentials)
    if not payload:
        return None
    
    user = await db.users.find_one({"user_id": payload.get("user_id")})
    return user

async def get_current_user_required(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Return current user or raise 401 if not authenticated"""
    user = await get_current_user(credentials)
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

def get_logged_in_status(request: Request):
    """Check if user is logged in based on cookie"""
    token = request.cookies.get("access_token")
    if token and verify_token(token):
        return True
    return False

##########
# User Tag Preferences
##########
async def update_user_tag_preferences(user_id: str, tags: List[str], engagement_weight: float = 1.0):
    """Update user's tag preferences based on engagement"""
    for tag in tags:
        existing_pref = await db.user_tag_preferences.find_one({
            "user_id": user_id,
            "tag": tag
        })
        
        if existing_pref:
            # Update weight (algorithm-based preferences get lower weight updates)
            if not existing_pref.get("user_chosen", False):
                new_weight = existing_pref.get("weight", 0) + (engagement_weight * 0.1)
                await db.user_tag_preferences.update_one(
                    {"user_id": user_id, "tag": tag},
                    {"$set": {"weight": min(new_weight, 10.0), "updated_at": datetime.utcnow()}}
                )
        else:
            # Create new algorithm-based preference
            await db.user_tag_preferences.insert_one({
                "user_id": user_id,
                "tag": tag,
                "weight": engagement_weight * 0.1,
                "user_chosen": False,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            })

async def get_user_preferred_tags(user_id: str, limit: int = 3):
    """Get user's top preferred tags"""
    preferences = await db.user_tag_preferences.find(
        {"user_id": user_id}
    ).sort("weight", -1).limit(limit).to_list(None)
    
    return [pref["tag"] for pref in preferences]


##############
# Startup Hook
##############
@app.on_event("startup")
async def startup_event():
    """Initialize DB indexes on startup"""
    # Unique constraints
    await db.users.create_index("username", unique=True)
    await db.users.create_index("email", unique=True)
    await db.posts.create_index("slug", unique=True)
    
    # Sorting & query optimization
    await db.posts.create_index([("created_at", DESCENDING)])
    await db.posts.create_index([("author", ASCENDING)])
    
    # Voting and tag preferences
    await db.post_votes.create_index([("user_id", ASCENDING), ("post_id", ASCENDING)], unique=True)
    await db.user_tag_preferences.create_index([("user_id", ASCENDING), ("tag", ASCENDING)], unique=True)
    await db.user_tag_preferences.create_index([("user_id", ASCENDING), ("weight", DESCENDING)])


##########
# Routes
##########

##################
# Home Page
##################
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    # Get public posts
    posts_cursor = db.posts.find({"private": False})
    posts = await posts_cursor.to_list(None)
    
    # Calculate scores and sort
    for post in posts:
        post['score'] = calculate_post_score(post)
    
    posts.sort(key=lambda x: x['score'], reverse=True)
    posts = posts[:20]  # Limit to top 20
    
    # Get user preferred tags if logged in
    user_tags = []
    token = request.cookies.get("access_token")
    if token:
        payload = verify_token(token)
        if payload:
            user_tags = await get_user_preferred_tags(payload.get("user_id"), 3)
    
    return templates.TemplateResponse("home.html", {
        "request": request,
        "title": "Fast Blog - Home",
        "posts": posts,
        "user_tags": user_tags,
        "logged_in": get_logged_in_status(request)
    })


##########
# Authentication
##########
@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    """Login page"""
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
    
    # Create JWT token and set cookie
    token = create_access_token({"user_id": user["user_id"]})
    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie("access_token", token, httponly=True)
    return response

@app.get("/signup", response_class=HTMLResponse)
async def signup_page(request: Request):
    """Sign-up page"""
    if get_logged_in_status(request):
        return RedirectResponse("/dashboard", status_code=302)
    
    return templates.TemplateResponse("signup.html", {
        "request": request,
        "title": "Sign Up",
        "logged_in": False
    })

@app.post("/signup")
async def signup(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    """Handle sign-up form submission"""
    # Check if username/email already exists
    existing_user = await db.users.find_one({"$or": [{"username": username}, {"email": email}]})
    if existing_user:
        error_msg = "Username already exists" if existing_user["username"] == username else "Email already exists"
        return templates.TemplateResponse("signup.html", {
            "request": request,
            "title": "Sign Up",
            "error": error_msg,
            "logged_in": False
        })
    
    # Create new user
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
        "created_at": datetime.utcnow()
    }
    
    await db.users.insert_one(user_doc)
    
    # Log in new user
    token = create_access_token({"user_id": user_id})
    response = RedirectResponse("/dashboard", status_code=302)
    response.set_cookie("access_token", token, httponly=True)
    return response

@app.get("/logout")
async def logout():
    """Log out user by deleting cookie"""
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("access_token")
    return response

#############
# Dashboard
#############
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, page: int = 1):
    """User dashboard showing their posts and tag preferences"""
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    
    # Verify token
    payload = verify_token(token)
    if not payload:
        return RedirectResponse("/login", status_code=302)
    
    # Get user from DB
    user = await db.users.find_one({"user_id": payload.get("user_id")})
    if not user:
        return RedirectResponse("/login", status_code=302)
    
    # Pagination setup
    posts_per_page = 10
    skip = (page - 1) * posts_per_page
    
    # Fetch user's posts
    posts_cursor = db.posts.find({"author": user["user_id"]}).sort("created_at", -1).skip(skip).limit(posts_per_page)
    posts = await posts_cursor.to_list(None)
    
    # Total pages for pagination
    total_posts = await db.posts.count_documents({"author": user["user_id"]})
    total_pages = math.ceil(total_posts / posts_per_page)
    
    # Fetch user's tag preferences
    user_tag_preferences = await db.user_tag_preferences.find(
        {"user_id": user["user_id"]}
    ).sort("weight", -1).to_list(None)
    
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "title": "Dashboard",
        "posts": posts,
        "current_page": page,
        "total_pages": total_pages,
        "logged_in": True,
        "user": user,
        "user_tag_preferences": user_tag_preferences
    })


####################
# Tag Management
####################
@app.get("/manage-tags", response_class=HTMLResponse)
async def manage_tags_page(request: Request):
    """Page for users to manage tag preferences"""
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    
    payload = verify_token(token)
    if not payload:
        return RedirectResponse("/login", status_code=302)
    
    user_id = payload.get("user_id")
    
    # Get current user's tag preferences
    user_tag_preferences = await db.user_tag_preferences.find(
        {"user_id": user_id}
    ).sort("weight", -1).to_list(None)
    
    # Aggregate most popular tags from posts
    all_tags_cursor = await db.posts.aggregate([
        {"$unwind": "$tags"},
        {"$group": {"_id": "$tags", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 50}
    ]).to_list(None)
    
    available_tags = [tag["_id"] for tag in all_tags_cursor]
    
    return templates.TemplateResponse("manage_tags.html", {
        "request": request,
        "title": "Manage Tag Preferences",
        "user_tag_preferences": user_tag_preferences,
        "available_tags": available_tags,
        "logged_in": True
    })

@app.post("/manage-tags")
async def update_tag_preferences(
    request: Request,
    selected_tags: str = Form(""),
    tag_weights: str = Form("")
):
    """Update user-chosen tag preferences"""
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    
    payload = verify_token(token)
    if not payload:
        return RedirectResponse("/login", status_code=302)
    
    user_id = payload.get("user_id")
    
    # Parse tags
    tags = [tag.strip() for tag in selected_tags.split(",") if tag.strip()]
    
    # Parse weights if provided
    weights = {}
    if tag_weights:
        weight_pairs = tag_weights.split(",")
        for pair in weight_pairs:
            if ":" in pair:
                tag, weight = pair.split(":", 1)
                try:
                    weights[tag.strip()] = float(weight.strip())
                except ValueError:
                    weights[tag.strip()] = 1.0
    
    # Remove previous user-chosen preferences
    await db.user_tag_preferences.delete_many({
        "user_id": user_id,
        "user_chosen": True
    })
    
    # Insert new preferences
    for tag in tags:
        weight = weights.get(tag, 5.0)  # Default weight
        await db.user_tag_preferences.insert_one({
            "user_id": user_id,
            "tag": tag,
            "weight": weight,
            "user_chosen": True,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
    
    return RedirectResponse("/dashboard", status_code=302)


##################
# Post Management
##################
@app.get("/create", response_class=HTMLResponse)
async def create_post_page(request: Request):
    """Page to create a new post"""
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
    """Handle new post submission"""
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
    
    # Insert post into DB
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
    
    # Save markdown content to file
    post_file_path = POSTS_DIR / f"{post_id}.md"
    with open(post_file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return RedirectResponse(f"/post/{slug}", status_code=302)

@app.get("/edit/{post_id}", response_class=HTMLResponse)
async def edit_post_page(request: Request, post_id: str):
    """Page to edit an existing post"""
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    
    payload = verify_token(token)
    if not payload:
        return RedirectResponse("/login", status_code=302)
    
    # Fetch post and verify ownership
    post = await db.posts.find_one({"post_id": post_id, "author": payload["user_id"]})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Load markdown content
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
    """Handle editing post submission"""
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    
    payload = verify_token(token)
    if not payload:
        return RedirectResponse("/login", status_code=302)
    
    post = await db.posts.find_one({"post_id": post_id, "author": payload["user_id"]})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Process tags
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
    """Delete a post and related data"""
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    
    payload = verify_token(token)
    if not payload:
        return RedirectResponse("/login", status_code=302)
    
    # Fetch post and verify ownership
    post = await db.posts.find_one({"post_id": post_id, "author": payload["user_id"]})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Delete post from DB
    await db.posts.delete_one({"post_id": post_id})
    
    # Delete votes related to post
    await db.post_votes.delete_many({"post_id": post_id})
    
    # Delete markdown file
    post_file_path = POSTS_DIR / f"{post_id}.md"
    if post_file_path.exists():
        post_file_path.unlink()
    
    return RedirectResponse("/dashboard", status_code=302)
#####################
# Post Viewing & Voting
#####################

@app.get("/post/{slug}", response_class=HTMLResponse)
async def view_post(request: Request, slug: str):
    """View a single post, handle private posts and track views/upvotes"""
    # Fetch post by slug
    post = await db.posts.find_one({"slug": slug})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Handle private posts: only author can view
    if post.get("private", False):
        token = request.cookies.get("access_token")
        if not token:
            raise HTTPException(status_code=404, detail="Post not found")
        
        payload = verify_token(token)
        if not payload or payload["user_id"] != post["author"]:
            raise HTTPException(status_code=404, detail="Post not found")
    
    # Increment post views
    await db.posts.update_one({"slug": slug}, {"$inc": {"views": 1}})
    
    # Update author's total reader count
    await db.users.update_one({"user_id": post["author"]}, {"$inc": {"total_reader": 1}})
    
    # Load markdown content
    post_file_path = POSTS_DIR / f"{post['post_id']}.md"
    content = ""
    if post_file_path.exists():
        with open(post_file_path, "r", encoding="utf-8") as f:
            content = f.read()
    
    # Convert markdown to HTML
    html_content = convert_markdown(content)
    
    # Get author info
    author = await db.users.find_one({"user_id": post["author"]})
    
    # Check if current user has voted on this post
    current_user_vote = None
    token = request.cookies.get("access_token")
    if token:
        payload = verify_token(token)
        if payload:
            vote_record = await db.post_votes.find_one({
                "user_id": payload["user_id"],
                "post_id": post["post_id"]
            })
            if vote_record:
                current_user_vote = vote_record["vote_type"]
    
    return templates.TemplateResponse("post.html", {
        "request": request,
        "title": post["title"],
        "post": post,
        "content": html_content,
        "author": author,
        "current_user_vote": current_user_vote,
        "logged_in": get_logged_in_status(request)
    })


@app.post("/vote/{post_id}")
async def vote_post(request: Request, post_id: str, vote_type: str = Form(...)):
    """Handle upvote/downvote logic for a post"""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    payload = verify_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    user_id = payload["user_id"]
    
    # Fetch post
    post = await db.posts.find_one({"post_id": post_id})
    if not post:
        raise HTTPException(status_code=404, detail="Post not found")
    
    # Check if user has already voted
    existing_vote = await db.post_votes.find_one({
        "user_id": user_id,
        "post_id": post_id
    })
    
    if existing_vote:
        old_vote_type = existing_vote["vote_type"]
        
        if old_vote_type == vote_type:
            # Remove vote if user votes the same way again
            await db.post_votes.delete_one({"user_id": user_id, "post_id": post_id})
            
            # Update post and author counts
            if vote_type == "upvote":
                await db.posts.update_one({"post_id": post_id}, {"$inc": {"upvotes": -1}})
                await db.users.update_one({"user_id": post["author"]}, {"$inc": {"total_upvotes": -1}})
            else:
                await db.posts.update_one({"post_id": post_id}, {"$inc": {"downvotes": -1}})
                await db.users.update_one({"user_id": post["author"]}, {"$inc": {"total_downvotes": -1}})
            
            return {"status": "vote_removed"}
        else:
            # Change vote
            await db.post_votes.update_one(
                {"user_id": user_id, "post_id": post_id},
                {"$set": {"vote_type": vote_type, "updated_at": datetime.utcnow()}}
            )
            
            # Update post and author counts (reverse old vote, apply new vote)
            if old_vote_type == "upvote":
                await db.posts.update_one({"post_id": post_id}, {"$inc": {"upvotes": -1, "downvotes": 1}})
                await db.users.update_one({"user_id": post["author"]}, {"$inc": {"total_upvotes": -1, "total_downvotes": 1}})
            else:
                await db.posts.update_one({"post_id": post_id}, {"$inc": {"upvotes": 1, "downvotes": -1}})
                await db.users.update_one({"user_id": post["author"]}, {"$inc": {"total_upvotes": 1, "total_downvotes": -1}})
            
            # Update tag preferences for upvotes
            if vote_type == "upvote":
                await update_user_tag_preferences(user_id, post.get("tags", []), 2.0)
            
            return {"status": "vote_changed"}
    else:
        # Insert new vote
        await db.post_votes.insert_one({
            "user_id": user_id,
            "post_id": post_id,
            "vote_type": vote_type,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        })
        
        # Update post and author counts
        if vote_type == "upvote":
            await db.posts.update_one({"post_id": post_id}, {"$inc": {"upvotes": 1}})
            await db.users.update_one({"user_id": post["author"]}, {"$inc": {"total_upvotes": 1}})
            
            # Update user's tag preferences based on engagement
            await update_user_tag_preferences(user_id, post.get("tags", []), 2.0)
        else:
            await db.posts.update_one({"post_id": post_id}, {"$inc": {"downvotes": 1}})
            await db.users.update_one({"user_id": post["author"]}, {"$inc": {"total_downvotes": 1}})
        
        return {"status": "vote_added"}


###############
# Entry Point
###############
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
