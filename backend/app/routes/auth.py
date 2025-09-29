from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from app.database import get_db
from app.models.auth import UserCreate, User, Token, LoginRequest, RegisterRequest
from app.services.auth_service import AuthService, ACCESS_TOKEN_EXPIRE_MINUTES, get_current_user

router = APIRouter(
    prefix="/auth",
    tags=["authentication"]
)

@router.post("/register", response_model=User)
async def register(
    user_data: RegisterRequest,
    db: Session = Depends(get_db)
):
    """Register a new user"""
    try:
        # Check if user already exists
        existing_user = AuthService.get_user_by_email(db, user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        
        # Create new user
        user = AuthService.create_user(
            db=db,
            email=user_data.email,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
        return user
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Registration failed for {user_data.email}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}"
        )

@router.post("/login", response_model=Token)
async def login(
    user_data: LoginRequest,
    db: Session = Depends(get_db)
):
    """Login user and return access token"""
    user = AuthService.authenticate_user(db, user_data.email, user_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = AuthService.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """OAuth2 compatible token endpoint"""
    user = AuthService.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = AuthService.create_access_token(
        data={"sub": user.email}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@router.get("/me", response_model=User)
async def get_current_user_info(
    current_user = Depends(get_current_user)
):
    """Get current authenticated user information"""
    return current_user

@router.get("/debug/users")
async def debug_list_users(db: Session = Depends(get_db)):
    """Debug endpoint to list all registered users (temporary)"""
    from app.database import User
    users = db.query(User).all()
    return {
        "total_users": len(users),
        "users": [{"email": user.email, "full_name": user.full_name, "created_at": user.created_at} for user in users]
    }

@router.get("/debug/database")
async def debug_database_info():
    """Debug endpoint to check database configuration"""
    import os
    from app.database import DATABASE_URL, engine
    
    return {
        "database_url": DATABASE_URL,
        "database_type": "postgresql" if "postgresql" in DATABASE_URL else "sqlite" if "sqlite" in DATABASE_URL else "unknown",
        "engine_url": str(engine.url),
        "environment_variables": {
            "DATABASE_URL": os.getenv("DATABASE_URL", "NOT SET"),
            "POSTGRES_URL": os.getenv("POSTGRES_URL", "NOT SET"),
            "PGHOST": os.getenv("PGHOST", "NOT SET"),
            "PGPORT": os.getenv("PGPORT", "NOT SET"),
            "PGUSER": os.getenv("PGUSER", "NOT SET"),
            "PGPASSWORD": os.getenv("PGPASSWORD", "NOT SET") if os.getenv("PGPASSWORD") else "NOT SET",
            "PGDATABASE": os.getenv("PGDATABASE", "NOT SET")
        }
    }

@router.get("/debug/test-db")
async def test_database_connection(db: Session = Depends(get_db)):
    """Test database connection and operations"""
    try:
        from app.database import User
        # Test basic query
        user_count = db.query(User).count()
        
        # Test inserting a dummy user
        test_email = "test@example.com"
        existing_test = db.query(User).filter(User.email == test_email).first()
        
        if not existing_test:
            from app.services.auth_service import AuthService
            test_user = User(
                email=test_email,
                full_name="Test User",
                hashed_password=AuthService.get_password_hash("test123")
            )
            db.add(test_user)
            db.commit()
            db.refresh(test_user)
            created_user_id = test_user.id
        else:
            created_user_id = existing_test.id
        
        return {
            "status": "success",
            "user_count": user_count,
            "test_user_created": created_user_id,
            "database_working": True
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "database_working": False
        }