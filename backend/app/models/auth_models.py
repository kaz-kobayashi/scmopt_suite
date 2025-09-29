"""
Authentication and authorization models for VRP API
"""
from pydantic import BaseModel, Field, EmailStr, validator
from typing import Dict, Any, Optional, List
from enum import Enum
from datetime import datetime, timedelta
import re

class UserRole(str, Enum):
    """User roles with different permissions"""
    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    VIEWER = "viewer"
    API_CLIENT = "api_client"

class PermissionLevel(str, Enum):
    """Permission levels for different operations"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"

class AuthProvider(str, Enum):
    """Authentication providers"""
    LOCAL = "local"
    OAUTH2 = "oauth2"
    LDAP = "ldap"
    SAML = "saml"
    API_KEY = "api_key"

class User(BaseModel):
    """User account information"""
    user_id: str = Field(description="Unique user identifier")
    username: str = Field(description="Username for login")
    email: EmailStr = Field(description="User email address")
    full_name: str = Field(description="Full name of user")
    role: UserRole = Field(description="User role")
    
    # Account status
    is_active: bool = Field(True, description="Whether account is active")
    is_verified: bool = Field(False, description="Whether email is verified")
    created_at: datetime = Field(default_factory=datetime.now, description="Account creation time")
    last_login: Optional[datetime] = Field(None, description="Last login timestamp")
    
    # Profile information
    organization: Optional[str] = Field(None, description="Organization name")
    department: Optional[str] = Field(None, description="Department")
    timezone: str = Field("UTC", description="User timezone")
    preferences: Dict[str, Any] = Field({}, description="User preferences")
    
    @validator('username')
    def validate_username(cls, v):
        if not re.match(r'^[a-zA-Z0-9_-]{3,50}$', v):
            raise ValueError('Username must be 3-50 characters, alphanumeric, underscore, or hyphen')
        return v

class UserCreate(BaseModel):
    """User creation request"""
    username: str = Field(description="Desired username")
    email: EmailStr = Field(description="Email address")
    password: str = Field(description="Password")
    full_name: str = Field(description="Full name")
    role: UserRole = Field(UserRole.VIEWER, description="User role")
    organization: Optional[str] = Field(None, description="Organization")
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdate(BaseModel):
    """User update request"""
    email: Optional[EmailStr] = Field(None, description="New email address")
    full_name: Optional[str] = Field(None, description="New full name")
    role: Optional[UserRole] = Field(None, description="New role")
    organization: Optional[str] = Field(None, description="New organization")
    is_active: Optional[bool] = Field(None, description="Account active status")
    preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences")

class LoginRequest(BaseModel):
    """Login request"""
    username: str = Field(description="Username or email")
    password: str = Field(description="Password")
    remember_me: bool = Field(False, description="Keep login session longer")

class TokenResponse(BaseModel):
    """Authentication token response"""
    access_token: str = Field(description="Access token")
    refresh_token: str = Field(description="Refresh token")
    token_type: str = Field("bearer", description="Token type")
    expires_in: int = Field(description="Token expiration time in seconds")
    scope: str = Field(description="Token scope/permissions")
    user_info: User = Field(description="User information")

class RefreshTokenRequest(BaseModel):
    """Token refresh request"""
    refresh_token: str = Field(description="Refresh token")

class APIKey(BaseModel):
    """API Key information"""
    key_id: str = Field(description="Unique key identifier")
    name: str = Field(description="Human-readable key name")
    key_prefix: str = Field(description="First 8 characters of key (for display)")
    permissions: List[str] = Field(description="List of permitted operations")
    rate_limit: int = Field(1000, description="Requests per hour limit")
    
    # Status and metadata
    is_active: bool = Field(True, description="Whether key is active")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    last_used: Optional[datetime] = Field(None, description="Last usage timestamp")
    expires_at: Optional[datetime] = Field(None, description="Expiration timestamp")
    
    # Ownership
    user_id: str = Field(description="Owner user ID")
    description: Optional[str] = Field(None, description="Key description/purpose")

class APIKeyCreate(BaseModel):
    """API Key creation request"""
    name: str = Field(description="Key name")
    permissions: List[str] = Field(description="Permitted operations")
    description: Optional[str] = Field(None, description="Key description")
    expires_in_days: Optional[int] = Field(None, description="Expiration in days")
    rate_limit: int = Field(1000, description="Requests per hour limit")

class APIKeyResponse(BaseModel):
    """API Key creation response"""
    key_info: APIKey = Field(description="Key information")
    secret_key: str = Field(description="Full secret key (only shown once)")
    usage_instructions: str = Field(description="How to use the API key")

class Permission(BaseModel):
    """Permission definition"""
    resource: str = Field(description="Resource identifier (e.g., 'vrp', 'reports')")
    action: str = Field(description="Action identifier (e.g., 'read', 'write', 'delete')")
    level: PermissionLevel = Field(description="Permission level")
    conditions: Optional[Dict[str, Any]] = Field(None, description="Additional conditions")

class RolePermissions(BaseModel):
    """Role-based permissions"""
    role: UserRole = Field(description="User role")
    permissions: List[Permission] = Field(description="List of permissions")
    description: str = Field(description="Role description")

class Session(BaseModel):
    """User session information"""
    session_id: str = Field(description="Unique session identifier")
    user_id: str = Field(description="User identifier")
    created_at: datetime = Field(description="Session creation time")
    expires_at: datetime = Field(description="Session expiration time")
    last_activity: datetime = Field(description="Last activity timestamp")
    ip_address: str = Field(description="Client IP address")
    user_agent: str = Field(description="Client user agent")
    is_active: bool = Field(True, description="Whether session is active")

class AuditLog(BaseModel):
    """Audit log entry"""
    log_id: str = Field(description="Unique log entry ID")
    user_id: Optional[str] = Field(None, description="User who performed action")
    api_key_id: Optional[str] = Field(None, description="API key used")
    action: str = Field(description="Action performed")
    resource: str = Field(description="Resource affected")
    resource_id: Optional[str] = Field(None, description="Specific resource ID")
    
    # Request details
    timestamp: datetime = Field(default_factory=datetime.now, description="When action occurred")
    ip_address: str = Field(description="Client IP address")
    user_agent: Optional[str] = Field(None, description="Client user agent")
    
    # Result information
    success: bool = Field(description="Whether action succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Dict[str, Any] = Field({}, description="Additional metadata")

class SecuritySettings(BaseModel):
    """System security settings"""
    password_min_length: int = Field(8, description="Minimum password length")
    password_require_uppercase: bool = Field(True, description="Require uppercase letters")
    password_require_lowercase: bool = Field(True, description="Require lowercase letters")
    password_require_numbers: bool = Field(True, description="Require numbers")
    password_require_symbols: bool = Field(False, description="Require symbols")
    
    # Session settings
    session_timeout_minutes: int = Field(480, description="Session timeout in minutes")
    max_failed_login_attempts: int = Field(5, description="Max failed login attempts")
    account_lockout_minutes: int = Field(30, description="Account lockout duration")
    
    # API security
    api_rate_limit_default: int = Field(1000, description="Default API rate limit per hour")
    require_api_key_for_access: bool = Field(True, description="Require API key for API access")
    
    # Multi-factor authentication
    mfa_enabled: bool = Field(False, description="Whether MFA is enabled")
    mfa_required_for_admin: bool = Field(True, description="Require MFA for admin users")

class OrganizationSettings(BaseModel):
    """Organization-level settings"""
    organization_id: str = Field(description="Organization identifier")
    name: str = Field(description="Organization name")
    
    # Branding
    logo_url: Optional[str] = Field(None, description="Organization logo URL")
    primary_color: str = Field("#1976d2", description="Primary brand color")
    secondary_color: str = Field("#dc004e", description="Secondary brand color")
    
    # Features
    enabled_features: List[str] = Field(description="List of enabled features")
    max_users: int = Field(100, description="Maximum number of users")
    max_api_keys: int = Field(50, description="Maximum number of API keys")
    
    # Settings
    default_timezone: str = Field("UTC", description="Default timezone")
    data_retention_days: int = Field(365, description="Data retention period in days")
    
    # Contact information
    admin_email: EmailStr = Field(description="Administrator email")
    support_email: Optional[EmailStr] = Field(None, description="Support contact email")