# SCMOPT2 Deployment Guide

## Option 1: Heroku Deployment

### Prerequisites
- Install [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)
- Create a Heroku account

### Steps

1. **Login to Heroku**
   ```bash
   heroku login
   ```

2. **Create Heroku App**
   ```bash
   heroku create your-app-name
   ```

3. **Add Buildpack**
   ```bash
   heroku buildpacks:set heroku/python
   ```

4. **Add Add-ons**
   ```bash
   heroku addons:create heroku-postgresql:essential-0
   heroku addons:create heroku-redis:mini
   ```

5. **Set Environment Variables**
   ```bash
   heroku config:set SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')
   heroku config:set PYTHONPATH=/app
   ```

6. **Deploy**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

7. **Scale Workers**
   ```bash
   heroku ps:scale web=1 worker=1
   ```

## Option 2: Docker Deployment

### Development
```bash
docker-compose up --build
```

### Production
```bash
docker-compose -f docker-compose.prod.yml up --build -d
```

## Authentication Endpoints

Once deployed, the following endpoints are available:

- `POST /api/auth/register` - Register new user
- `POST /api/auth/login` - Login and get JWT token  
- `POST /api/auth/token` - OAuth2 compatible token endpoint

### Example Registration
```bash
curl -X POST "https://your-app.herokuapp.com/api/auth/register" \
     -H "Content-Type: application/json" \
     -d '{
       "email": "user@example.com",
       "password": "secure_password",
       "full_name": "Test User"
     }'
```

### Example Login
```bash
curl -X POST "https://your-app.herokuapp.com/api/auth/login" \
     -H "Content-Type: application/json" \
     -d '{
       "email": "user@example.com", 
       "password": "secure_password"
     }'
```

## Frontend Integration

The frontend needs to be updated to include login/register forms and handle JWT tokens for authenticated requests.

## Environment Variables

- `DATABASE_URL` - PostgreSQL connection string (auto-set by Heroku)
- `REDIS_URL` - Redis connection string (auto-set by Heroku)
- `SECRET_KEY` - JWT secret key (set manually)
- `PYTHONPATH` - Python path (set to /app)