# SCMOPT2 - Supply Chain Management Optimization Suite

A comprehensive web application for supply chain management optimization with authentication and advanced analytics.

## 🚀 Features

- **Authentication System**: Email/password login with JWT tokens
- **Supply Chain Risk Management (SCRM)**: Expected Value, CVaR models
- **Vehicle Routing Problem (VRP)**: PyVRP integration with advanced algorithms
- **Inventory Optimization**: Multi-echelon inventory management
- **Logistics Network Design**: Facility location and network optimization
- **Job Shop Scheduling**: Production planning and scheduling
- **Revenue Management**: Dynamic pricing and demand forecasting
- **Real-time Analytics**: Live dashboards and monitoring

## 🏗️ Architecture

### Backend
- **FastAPI** - High-performance Python web framework
- **PostgreSQL** - Production database
- **SQLAlchemy** - ORM and database management
- **JWT Authentication** - Secure token-based auth
- **Celery + Redis** - Async task processing
- **Docker** - Containerized deployment

### Frontend
- **React + TypeScript** - Modern web UI
- **Material-UI** - Component library
- **Plotly** - Interactive visualizations
- **WebSocket** - Real-time updates

## 🚀 Quick Deployment

### Option 1: Railway (Recommended)
1. Fork this repository
2. Connect to [Railway](https://railway.app)
3. Deploy from GitHub repo
4. Add PostgreSQL and Redis databases
5. Set environment variables

### Option 2: Docker
```bash
# Development
docker-compose up --build

# Production
docker-compose -f docker-compose.prod.yml up --build -d
```

### Option 3: Heroku
```bash
heroku create your-app-name
heroku addons:create heroku-postgresql:essential-0
heroku addons:create heroku-redis:mini
git push heroku main
```

## 🔧 Environment Variables

```bash
SECRET_KEY=your-jwt-secret-key
DATABASE_URL=postgresql://user:pass@host:port/db
REDIS_URL=redis://host:port/0
PYTHONPATH=/app
```

## 📖 API Documentation

Once deployed, visit `/docs` for interactive API documentation.

### Authentication Endpoints
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/token` - OAuth2 token endpoint

### Core Features
- `/api/scrm/` - Supply Chain Risk Management
- `/api/routing/` - Vehicle Routing Optimization
- `/api/inventory/` - Inventory Management
- `/api/jobshop/` - Production Scheduling
- `/api/analytics/` - Business Analytics

## 🧪 Testing

```bash
# Backend tests
cd backend
python -m pytest

# Frontend tests
cd frontend
npm test
```

## 📊 Cost Estimation

| Platform | Monthly Cost | Free Tier |
|----------|-------------|-----------|
| Railway | $5 | 500 hours |
| Heroku | $13-18 | None |
| DigitalOcean | $5-10 | $200 credit |

## 📈 Performance

- **Optimization Algorithms**: Gurobi, PuLP, OR-Tools
- **Scalability**: Horizontal scaling with Celery workers
- **Caching**: Redis for session and computation caching
- **Real-time**: WebSocket connections for live updates

## 🔒 Security

- JWT token authentication
- Password hashing with bcrypt
- CORS configuration
- Environment variable protection
- SQL injection prevention

## 📚 Documentation

- [Deployment Guide](backend/DEPLOYMENT.md)
- [Railway Setup](backend/RAILWAY_DEPLOY.md)
- [API Reference](backend/docs/)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
- Create GitHub Issues for bugs
- Check documentation in `/docs`
- Review deployment guides

---

Built with ❤️ for supply chain optimization