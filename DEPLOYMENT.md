# 🚀 Break-even Lab Deployment Guide

This guide will walk you through deploying the Break-even Lab application to various platforms.

## 📋 Prerequisites

- GitHub account
- Python 3.9+ installed locally
- Git installed
- API keys (optional but recommended)

## 🎯 Deployment Options

### 1. Streamlit Community Cloud (Recommended)

**Pros:**
- Free hosting
- Easy deployment from GitHub
- Automatic updates
- Built-in analytics

**Steps:**

1. **Fork the Repository**
   ```bash
   # Go to https://github.com/yourusername/breakeven-lab
   # Click "Fork" button
   ```

2. **Connect to Streamlit Community Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account
   - Click "New app"

3. **Configure the App**
   - **Repository**: Select your forked repository
   - **Branch**: `main`
   - **Main file path**: `app/app.py`
   - **App URL**: Choose a unique name (e.g., `breakeven-lab`)

4. **Set Environment Variables**
   - In the Streamlit Community Cloud dashboard
   - Go to your app settings
   - Add the following environment variables:
     ```
     ALPHA_VANTAGE_API_KEY=your_key_here
     FINNHUB_API_KEY=your_key_here
     STRIPE_PUBLISHABLE_KEY=your_key_here
     STRIPE_SECRET_KEY=your_key_here
     ```

5. **Deploy**
   - Click "Deploy"
   - Wait for the deployment to complete
   - Your app will be available at `https://breakeven-lab.streamlit.app`

### 2. Heroku

**Pros:**
- Easy deployment
- Good for small to medium apps
- Built-in monitoring

**Steps:**

1. **Install Heroku CLI**
   ```bash
   # Download from https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **Login to Heroku**
   ```bash
   heroku login
   ```

3. **Create Heroku App**
   ```bash
   heroku create your-app-name
   ```

4. **Set Environment Variables**
   ```bash
   heroku config:set ALPHA_VANTAGE_API_KEY=your_key_here
   heroku config:set FINNHUB_API_KEY=your_key_here
   heroku config:set STRIPE_PUBLISHABLE_KEY=your_key_here
   heroku config:set STRIPE_SECRET_KEY=your_key_here
   ```

5. **Deploy**
   ```bash
   git push heroku main
   ```

### 3. Docker Deployment

**Pros:**
- Consistent environment
- Easy to scale
- Works on any platform

**Steps:**

1. **Build Docker Image**
   ```bash
   docker build -t breakeven-lab .
   ```

2. **Run Container**
   ```bash
   docker run -p 8501:8501 breakeven-lab
   ```

3. **Deploy to Cloud**
   - Use services like Google Cloud Run, AWS ECS, or Azure Container Instances
   - Push your image to a container registry
   - Deploy using the cloud provider's tools

### 4. VPS Deployment

**Pros:**
- Full control
- Cost-effective for high traffic
- Custom configurations

**Steps:**

1. **Set up VPS**
   - Choose a provider (DigitalOcean, Linode, AWS EC2)
   - Create Ubuntu 20.04+ instance
   - Configure firewall (open port 8501)

2. **Install Dependencies**
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip git nginx
   ```

3. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/breakeven-lab.git
   cd breakeven-lab
   ```

4. **Install Python Dependencies**
   ```bash
   pip3 install -r requirements.txt
   ```

5. **Set up Environment Variables**
   ```bash
   cp env.example .env
   nano .env  # Edit with your API keys
   ```

6. **Run Application**
   ```bash
   streamlit run app/app.py --server.port=8501 --server.address=0.0.0.0
   ```

7. **Set up Nginx (Optional)**
   ```bash
   sudo nano /etc/nginx/sites-available/breakeven-lab
   ```
   
   Add configuration:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

8. **Enable Site**
   ```bash
   sudo ln -s /etc/nginx/sites-available/breakeven-lab /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# API Keys (optional)
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key
FINNHUB_API_KEY=your_finnhub_key

# Stripe (for Pro features)
STRIPE_PUBLISHABLE_KEY=pk_test_your_key
STRIPE_SECRET_KEY=sk_test_your_key
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret

# Email (for alerts)
RESEND_API_KEY=re_your_resend_key

# App Configuration
SECRET_KEY=your_secret_key
ENVIRONMENT=production
```

### API Keys Setup

1. **Alpha Vantage**
   - Go to [alphavantage.co](https://www.alphavantage.co/support/#api-key)
   - Sign up for free account
   - Get your API key

2. **Finnhub**
   - Go to [finnhub.io](https://finnhub.io/register)
   - Sign up for free account
   - Get your API key

3. **Stripe**
   - Go to [stripe.com](https://stripe.com)
   - Create account
   - Get your publishable and secret keys

4. **Resend**
   - Go to [resend.com](https://resend.com)
   - Sign up for free account
   - Get your API key

## 📊 Monitoring

### Streamlit Community Cloud
- Built-in analytics dashboard
- View app usage and performance
- Monitor errors and logs

### Heroku
```bash
# View logs
heroku logs --tail

# Monitor app performance
heroku ps:scale web=1
```

### Custom Monitoring
- Set up monitoring with services like DataDog, New Relic, or Sentry
- Monitor application performance and errors
- Set up alerts for downtime

## 🔒 Security

### SSL/HTTPS
- Streamlit Community Cloud: Automatic HTTPS
- Heroku: Automatic HTTPS
- VPS: Use Let's Encrypt for free SSL certificates

### Environment Variables
- Never commit API keys to version control
- Use environment variables for sensitive data
- Rotate API keys regularly

### Rate Limiting
- Implement rate limiting for API endpoints
- Monitor for abuse and unusual traffic patterns

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install -r requirements.txt
   ```

2. **Port Already in Use**
   ```bash
   # Find process using port 8501
   lsof -i :8501
   # Kill process
   kill -9 PID
   ```

3. **Environment Variables Not Loading**
   - Check `.env` file exists
   - Verify variable names match exactly
   - Restart the application

4. **Database Issues**
   ```bash
   # Delete database file to reset
   rm users.db
   # Restart application
   ```

### Logs and Debugging

1. **Enable Debug Mode**
   ```bash
   export STREAMLIT_LOGGER_LEVEL=debug
   streamlit run app/app.py
   ```

2. **Check Application Logs**
   - Streamlit Community Cloud: Built-in logs
   - Heroku: `heroku logs --tail`
   - VPS: Check system logs

## 📈 Scaling

### Horizontal Scaling
- Use load balancers
- Deploy multiple instances
- Use container orchestration (Kubernetes, Docker Swarm)

### Vertical Scaling
- Increase server resources
- Optimize database queries
- Use caching (Redis, Memcached)

### Database Scaling
- Use managed database services
- Implement database sharding
- Use read replicas for heavy read workloads

## 🔄 Updates and Maintenance

### Automatic Updates
- Streamlit Community Cloud: Automatic from GitHub
- Heroku: Manual deployment required
- VPS: Set up CI/CD pipeline

### Manual Updates
```bash
# Pull latest changes
git pull origin main

# Install new dependencies
pip install -r requirements.txt

# Restart application
# Method depends on your deployment
```

### Backup Strategy
- Regular database backups
- Code repository backups
- Environment variable backups
- Monitor for data loss

## 📞 Support

### Getting Help
- Check the [README.md](README.md) for basic usage
- Open an issue on GitHub for bugs
- Check Streamlit documentation for framework issues

### Community
- Join the Streamlit community
- Participate in trading/finance forums
- Share your deployment experiences

---

**Happy Trading! 📊**

Remember to always test your deployment in a staging environment before going live with real users and data.
