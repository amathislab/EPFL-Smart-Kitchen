# 🎉 Node.js Website Conversion - Quick Start

## What's Been Created

A complete Node.js version of your EPFL Smart Kitchen website in the `website_node/` folder with:

### ✅ Core Files
- **`server.js`** - Express.js server with security & compression
- **`package.json`** - Project configuration with all dependencies
- **`scripts/copy-assets.js`** - Automated asset copying from original website
- **`public/`** - Static files directory (created on build)
- **`.env.example`** - Environment variables template

### ✅ Deployment Configurations
- **`.github/workflows/deploy-node.yml`** - GitHub Actions workflow
 - (Docker entries removed)
- **`Procfile`** - Heroku deployment
- **`vercel.json`** - Vercel deployment
- **`render.yaml`** - Render.com deployment
- **`ecosystem.config.js`** - PM2 process manager config

### ✅ Documentation
- **`README.md`** - Project overview & getting started
- **`.gitignore`** - Git ignore rules

---

## 🚀 Quick Start (3 Steps)

### 1. Install Dependencies
```bash
cd website_node
npm install
```

### 2. Copy Assets from Original Website
```bash
npm run build
```

### 3. Start Development Server
```bash
npm run dev
```

Visit: `http://localhost:3000`

---

## 📦 What's Different from the Original?

### Improvements:
1. **Clean URLs**: `/dataset` instead of `/pages/EPFLSmartKitchenResearchDataset.html`
2. **Security**: Helmet.js for HTTP security headers
3. **Performance**: Gzip compression, static file caching
4. **SEO**: Better URL structure, 301 redirects
5. **Deployment**: One-click deploy to multiple platforms
6. **Monitoring**: Health checks, error handling
7. **Scalability**: Cluster mode support via PM2

### Structure:
```
website_node/
├── server.js              # Express server
├── package.json           # Dependencies
├── public/                # Static files (after build)
│   ├── index.html
│   ├── pages/
│   └── assets/
├── scripts/               # Build scripts
├── .github/workflows/     # CI/CD
└── (no Docker files)     # Container config removed
```

---

## 🌐 Deployment Options

### Easiest (Render.com - Recommended for free tier)
1. Push code to GitHub
2. Connect Render.com to your repo
3. Select `website_node` folder
4. Deploy automatically

### Fastest (Vercel)
```bash
npm i -g vercel
cd website_node
vercel
```

### Most Popular (Heroku)
```bash
cd website_node
heroku create
git push heroku main
```

### Full Control

Use your chosen platform (Render, Vercel, Heroku) — see `DEPLOYMENT.md` for instructions.

See `DEPLOYMENT.md` for detailed guides on all platforms.

---

## 🔄 GitHub Workflow

The workflow automatically:
1. Runs on push to `main` branch
2. Installs dependencies
3. Copies assets from original website
4. Creates deployment package
5. Uploads artifact

**To enable automatic deployment**, uncomment the relevant section in:
`.github/workflows/deploy-node.yml`

---

## 🛠️ Development Commands

```bash
# Install dependencies
npm install

# Copy assets from original website
npm run build

# Start development server (with auto-reload)
npm run dev

# Start production server
npm start

# Run with your chosen platform (see DEPLOYMENT.md)

# Deploy to Vercel
vercel --prod
```

---

## 🔧 Configuration

### Environment Variables
Copy `.env.example` to `.env`:
```bash
PORT=3000
NODE_ENV=development
```

### Routes
- `/` → Redirect to dataset page
- `/dataset` → Main dataset page
- `/pages/EPFLSmartKitchenResearchDataset.html` → 301 redirect to `/dataset`
- `/assets/*` → Static files

---

## 📊 Platform Comparison

| Platform | Setup Time | Free Tier | Always On | Best For |
|----------|-----------|-----------|-----------|----------|
| **Render** | 2 min | ✅ Yes | ✅ Yes | **Recommended** |
| Vercel | 1 min | ✅ Yes | ✅ Yes | Fast deploys |
| Heroku | 3 min | ✅ Yes | ❌ Sleeps | Quick testing |
| Railway | 2 min | ✅ Yes | ✅ Yes | Modern UI |
| Docker | 5 min | N/A | N/A | Any platform |

---

## 🎯 Next Steps

1. **Test locally**:
   ```bash
   cd website_node
   npm install
   npm run build
   npm run dev
   ```

2. **Choose deployment platform** (see `DEPLOYMENT.md`)

3. **Configure GitHub workflow** (optional)

4. **Set up custom domain** (on your hosting platform)

5. **Monitor and optimize**

---

## 🔐 Security Features

- Helmet.js security headers
- HTTPS redirect support
- Content Security Policy
- CORS configuration
- Rate limiting ready
- Input validation

---

## 📈 Performance Features

- Gzip compression
- Static file caching
- CDN-ready
- Cluster mode support
- Health checks
- Error logging

---

## 🐛 Troubleshooting

**Assets not showing?**
```bash
npm run build
```

**Port already in use?**
```bash
# Edit .env to use different port
PORT=3001
```

**Node version issues?**
```bash
# Check Node version (needs >= 18)
node --version
```

---

## 📞 Support

For detailed deployment instructions, see `DEPLOYMENT.md`

For questions or issues:
- Check the README.md
- Review DEPLOYMENT.md
- Contact EPFL AM Lab team

---

## ✨ Summary

You now have a production-ready Node.js website with:
- ✅ Modern Express.js server
- ✅ Security best practices
- ✅ Multiple deployment options
- ✅ GitHub Actions workflow
- Docker support removed
- ✅ Comprehensive documentation

**Happy deploying! 🚀**
