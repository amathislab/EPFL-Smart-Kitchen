# 🏗️ Architecture & Structure

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    EPFL Smart Kitchen Website                │
│                      Node.js Architecture                     │
└─────────────────────────────────────────────────────────────┘

┌────────────────┐      ┌──────────────────┐      ┌────────────────┐
│   Web Browser  │─────▶│  Express.js      │─────▶│  Static Files  │
│                │◀─────│  Server          │◀─────│  (public/)     │
└────────────────┘      └──────────────────┘      └────────────────┘
                              │
                              ├─ Helmet.js (Security)
                              ├─ Compression (Gzip)
                              ├─ Router (Clean URLs)
                              └─ Error Handler
```

---

## Request Flow

```
User Request → Load Balancer (Optional) → Express Server
                                              │
                                              ├─ Security Headers (Helmet)
                                              │
                                              ├─ Compression Check
                                              │
                                              ├─ Route Matching
                                              │   ├─ / → index.html
                                              │   ├─ /dataset → dataset page
                                              │   ├─ /assets/* → static files
                                              │   └─ * → 404
                                              │
                                              ├─ Response
                                              │
                                              └─ Logging
```

---

## Directory Structure

```
website_node/
│
├─ 📄 Core Application
│  ├─ server.js                 Express server entry point
│  ├─ package.json              Dependencies & scripts
│  └─ .env.example              Environment config template
│
├─ 🌐 Public Files (Generated)
│  └─ public/
│     ├─ index.html             Landing page
│     ├─ pages/                 HTML pages
│     │  └─ EPFLSmartKitchenResearchDataset.html
│     └─ assets/                Static assets
│        ├─ css/                Stylesheets
│        │  ├─ index.css
│        │  └─ main.css
│        └─ images/             Images & videos
│           └─ [30+ files]
│
├─ 🔨 Build Scripts
│  └─ scripts/
│     └─ copy-assets.js         Copy from ../website/
│
├─ 📚 Documentation
│  ├─ README.md                 Project overview
│  ├─ SUMMARY.md                Complete overview
│  ├─ QUICKSTART.md             Quick start guide
│  ├─ DEPLOYMENT.md             Deployment guides
│  ├─ MIGRATION.md              Migration guide
│  ├─ INDEX.md                  Documentation index
│  └─ ARCHITECTURE.md           This file
│
├─ 🚀 Deployment Configs
│  ├─ Procfile                  Heroku config
│  ├─ vercel.json               Vercel config
│  ├─ render.yaml               Render.com config
│  └─ ecosystem.config.js       PM2 process manager
│
├─ ⚙️ Automation
│  ├─ setup.sh                  Automated setup
│  └─ deploy.sh                 Deployment helper
│
└─ 🔧 Configuration
   ├─ .gitignore                Git ignore rules
   └─ .env                      Environment vars (not in git)
```

---

## Component Architecture

### 1. Express Server (server.js)

```javascript
┌─────────────────────────────────────┐
│         Express Application          │
├─────────────────────────────────────┤
│                                      │
│  ┌────────────────────────────┐     │
│  │   Security Middleware      │     │
│  │   - Helmet.js              │     │
│  │   - CSP Headers            │     │
│  │   - XSS Protection         │     │
│  └────────────────────────────┘     │
│                ↓                     │
│  ┌────────────────────────────┐     │
│  │   Performance Middleware   │     │
│  │   - Compression (Gzip)     │     │
│  │   - Static Caching         │     │
│  └────────────────────────────┘     │
│                ↓                     │
│  ┌────────────────────────────┐     │
│  │   Router                   │     │
│  │   - / → index.html         │     │
│  │   - /dataset → dataset     │     │
│  │   - /assets/* → static     │     │
│  │   - 301 redirects          │     │
│  └────────────────────────────┘     │
│                ↓                     │
│  ┌────────────────────────────┐     │
│  │   Error Handler            │     │
│  │   - 404 Page               │     │
│  │   - 500 Errors             │     │
│  │   - Logging                │     │
│  └────────────────────────────┘     │
│                                      │
└─────────────────────────────────────┘
```

### 2. Static File Serving

```
┌──────────────────────┐
│  Static Middleware   │
├──────────────────────┤
│                      │
│  Route: /assets/*    │
│      ↓               │
│  Check Cache         │
│      ↓               │
│  Serve File          │
│      ↓               │
│  Set Headers:        │
│  - Cache-Control     │
│  - ETag              │
│  - Content-Type      │
│                      │
└──────────────────────┘
```

### 3. Build Process

```
┌────────────────┐
│  npm run build │
└───────┬────────┘
        │
        ↓
┌────────────────────────┐
│  scripts/copy-assets   │
├────────────────────────┤
│                        │
│  1. Create public/     │
│  2. Copy index.html    │
│  3. Copy pages/        │
│  4. Copy assets/       │
│                        │
│  Source: ../website/   │
│  Target: ./public/     │
│                        │
└────────────────────────┘
```

---

## Deployment Architectures

### Option 1: Heroku

```
┌─────────────┐
│  Git Push   │
└──────┬──────┘
       │
       ↓
┌──────────────────┐
│  Heroku Platform │
├──────────────────┤
│  1. Build        │
│  2. Install deps │
│  3. Run Procfile │
│  4. Start server │
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  Your App        │
│  (with SSL)      │
└──────────────────┘
```

### Option 2: Docker

```
┌──────────────┐
│  Dockerfile  │
└──────┬───────┘
       │
       ↓
┌──────────────────┐
│  Build Stage     │
├──────────────────┤
│  - npm ci        │
│  - Dependencies  │
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  Production Stage│
├──────────────────┤
│  - Copy files    │
│  - Non-root user │
│  - Health check  │
└──────┬───────────┘
       │
       ↓
┌──────────────────┐
│  Container       │
│  Port: 3000      │
└──────────────────┘
```

### Option 3: Render.com

```
┌────────────────┐
│  GitHub Push   │
└───────┬────────┘
        │
        ↓
┌────────────────────┐
│  Render Platform   │
├────────────────────┤
│  1. Detect Node.js │
│  2. npm install    │
│  3. npm run build  │
│  4. npm start      │
└────────┬───────────┘
         │
         ↓
┌────────────────────┐
│  Live Website      │
│  (Always-on, SSL)  │
└────────────────────┘
```

### Option 4: Vercel

```
┌────────────────┐
│  Git Push      │
└───────┬────────┘
        │
        ↓
┌────────────────────┐
│  Vercel Edge       │
├────────────────────┤
│  1. Build          │
│  2. Optimize       │
│  3. Deploy to CDN  │
└────────┬───────────┘
         │
         ↓
┌────────────────────┐
│  Global CDN        │
│  (Ultra-fast)      │
└────────────────────┘
```

---

## CI/CD Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    GitHub Actions Workflow                   │
└─────────────────────────────────────────────────────────────┘

                           Push to main
                                 │
                                 ↓
                      ┌──────────────────┐
                      │  Trigger Workflow │
                      └─────────┬─────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                    ↓                       ↓
         ┌──────────────────┐    ┌──────────────────┐
         │  Test Workflow   │    │ Deploy Workflow  │
         ├──────────────────┤    ├──────────────────┤
         │ 1. Checkout      │    │ 1. Checkout      │
         │ 2. Setup Node    │    │ 2. Setup Node    │
         │ 3. Install deps  │    │ 3. Install deps  │
         │ 4. Copy assets   │    │ 4. Copy assets   │
         │ 5. Test server   │    │ 5. Package       │
         └──────────────────┘    │ 6. Deploy        │
                                 └────────┬─────────┘
                                          │
                                          ↓
                                 ┌────────────────┐
                                 │  Live Website  │
                                 └────────────────┘
```

---

## Data Flow Diagram

```
┌──────────┐         ┌─────────────┐         ┌──────────┐
│  Client  │◀───────▶│   Express   │◀───────▶│  Files   │
│ (Browser)│         │   Server    │         │ (public/)│
└──────────┘         └─────────────┘         └──────────┘
     │                      │
     │                      │
     ↓                      ↓
┌──────────┐         ┌─────────────┐
│  Cache   │         │   Logs      │
│ (Browser)│         │ (Server)    │
└──────────┘         └─────────────┘
```

---

## Security Layers

```
┌─────────────────────────────────────────┐
│           Security Architecture          │
└─────────────────────────────────────────┘

Layer 1: Network
├─ HTTPS/TLS
├─ SSL Certificate
└─ Secure Headers

Layer 2: Application
├─ Helmet.js
│  ├─ CSP (Content Security Policy)
│  ├─ HSTS (HTTP Strict Transport Security)
│  ├─ X-Frame-Options
│  ├─ X-Content-Type-Options
│  └─ XSS Protection
│
└─ Express.js
   ├─ Path traversal protection
   ├─ Query parameter sanitization
   └─ Error handling (no stack traces)

Layer 3: Infrastructure
├─ Non-root Docker user
├─ Read-only file system (except logs)
└─ Resource limits
```

---

## Performance Architecture

```
┌────────────────────────────────────────┐
│        Performance Optimizations        │
└────────────────────────────────────────┘

1. Compression
   ├─ Gzip enabled
   ├─ Brotli ready
   └─ ~70% size reduction

2. Caching
   ├─ Static files: 1 day
   ├─ Browser cache
   └─ CDN cache (platform-dependent)

3. Static Files
   ├─ Separate route
   ├─ Efficient serving
   └─ ETag support

4. Process Management (PM2)
   ├─ Cluster mode
   ├─ Auto-restart
   └─ Load balancing
```

---

## Monitoring & Logging

```
┌────────────────────────────────────┐
│      Monitoring Architecture        │
└────────────────────────────────────┘

Application Logs
├─ Console output
├─ Error tracking
└─ Request logging (optional)

Health Checks
├─ HTTP endpoint
├─ Docker healthcheck
└─ Platform monitoring

Performance Metrics
├─ Response time
├─ Memory usage
└─ CPU usage

Error Handling
├─ 404 errors
├─ 500 errors
└─ Stack traces (dev only)
```

---

## Scaling Options

### Vertical Scaling (Single Instance)
```
┌──────────────┐
│   PM2        │
│   Cluster    │
├──────────────┤
│ Worker 1     │  ← CPU Core 1
│ Worker 2     │  ← CPU Core 2
│ Worker 3     │  ← CPU Core 3
│ Worker 4     │  ← CPU Core 4
└──────────────┘
```

### Horizontal Scaling (Multiple Instances)
```
┌─────────────────┐
│ Load Balancer   │
└────────┬────────┘
         │
    ┌────┴────┬────────┬────────┐
    │         │        │        │
    ↓         ↓        ↓        ↓
┌────────┐┌────────┐┌────────┐┌────────┐
│ Node 1 ││ Node 2 ││ Node 3 ││ Node 4 │
└────────┘└────────┘└────────┘└────────┘
```

---

## Environment Configurations

```
┌──────────────────────────────────────┐
│        Environment Architecture       │
└──────────────────────────────────────┘

Development
├─ nodemon (auto-reload)
├─ Source maps
├─ Verbose logging
└─ No compression

Production
├─ PM2 cluster mode
├─ Compression enabled
├─ Minimal logging
└─ Error tracking
```

---

## File Processing Pipeline

```
Original Website          Build Process         Node.js Website
(website/)               (npm run build)       (website_node/public/)

index.html        ─────▶  Copy  ─────▶        index.html
pages/            ─────▶  Copy  ─────▶        pages/
assets/css/       ─────▶  Copy  ─────▶        assets/css/
assets/images/    ─────▶  Copy  ─────▶        assets/images/
                                              
                         ┌─────────────┐
                         │   Serve     │
                         │  via        │
                         │  Express.js │
                         └─────────────┘
```

---

## Technology Stack

```
┌─────────────────────────────────────┐
│         Technology Stack             │
├─────────────────────────────────────┤
│                                      │
│  Runtime:     Node.js >= 18         │
│  Framework:   Express.js 4.x        │
│  Security:    Helmet.js 7.x         │
│  Compression: compression 1.x       │
│  Process:     PM2 (optional)        │
│  Container:   Docker                │
│  CI/CD:       GitHub Actions        │
│                                      │
└─────────────────────────────────────┘

Frontend (Unchanged)
├─ HTML5
├─ CSS3
└─ Vanilla JavaScript
```

---

## Summary

This architecture provides:

✅ **Security**: Multi-layer protection  
✅ **Performance**: Compression & caching  
✅ **Scalability**: Horizontal & vertical  
✅ **Reliability**: Error handling & monitoring  
✅ **Flexibility**: Multiple deployment options  
✅ **Maintainability**: Clear structure & documentation  

For more details, see individual documentation files.
