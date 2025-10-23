# EPFL Smart Kitchen Dataset - Node.js Website

> **Modern, secure, and production-ready website built with Node.js and Express.js**

[![Node.js](https://img.shields.io/badge/Node.js-18+-green.svg)](https://nodejs.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Deployment](https://img.shields.io/badge/deploy-ready-brightgreen.svg)]()

This is the Node.js version of the EPFL Smart Kitchen Dataset website, built with Express.js for easy deployment, better performance, and modern infrastructure.

## 📚 Documentation

- **[🚀 QUICKSTART.md](QUICKSTART.md)** - 5-minute quick start
- **[🏗️ ARCHITECTURE.md](ARCHITECTURE.md)** - System architecture & diagrams

## ✨ Features

- ✅ Express.js server with security middleware (Helmet)
- ✅ Gzip compression for optimized asset delivery
- ✅ Clean URL routing (no `.html` extensions)
- ✅ Automatic deployment via GitHub Actions
- ✅ No Docker support (this repository does not use Docker)
- ✅ Production-ready configuration
- ✅ Multiple deployment platform support
- ✅ Health checks and monitoring
- ✅ PM2 cluster mode ready

## 🚀 Quick Start (2 Commands!)

```bash
./setup.sh              # Automated setup
# or
npm install && npm run build && npm run dev
```

Visit: `http://localhost:3000`

**For detailed instructions, see [QUICKSTART.md](QUICKSTART.md)**

## Getting Started

### Prerequisites

- Node.js >= 18.0.0
- npm >= 9.0.0

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd website_node
```

2. Install dependencies:
```bash
npm install
```

3. Copy the environment file:
```bash
cp .env.example .env
```

4. Copy assets from the original website:
```bash
npm run build
```

### Development

Run the development server with auto-reload:
```bash
npm run dev
```

The website will be available at `http://localhost:3000`

### Production

Start the production server:
```bash
NODE_ENV=production npm start
```

## Project Structure

```
website_node/
├── public/              # Static files
│   ├── index.html      # Landing page
│   ├── pages/          # HTML pages
│   └── assets/         # CSS, images, videos
├── scripts/            # Build scripts
├── server.js           # Express server
├── package.json        # Dependencies
└── README.md          # Documentation
```

## Deployment

The website is automatically deployed via GitHub Actions when pushing to the `main` branch. See `.github/workflows/deploy.yml` for configuration.

### Manual Deployment

For platforms like Heroku, Render, or Railway:

1. Set environment variables
2. Deploy the repository
3. The platform will automatically run `npm start`

## Routes

- `/` - Landing page (redirects to dataset page)
- `/dataset` - Main dataset page
- Legacy routes are automatically redirected to new clean URLs

## License

See LICENSE file in the root directory.

## Contact

For questions or issues, please contact the EPFL AM Lab team.
