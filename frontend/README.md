# StockSensePro Frontend

A modern, premium stock analysis dashboard built with Next.js 16, Tailwind CSS v4, and TypeScript.

## Features

### 🎨 Premium Design
- **Dark Theme**: Sleek dark mode with vibrant gradients
- **Glassmorphism**: Modern glass effects throughout the UI
- **Smooth Animations**: Micro-interactions and transitions
- **Responsive**: Mobile-first design that works on all devices

### 🔐 Authentication
The application uses a secure authentication system integrated with the FastAPI backend:
- **JWT (JSON Web Tokens)**: Used for secure user sessions.
- **Role-Based Access Control (RBAC)**: Supports Admin, Manager, and Analyst roles.
- **Secure Password Hashing**: Uses bcrypt on the backend.

### 🎨 Tailwind Play CDN
Styling is powered by the **Tailwind Play CDN** for instant turnaround and ease of use in local development. Custom theme configuration is injected directly in the root layout.


### 📊 Dashboard Features
- **Portfolio Overview**: Real-time portfolio value and performance
- **Stock Performance**: Top performing stocks with live data
- **Recent Activity**: Transaction history and trading activity
- **Role-Specific Content**:
  - **Admin**: User management and system controls
  - **Manager**: Team overview and performance metrics
  - **Analyst**: Standard portfolio analytics

## Tech Stack

- **Framework**: Next.js 16.1.1 (App Router)
- **Styling**: Tailwind CSS v4
- **Language**: TypeScript
- **Icons**: Custom SVG icons
- **Fonts**: Inter (Google Fonts)

## Getting Started

### Prerequisites
- Node.js 20+ installed
- npm or yarn package manager

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Run the development server:
```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser

## Demo Credentials

Use these email addresses to test different role-based dashboards:

- **Manager Dashboard**: `manager@example.com` (password: any)
- **Admin Dashboard**: `admin@example.com` (password: any)
- **Analyst Dashboard**: `analyst@example.com` (password: any)

## Project Structure

```
frontend/
├── app/
│   ├── auth/
│   │   ├── login/
│   │   │   └── page.tsx          # Login page
│   │   └── signup/
│   │       └── page.tsx          # Signup page
│   ├── dashboard/
│   │   └── page.tsx              # Role-based dashboard
│   ├── globals.css               # Global styles & design system
│   ├── layout.tsx                # Root layout
│   └── page.tsx                  # Landing page
├── components/
│   ├── Button.tsx                # Reusable button component
│   ├── Card.tsx                  # Card component system
│   ├── Icons.tsx                 # SVG icon library
│   └── Input.tsx                 # Form input component
└── public/                       # Static assets
```

## Components

### Button
Multi-variant button component with sizes and styles:
- Variants: `primary`, `secondary`, `outline`, `ghost`
- Sizes: `sm`, `md`, `lg`

### Card
Flexible card component with subcomponents:
- `Card`: Main container
- `CardHeader`: Header section
- `CardTitle`: Title text
- `CardDescription`: Description text
- `CardContent`: Content area

### Input
Form input with label, icon, and error states

### Icons
Comprehensive SVG icon library including:
- Mail, Lock, User icons
- Chart, Trending, Activity icons
- Settings, Logout, Bell icons
- And more...

## Design System

### Colors
- **Background**: `#0a0a0f`
- **Surface**: `#13131a`
- **Primary**: `#6366f1` (Indigo)
- **Secondary**: `#8b5cf6` (Purple)
- **Accent**: `#ec4899` (Pink)

### Gradients
- Primary: Indigo to Purple
- Secondary: Pink to Rose
- Success: Blue to Cyan
- Accent: Pink to Yellow

### Typography
- **Font Family**: Inter
- **Weights**: 300, 400, 500, 600, 700, 800

## Features Implemented

✅ Landing page with hero section and features
✅ Login/Signup authentication flow
✅ Role-based dashboard routing
✅ Portfolio statistics cards
✅ Stock performance tracking
✅ Recent activity feed
✅ Responsive navigation
✅ Glassmorphism effects
✅ Smooth animations
✅ Custom scrollbar
✅ Premium dark theme

## Build for Production

```bash
npm run build
npm start
```

## Notes

- The `@theme` warning in CSS is expected - it's a valid Tailwind CSS v4 directive
- Authentication is currently client-side only (localStorage)
- All data is dummy/mock data for demonstration
- Tailwind CSS v4 uses the new `@import "tailwindcss"` syntax

## Future Enhancements

- Backend API integration
- Real-time stock data
- Advanced charting with Chart.js or Recharts
- WebSocket for live updates
- User profile management
- Dark/light theme toggle
- Mobile app version

## License

MIT
