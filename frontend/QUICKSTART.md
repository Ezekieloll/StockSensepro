# StockSensePro - Quick Start Guide

## ✅ Application Successfully Created!

Your Next.js frontend is now running at: **http://localhost:3000**

## 🎯 What's Been Built

### Pages Created:
1. **Landing Page** (`/`) - Premium hero section with features and stats
2. **Login Page** (`/auth/login`) - Authentication with role-based routing
3. **Signup Page** (`/auth/signup`) - User registration with role selection
4. **Dashboard** (`/dashboard`) - Role-based dashboard with analytics

### Components Created:
- `Button.tsx` - Multi-variant button component
- `Input.tsx` - Form input with icons and validation
- `Card.tsx` - Card system with Header, Title, Description, Content
- `Icons.tsx` - Comprehensive SVG icon library

## 🚀 Quick Start Steps

1.  **Start the Backend**:
    *   Make sure your FastAPI backend is running on `http://localhost:8000`.
    *   Ensure the database is initialized.

2.  **Start the Frontend**:
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

3.  **Access the App**:
    *   Open [http://localhost:3000](http://localhost:3000)

## 🔑 Demo Credentials

| Role | Email | Password |
| :--- | :--- | :--- |
| **Admin** | admin@example.com | admin123 |
| **Manager** | manager@example.com | manager123 |
| **Analyst** | analyst@example.com | analyst123 |
*(Ensure these users are created in your backend database or use the Signup page)*


## 🎨 Design Features

✨ **Premium Dark Theme** with vibrant gradients
✨ **Glassmorphism Effects** throughout the UI
✨ **Smooth Animations** and micro-interactions
✨ **Responsive Design** - works on all devices
✨ **Custom Icons** - SVG icon library
✨ **Inter Font** from Google Fonts

## 🚀 How to Use

1. **View Landing Page**: Navigate to http://localhost:3000
2. **Sign Up**: Click "Get Started" or "Sign Up"
   - Fill in name, email, password
   - Select role (Analyst, Manager, or Admin)
   - Submit to create account
3. **Login**: Use demo credentials or your created account
4. **Dashboard**: View role-specific dashboard with:
   - Portfolio statistics
   - Top performing stocks
   - Recent activity
   - Role-specific controls

## 📁 Project Structure

```
frontend/
├── app/
│   ├── auth/login/page.tsx       # Login page
│   ├── auth/signup/page.tsx      # Signup page
│   ├── dashboard/page.tsx        # Dashboard (role-based)
│   ├── page.tsx                  # Landing page
│   ├── layout.tsx                # Root layout
│   └── globals.css               # Design system
├── components/
│   ├── Button.tsx
│   ├── Card.tsx
│   ├── Icons.tsx
│   └── Input.tsx
└── package.json
```

## 🛠️ Tech Stack

- **Next.js 16.1.1** - React framework with App Router
- **Tailwind CSS v4** - Utility-first CSS framework
- **TypeScript** - Type-safe development
- **React 19** - Latest React features

## 📝 Notes

- Authentication is client-side using localStorage
- All data is dummy/mock data for demonstration
- The `@theme` CSS warning is expected (Tailwind v4 feature)
- Responsive design works on mobile, tablet, and desktop

## 🎯 Next Steps

You can now:
1. Browse the landing page
2. Test the authentication flow
3. Explore different role-based dashboards
4. Customize the design and components
5. Integrate with your backend API

## 🐛 Troubleshooting

**If the page doesn't load:**
- Check that the dev server is running
- Refresh the browser
- Clear browser cache

**If you see CSS errors:**
- Make sure all imports are in the correct order
- Check that Tailwind CSS v4 is installed

Enjoy your premium StockSensePro dashboard! 🚀
