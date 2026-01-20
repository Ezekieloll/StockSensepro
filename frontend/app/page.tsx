'use client';

import Link from 'next/link';
import Button from '@/components/Button';
import { TrendingUpIcon, ChartIcon, ShieldIcon, ArrowRightIcon } from '@/components/Icons';

export default function Home() {
  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-indigo-500/20 rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
      </div>

      {/* Navigation */}
      <nav className="relative z-10 glass-strong border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-lg flex items-center justify-center">
                <TrendingUpIcon className="text-white" size={24} />
              </div>
              <span className="text-2xl font-bold gradient-text">StockSensePro</span>
            </div>
            <div className="flex items-center gap-4">
              <Link href="/auth/login">
                <Button variant="ghost">Login</Button>
              </Link>
              <Link href="/auth/signup">
                <Button variant="primary">Get Started</Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative z-10 max-w-7xl mx-auto px-6 py-24">
        <div className="text-center animate-fadeIn">
          <h1 className="text-6xl md:text-7xl font-bold mb-6">
            <span className="gradient-text">Smart Stock Analysis</span>
            <br />
            <span className="text-foreground">Made Simple</span>
          </h1>
          <p className="text-xl text-muted max-w-2xl mx-auto mb-12">
            Harness the power of AI-driven insights to make informed investment decisions.
            Real-time analytics, predictive modeling, and comprehensive market intelligence.
          </p>
          <div className="flex items-center justify-center gap-4">
            <Link href="/auth/signup">
              <Button variant="primary" size="lg">
                Start Free Trial
                <ArrowRightIcon size={20} />
              </Button>
            </Link>
            <Link href="/auth/login">
              <Button variant="secondary" size="lg">
                Sign In
              </Button>
            </Link>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mt-24 animate-slideInRight">
          <div className="glass-strong rounded-2xl p-8 border border-white/10 hover:border-indigo-500/50 transition-smooth hover:-translate-y-2">
            <div className="w-14 h-14 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center mb-6">
              <ChartIcon className="text-white" size={28} />
            </div>
            <h3 className="text-2xl font-bold mb-4">Advanced Analytics</h3>
            <p className="text-muted">
              Deep dive into market trends with our sophisticated analytical tools and real-time data visualization.
            </p>
          </div>

          <div className="glass-strong rounded-2xl p-8 border border-white/10 hover:border-purple-500/50 transition-smooth hover:-translate-y-2" style={{ animationDelay: '0.1s' }}>
            <div className="w-14 h-14 bg-gradient-to-br from-purple-600 to-pink-600 rounded-xl flex items-center justify-center mb-6">
              <TrendingUpIcon className="text-white" size={28} />
            </div>
            <h3 className="text-2xl font-bold mb-4">Predictive Insights</h3>
            <p className="text-muted">
              AI-powered predictions help you stay ahead of market movements and identify opportunities early.
            </p>
          </div>

          <div className="glass-strong rounded-2xl p-8 border border-white/10 hover:border-pink-500/50 transition-smooth hover:-translate-y-2" style={{ animationDelay: '0.2s' }}>
            <div className="w-14 h-14 bg-gradient-to-br from-pink-600 to-rose-600 rounded-xl flex items-center justify-center mb-6">
              <ShieldIcon className="text-white" size={28} />
            </div>
            <h3 className="text-2xl font-bold mb-4">Enterprise Security</h3>
            <p className="text-muted">
              Bank-grade encryption and security protocols ensure your data and investments are always protected.
            </p>
          </div>
        </div>

        {/* Stats Section */}
        <div className="grid md:grid-cols-4 gap-6 mt-24">
          <div className="text-center">
            <div className="text-5xl font-bold gradient-text mb-2">99.9%</div>
            <div className="text-muted">Uptime</div>
          </div>
          <div className="text-center">
            <div className="text-5xl font-bold gradient-text mb-2">50K+</div>
            <div className="text-muted">Active Users</div>
          </div>
          <div className="text-center">
            <div className="text-5xl font-bold gradient-text mb-2">$2B+</div>
            <div className="text-muted">Assets Tracked</div>
          </div>
          <div className="text-center">
            <div className="text-5xl font-bold gradient-text mb-2">24/7</div>
            <div className="text-muted">Support</div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/10 mt-24">
        <div className="max-w-7xl mx-auto px-6 py-8">
          <div className="text-center text-muted">
            <p>&copy; 2026 StockSensePro. All rights reserved.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
