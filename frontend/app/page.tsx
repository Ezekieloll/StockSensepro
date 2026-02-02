'use client';

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import Button from '@/components/ui/Button';
import { TrendingUpIcon, ChartIcon, ShieldIcon, ArrowRightIcon } from '@/components/ui/Icons';

export default function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [metrics, setMetrics] = useState({
    marketCap: 2847,
    activeStocks: 1523,
    predictions: 847,
    accuracy: 94.2
  });

  // Animated background with data particles and graphs
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Set canvas size
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
    };
    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Store canvas dimensions for use in classes
    const canvasWidth = canvas.width;
    const canvasHeight = canvas.height;

    // Particle system for floating data points
    class Particle {
      x: number;
      y: number;
      vx: number;
      vy: number;
      size: number;
      opacity: number;
      color: string;

      constructor() {
        this.x = Math.random() * canvasWidth;
        this.y = Math.random() * canvasHeight;
        this.vx = (Math.random() - 0.5) * 0.5;
        this.vy = (Math.random() - 0.5) * 0.5;
        this.size = Math.random() * 3 + 1;
        this.opacity = Math.random() * 0.5 + 0.2;
        const colors = ['#6366f1', '#8b5cf6', '#ec4899', '#3b82f6'];
        this.color = colors[Math.floor(Math.random() * colors.length)];
      }

      update() {
        this.x += this.vx;
        this.y += this.vy;

        if (this.x < 0 || this.x > canvasWidth) this.vx *= -1;
        if (this.y < 0 || this.y > canvasHeight) this.vy *= -1;
      }

      draw(ctx: CanvasRenderingContext2D) {
        ctx.beginPath();
        ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
        ctx.fillStyle = this.color + Math.floor(this.opacity * 255).toString(16).padStart(2, '0');
        ctx.fill();
      }
    }

    // Animated chart lines
    class ChartLine {
      points: { x: number; y: number }[];
      color: string;
      offset: number;
      speed: number;

      constructor(startY: number, color: string) {
        this.points = [];
        this.color = color;
        this.offset = 0;
        this.speed = 0.5;

        // Generate smooth wave points
        for (let i = 0; i < 100; i++) {
          this.points.push({
            x: (canvasWidth / 100) * i,
            y: startY + Math.sin(i * 0.1) * 30 + Math.random() * 20
          });
        }
      }

      update() {
        this.offset += this.speed;
        // Regenerate points for animation
        for (let i = 0; i < this.points.length; i++) {
          this.points[i].y += (Math.random() - 0.5) * 2;
        }
      }

      draw(ctx: CanvasRenderingContext2D) {
        ctx.beginPath();
        ctx.moveTo(this.points[0].x, this.points[0].y);

        for (let i = 1; i < this.points.length; i++) {
          ctx.lineTo(this.points[i].x, this.points[i].y);
        }

        ctx.strokeStyle = this.color + '40';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Add glow effect
        ctx.strokeStyle = this.color + '20';
        ctx.lineWidth = 4;
        ctx.stroke();
      }
    }

    // Candlestick chart elements
    class Candlestick {
      x: number;
      open: number;
      close: number;
      high: number;
      low: number;
      width: number;
      color: string;

      constructor(x: number, baseY: number) {
        this.x = x;
        this.width = 8;
        this.open = baseY + (Math.random() - 0.5) * 40;
        this.close = this.open + (Math.random() - 0.5) * 60;
        this.high = Math.min(this.open, this.close) - Math.random() * 20;
        this.low = Math.max(this.open, this.close) + Math.random() * 20;
        this.color = this.close > this.open ? '#10b981' : '#ef4444';
      }

      draw(ctx: CanvasRenderingContext2D) {
        // Draw wick
        ctx.beginPath();
        ctx.moveTo(this.x, this.high);
        ctx.lineTo(this.x, this.low);
        ctx.strokeStyle = this.color + '60';
        ctx.lineWidth = 1;
        ctx.stroke();

        // Draw body
        const bodyTop = Math.min(this.open, this.close);
        const bodyHeight = Math.abs(this.close - this.open);
        ctx.fillStyle = this.color + '80';
        ctx.fillRect(this.x - this.width / 2, bodyTop, this.width, bodyHeight);
      }
    }

    // Initialize elements
    const particles: Particle[] = [];
    for (let i = 0; i < 80; i++) {
      particles.push(new Particle());
    }

    const chartLines: ChartLine[] = [
      new ChartLine(canvasHeight * 0.3, '#6366f1'),
      new ChartLine(canvasHeight * 0.5, '#8b5cf6'),
      new ChartLine(canvasHeight * 0.7, '#ec4899')
    ];

    const candlesticks: Candlestick[] = [];
    for (let i = 0; i < 30; i++) {
      candlesticks.push(new Candlestick(canvasWidth * 0.8 + i * 15, canvasHeight * 0.4));
    }

    // Animation loop
    let animationFrame: number;
    const animate = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Draw and update particles
      particles.forEach(particle => {
        particle.update();
        particle.draw(ctx);
      });

      // Draw connections between nearby particles
      particles.forEach((p1, i) => {
        particles.slice(i + 1).forEach(p2 => {
          const dx = p1.x - p2.x;
          const dy = p1.y - p2.y;
          const distance = Math.sqrt(dx * dx + dy * dy);

          if (distance < 150) {
            ctx.beginPath();
            ctx.moveTo(p1.x, p1.y);
            ctx.lineTo(p2.x, p2.y);
            ctx.strokeStyle = `rgba(99, 102, 241, ${0.1 * (1 - distance / 150)})`;
            ctx.lineWidth = 1;
            ctx.stroke();
          }
        });
      });

      // Draw and update chart lines
      chartLines.forEach(line => {
        line.update();
        line.draw(ctx);
      });

      // Draw candlesticks
      candlesticks.forEach(candle => {
        candle.draw(ctx);
      });

      animationFrame = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      window.removeEventListener('resize', resizeCanvas);
      cancelAnimationFrame(animationFrame);
    };
  }, []);

  // Animated metrics counter
  useEffect(() => {
    const interval = setInterval(() => {
      setMetrics(prev => ({
        marketCap: prev.marketCap + (Math.random() - 0.5) * 10,
        activeStocks: prev.activeStocks + Math.floor((Math.random() - 0.5) * 5),
        predictions: prev.predictions + Math.floor((Math.random() - 0.5) * 3),
        accuracy: Math.min(99.9, Math.max(90, prev.accuracy + (Math.random() - 0.5) * 0.2))
      }));
    }, 2000);

    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Animated Canvas Background */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 z-0"
        style={{ opacity: 0.4 }}
      />

      {/* Gradient Overlays */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-0 left-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl animate-pulse-slow"></div>
        <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-purple-500/10 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '1s' }}></div>
        <div className="absolute top-1/2 left-1/2 w-96 h-96 bg-pink-500/10 rounded-full blur-3xl animate-pulse-slow" style={{ animationDelay: '2s' }}></div>
      </div>

      {/* Navigation */}
      <nav className="relative z-10 glass-strong border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-lg flex items-center justify-center animate-glow">
                <TrendingUpIcon className="text-white" size={24} />
              </div>
              <span className="text-2xl font-bold gradient-text">StockSense</span>
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
          <div className="inline-block mb-6 px-4 py-2 glass-strong rounded-full border border-indigo-500/30">
            <span className="text-sm text-indigo-400 font-semibold">🚀 AI-Powered Stock Analytics Platform</span>
          </div>

          <h1 className="text-6xl md:text-7xl lg:text-8xl font-bold mb-6 leading-tight">
            <span className="gradient-text">Real-Time Market</span>
            <br />
            <span className="text-foreground">Intelligence</span>
          </h1>

          <p className="text-xl md:text-2xl text-muted max-w-3xl mx-auto mb-8 leading-relaxed">
            Transform raw market data into actionable insights with our advanced AI algorithms.
            <br />
            <span className="text-indigo-400">Predict trends. Minimize risk. Maximize returns.</span>
          </p>

          {/* Live Metrics Dashboard */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 max-w-4xl mx-auto mb-12">
            <div className="glass-strong rounded-xl p-4 border border-indigo-500/20 hover:border-indigo-500/50 transition-smooth">
              <div className="text-3xl font-bold gradient-text mb-1">${metrics.marketCap.toFixed(0)}B</div>
              <div className="text-xs text-muted uppercase tracking-wide">Market Cap</div>
              <div className="text-xs text-green-400 mt-1">↑ Live</div>
            </div>
            <div className="glass-strong rounded-xl p-4 border border-purple-500/20 hover:border-purple-500/50 transition-smooth">
              <div className="text-3xl font-bold gradient-text mb-1">{metrics.activeStocks}</div>
              <div className="text-xs text-muted uppercase tracking-wide">Active Stocks</div>
              <div className="text-xs text-green-400 mt-1">↑ Tracking</div>
            </div>
            <div className="glass-strong rounded-xl p-4 border border-pink-500/20 hover:border-pink-500/50 transition-smooth">
              <div className="text-3xl font-bold gradient-text mb-1">{metrics.predictions}</div>
              <div className="text-xs text-muted uppercase tracking-wide">Predictions</div>
              <div className="text-xs text-blue-400 mt-1">⚡ Today</div>
            </div>
            <div className="glass-strong rounded-xl p-4 border border-blue-500/20 hover:border-blue-500/50 transition-smooth">
              <div className="text-3xl font-bold gradient-text mb-1">{metrics.accuracy.toFixed(1)}%</div>
              <div className="text-xs text-muted uppercase tracking-wide">Accuracy</div>
              <div className="text-xs text-green-400 mt-1">✓ Verified</div>
            </div>
          </div>

          <div className="flex items-center justify-center gap-4 flex-wrap">
            <Link href="/auth/signup">
              <Button variant="primary" size="lg">
                Start Free Trial
                <ArrowRightIcon size={20} />
              </Button>
            </Link>
            <Link href="/auth/login">
              <Button variant="secondary" size="lg">
                <ChartIcon size={20} />
                View Demo
              </Button>
            </Link>
          </div>

          <p className="text-sm text-muted mt-6">
            No credit card required • 14-day free trial • Cancel anytime
          </p>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mt-32 animate-slideInRight">
          <div className="glass-strong rounded-2xl p-8 border border-white/10 hover:border-indigo-500/50 transition-smooth hover:-translate-y-2 group">
            <div className="w-14 h-14 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-smooth">
              <ChartIcon className="text-white" size={28} />
            </div>
            <h3 className="text-2xl font-bold mb-4">Advanced Analytics</h3>
            <p className="text-muted leading-relaxed">
              Deep dive into market trends with sophisticated analytical tools, real-time data visualization, and predictive modeling powered by machine learning.
            </p>
            <div className="mt-6 flex items-center gap-2 text-indigo-400 text-sm font-semibold">
              Learn more <ArrowRightIcon size={16} />
            </div>
          </div>

          <div className="glass-strong rounded-2xl p-8 border border-white/10 hover:border-purple-500/50 transition-smooth hover:-translate-y-2 group" style={{ animationDelay: '0.1s' }}>
            <div className="w-14 h-14 bg-gradient-to-br from-purple-600 to-pink-600 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-smooth">
              <TrendingUpIcon className="text-white" size={28} />
            </div>
            <h3 className="text-2xl font-bold mb-4">Predictive Insights</h3>
            <p className="text-muted leading-relaxed">
              AI-powered predictions help you stay ahead of market movements, identify opportunities early, and make data-driven investment decisions with confidence.
            </p>
            <div className="mt-6 flex items-center gap-2 text-purple-400 text-sm font-semibold">
              Learn more <ArrowRightIcon size={16} />
            </div>
          </div>

          <div className="glass-strong rounded-2xl p-8 border border-white/10 hover:border-pink-500/50 transition-smooth hover:-translate-y-2 group" style={{ animationDelay: '0.2s' }}>
            <div className="w-14 h-14 bg-gradient-to-br from-pink-600 to-rose-600 rounded-xl flex items-center justify-center mb-6 group-hover:scale-110 transition-smooth">
              <ShieldIcon className="text-white" size={28} />
            </div>
            <h3 className="text-2xl font-bold mb-4">Enterprise Security</h3>
            <p className="text-muted leading-relaxed">
              Bank-grade encryption and security protocols ensure your data and investments are always protected with industry-leading compliance standards.
            </p>
            <div className="mt-6 flex items-center gap-2 text-pink-400 text-sm font-semibold">
              Learn more <ArrowRightIcon size={16} />
            </div>
          </div>
        </div>

        {/* Stats Section */}
        <div className="grid md:grid-cols-4 gap-6 mt-32">
          <div className="text-center group">
            <div className="text-5xl md:text-6xl font-bold gradient-text mb-2 group-hover:scale-110 transition-smooth">99.9%</div>
            <div className="text-muted font-semibold">System Uptime</div>
            <div className="text-xs text-green-400 mt-2">Industry Leading</div>
          </div>
          <div className="text-center group">
            <div className="text-5xl md:text-6xl font-bold gradient-text mb-2 group-hover:scale-110 transition-smooth">50K+</div>
            <div className="text-muted font-semibold">Active Traders</div>
            <div className="text-xs text-blue-400 mt-2">Growing Daily</div>
          </div>
          <div className="text-center group">
            <div className="text-5xl md:text-6xl font-bold gradient-text mb-2 group-hover:scale-110 transition-smooth">$2B+</div>
            <div className="text-muted font-semibold">Assets Tracked</div>
            <div className="text-xs text-purple-400 mt-2">Real-Time</div>
          </div>
          <div className="text-center group">
            <div className="text-5xl md:text-6xl font-bold gradient-text mb-2 group-hover:scale-110 transition-smooth">24/7</div>
            <div className="text-muted font-semibold">Expert Support</div>
            <div className="text-xs text-indigo-400 mt-2">Always Available</div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/10 mt-32">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <div className="text-center text-muted">
            <p className="mb-2">&copy; 2026 StockSense. All rights reserved.</p>
            <p className="text-sm">Empowering traders with intelligent market analytics.</p>
          </div>
        </div>
      </footer>
    </div>
  );
}
