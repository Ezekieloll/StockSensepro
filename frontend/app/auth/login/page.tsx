'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Button from '@/components/Button';
import Input from '@/components/Input';
import { MailIcon, LockIcon, TrendingUpIcon } from '@/components/Icons';
import api from '@/lib/api';

export default function LoginPage() {
    const router = useRouter();
    const [formData, setFormData] = useState({
        email: '',
        password: ''
    });
    const [errors, setErrors] = useState<{ email?: string; password?: string; general?: string }>({});
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();

        // Simple validation
        const newErrors: { email?: string; password?: string } = {};
        if (!formData.email) newErrors.email = 'Email is required';
        if (!formData.password) newErrors.password = 'Password is required';

        if (Object.keys(newErrors).length > 0) {
            setErrors(newErrors);
            return;
        }

        setLoading(true);
        setErrors({});

        try {
            // Call backend API
            const response = await api.login({
                email: formData.email,
                password: formData.password
            });

            // Store user data (backend returns user info directly)
            localStorage.setItem('user', JSON.stringify({
                id: response.id,
                name: response.name,
                email: response.email,
                role: response.role.toLowerCase()
            }));

            // Redirect to dashboard
            router.push('/dashboard');
        } catch (error: any) {
            console.error('Login error:', error);
            setErrors({
                general: error.message || 'Invalid email or password. Please try again.'
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center relative overflow-hidden p-6">
            {/* Animated background */}
            <div className="absolute inset-0">
                <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-indigo-500/20 rounded-full blur-3xl animate-pulse"></div>
                <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
            </div>

            {/* Login Card */}
            <div className="relative z-10 w-full max-w-md animate-fadeIn">
                <div className="glass-strong rounded-2xl p-8 border border-white/10">
                    {/* Logo */}
                    <div className="flex items-center justify-center gap-2 mb-8">
                        <div className="w-12 h-12 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-xl flex items-center justify-center">
                            <TrendingUpIcon className="text-white" size={28} />
                        </div>
                        <span className="text-3xl font-bold gradient-text">StockSensePro</span>
                    </div>

                    <div className="text-center mb-8">
                        <h1 className="text-3xl font-bold mb-2">Welcome Back</h1>
                        <p className="text-muted">Sign in to access your dashboard</p>
                    </div>

                    {errors.general && (
                        <div className="mb-6 p-4 bg-error/10 border border-error/20 rounded-lg">
                            <p className="text-error text-sm">{errors.general}</p>
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-6">
                        <Input
                            type="email"
                            label="Email Address"
                            placeholder="you@example.com"
                            icon={<MailIcon />}
                            value={formData.email}
                            onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                            error={errors.email}
                            disabled={loading}
                        />

                        <Input
                            type="password"
                            label="Password"
                            placeholder="••••••••"
                            icon={<LockIcon />}
                            value={formData.password}
                            onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                            error={errors.password}
                            disabled={loading}
                        />

                        <div className="flex items-center justify-between text-sm">
                            <label className="flex items-center gap-2 cursor-pointer">
                                <input type="checkbox" className="w-4 h-4 rounded border-white/10 bg-surface" />
                                <span className="text-muted">Remember me</span>
                            </label>
                            <a href="#" className="text-primary hover:text-primary-dark transition-smooth">
                                Forgot password?
                            </a>
                        </div>

                        <Button type="submit" variant="primary" size="lg" fullWidth disabled={loading}>
                            {loading ? 'Signing in...' : 'Sign In'}
                        </Button>
                    </form>

                    <div className="mt-6 text-center">
                        <p className="text-muted">
                            Don't have an account?{' '}
                            <Link href="/auth/signup" className="text-primary hover:text-primary-dark transition-smooth font-semibold">
                                Sign up
                            </Link>
                        </p>
                    </div>

                    {/* Backend status hint */}
                    <div className="mt-8 p-4 bg-indigo-500/10 border border-indigo-500/20 rounded-lg">
                        <p className="text-sm text-muted text-center mb-2 font-medium">Backend Connection:</p>
                        <p className="text-xs text-muted text-center">
                            Ensure your FastAPI backend is running on <code className="bg-surface-elevated px-1 py-0.5 rounded">localhost:8000</code>.
                        </p>
                    </div>
                </div>

                <div className="text-center mt-6">
                    <Link href="/" className="text-muted hover:text-foreground transition-smooth">
                        ← Back to home
                    </Link>
                </div>
            </div>
        </div>
    );
}
