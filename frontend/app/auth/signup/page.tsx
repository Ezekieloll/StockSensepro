'use client';

import { useState } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import Button from '@/components/ui/Button';
import Input from '@/components/ui/Input';
import { MailIcon, LockIcon, UserIcon, BriefcaseIcon, TrendingUpIcon } from '@/components/ui/Icons';
import api from '@/lib/api';

export default function SignupPage() {
    const router = useRouter();
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        password: '',
        role: 'Analyst'
    });
    const [errors, setErrors] = useState<{ name?: string; email?: string; password?: string; general?: string }>({});
    const [loading, setLoading] = useState(false);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        setLoading(true);
        setErrors({});

        // Simple validation
        const newErrors: { name?: string; email?: string; password?: string } = {};
        if (!formData.name) newErrors.name = 'Name is required';
        if (!formData.email) newErrors.email = 'Email is required';
        if (!formData.password) newErrors.password = 'Password is required';
        if (formData.password && formData.password.length < 6) {
            newErrors.password = 'Password must be at least 6 characters';
        }

        if (Object.keys(newErrors).length > 0) {
            setErrors(newErrors);
            setLoading(false);
            return;
        }

        try {
            // Call backend API - signup returns user data directly
            const response = await api.signup({
                name: formData.name,
                email: formData.email,
                password: formData.password,
                role: formData.role
            });

            // Store user data from signup response
            localStorage.setItem('user', JSON.stringify({
                id: response.id,
                name: response.name,
                email: response.email,
                role: response.role.toLowerCase()
            }));

            // Redirect to dashboard
            router.push('/dashboard');
        } catch (error: any) {
            console.error('Signup error:', error);
            setErrors({
                general: error.message || 'An error occurred during signup. Please try again.'
            });
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center relative overflow-hidden p-6">
            {/* Animated background */}
            <div className="absolute inset-0">
                <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl animate-pulse"></div>
                <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-pink-500/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
            </div>

            {/* Signup Card */}
            <div className="relative z-10 w-full max-w-md animate-fadeIn">
                <div className="glass-strong rounded-2xl p-8 border border-white/10">
                    {/* Logo */}
                    <div className="flex items-center justify-center gap-2 mb-8">
                        <div className="w-12 h-12 bg-gradient-to-br from-purple-600 to-pink-600 rounded-xl flex items-center justify-center">
                            <TrendingUpIcon className="text-white" size={28} />
                        </div>
                        <span className="text-3xl font-bold gradient-text">StockSensePro</span>
                    </div>

                    <div className="text-center mb-8">
                        <h1 className="text-3xl font-bold mb-2">Create Account</h1>
                        <p className="text-muted">Start your journey to smarter investing</p>
                    </div>

                    {errors.general && (
                        <div className="mb-6 p-4 bg-error/10 border border-error/20 rounded-lg">
                            <p className="text-error text-sm">{errors.general}</p>
                        </div>
                    )}

                    <form onSubmit={handleSubmit} className="space-y-5">
                        <Input
                            type="text"
                            label="Full Name"
                            placeholder="John Doe"
                            icon={<UserIcon />}
                            value={formData.name}
                            onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                            error={errors.name}
                            disabled={loading}
                        />

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

                        <div className="w-full">
                            <label className="block text-sm font-medium text-foreground mb-2">
                                Role
                            </label>
                            <div className="relative">
                                <BriefcaseIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none z-10" />
                                <select
                                    className="input pl-10 cursor-pointer"
                                    value={formData.role}
                                    onChange={(e) => setFormData({ ...formData, role: e.target.value })}
                                    disabled={loading}
                                >
                                    <option value="Analyst">Analyst</option>
                                    <option value="Manager">Manager</option>
                                    <option value="Admin">Admin</option>
                                </select>
                            </div>
                        </div>

                        <div className="flex items-start gap-2">
                            <input type="checkbox" className="w-4 h-4 mt-1 rounded border-white/10 bg-surface" required />
                            <label className="text-sm text-muted">
                                I agree to the{' '}
                                <a href="#" className="text-primary hover:text-primary-dark transition-smooth">
                                    Terms of Service
                                </a>{' '}
                                and{' '}
                                <a href="#" className="text-primary hover:text-primary-dark transition-smooth">
                                    Privacy Policy
                                </a>
                            </label>
                        </div>

                        <Button type="submit" variant="primary" size="lg" fullWidth disabled={loading}>
                            {loading ? 'Creating Account...' : 'Create Account'}
                        </Button>
                    </form>

                    <div className="mt-6 text-center">
                        <p className="text-muted">
                            Already have an account?{' '}
                            <Link href="/auth/login" className="text-primary hover:text-primary-dark transition-smooth font-semibold">
                                Sign in
                            </Link>
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

