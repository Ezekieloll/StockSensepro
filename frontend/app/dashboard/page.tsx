'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Card, { CardHeader, CardTitle, CardDescription, CardContent } from '@/components/Card';
import Button from '@/components/Button';
import Badge from '@/components/Badge';
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from '@/components/Table';
import {
    TrendingUpIcon,
    TrendingDownIcon,
    DollarIcon,
    ActivityIcon,
    ChartIcon,
    UserIcon,
    SettingsIcon,
    LogoutIcon,
    BellIcon,
    SearchIcon,
    ShieldIcon,
    BriefcaseIcon
} from '@/components/Icons';

interface User {
    name?: string;
    email: string;
    role: string;
}

export default function DashboardPage() {
    const router = useRouter();
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Check if user is logged in
        const userData = localStorage.getItem('user');
        if (!userData) {
            router.push('/auth/login');
            return;
        }
        const parsed = JSON.parse(userData);
        // Redirect admins to admin dashboard
        if (parsed.role === 'admin') {
            router.push('/admin');
            return;
        }
        // Redirect managers to manager dashboard
        if (parsed.role === 'manager') {
            router.push('/manager');
            return;
        }
        // Redirect analysts to analyst dashboard
        if (parsed.role === 'analyst') {
            router.push('/analyst');
            return;
        }
        setUser(parsed);
        setLoading(false);
    }, [router]);

    const handleLogout = () => {
        localStorage.removeItem('user');
        localStorage.removeItem('token');
        router.push('/');
    };

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-background">
                <div className="animate-pulse text-2xl font-bold gradient-text">Loading StockSensePro...</div>
            </div>
        );
    }

    if (!user) return null;

    const role = user.role?.toLowerCase();

    // Dummy data
    const portfolioValue = 1234567.89;
    const dailyChange = 12345.67;
    const dailyChangePercent = 1.23;

    const topStocks = [
        { symbol: 'AAPL', name: 'Apple Inc.', price: 178.45, change: 2.34, changePercent: 1.33 },
        { symbol: 'MSFT', name: 'Microsoft Corp.', price: 378.91, change: -1.23, changePercent: -0.32 },
        { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 142.67, change: 3.45, changePercent: 2.48 },
        { symbol: 'AMZN', name: 'Amazon.com Inc.', price: 156.23, change: 1.89, changePercent: 1.22 },
    ];

    const recentActivity = [
        { action: 'Buy', stock: 'AAPL', shares: 50, price: 178.45, time: '2 hours ago', status: 'Completed' },
        { action: 'Sell', stock: 'TSLA', shares: 25, price: 245.67, time: '5 hours ago', status: 'Completed' },
        { action: 'Buy', stock: 'NVDA', shares: 30, price: 489.12, time: '1 day ago', status: 'Pending' },
    ];

    return (
        <div className="min-h-screen bg-background text-foreground selection:bg-primary/30">
            {/* Navigation */}
            <nav className="glass border-b border-white/10 sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-8">
                            <div className="flex items-center gap-2">
                                <div className="w-10 h-10 bg-gradient-primary rounded-lg flex items-center justify-center shadow-lg shadow-primary/20">
                                    <TrendingUpIcon className="text-white" size={24} />
                                </div>
                                <span className="text-xl font-bold gradient-text">StockSensePro</span>
                            </div>
                            <div className="hidden md:flex items-center gap-6">
                                <a href="#" className="text-foreground hover:text-primary transition-colors font-medium">Dashboard</a>
                                <a href="#" className="text-muted hover:text-foreground transition-colors">Portfolio</a>
                                <a href="#" className="text-muted hover:text-foreground transition-colors">Market</a>
                                <a href="#" className="text-muted hover:text-foreground transition-colors">News</a>
                            </div>
                        </div>
                        <div className="flex items-center gap-4">
                            <div className="relative hidden sm:block">
                                <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" size={16} />
                                <input
                                    type="text"
                                    placeholder="Quick Search..."
                                    className="bg-surface-elevated border border-white/10 rounded-full py-1.5 pl-9 pr-4 text-xs focus:ring-1 focus:ring-primary outline-none transition-all w-48 focus:w-64"
                                />
                            </div>
                            <button className="p-2 hover:bg-white/5 rounded-lg transition-colors relative">
                                <BellIcon className="text-muted" size={20} />
                                <span className="absolute top-2 right-2 w-1.5 h-1.5 bg-error rounded-full outline outline-2 outline-surface"></span>
                            </button>
                            <div className="h-8 w-px bg-white/10 mx-1"></div>
                            <button onClick={handleLogout} className="flex items-center gap-2 group">
                                <div className="w-8 h-8 rounded-full bg-gradient-to-r from-indigo-500 to-purple-500 p-[1px]">
                                    <div className="w-full h-full rounded-full bg-surface-elevated flex items-center justify-center text-[10px] font-bold text-white">
                                        {user.name ? user.name[0] : user.email[0].toUpperCase()}
                                    </div>
                                </div>
                                <LogoutIcon className="text-muted group-hover:text-error transition-colors" size={18} />
                            </button>
                        </div>
                    </div>
                </div>
            </nav>

            <div className="max-w-7xl mx-auto px-6 py-8">
                {/* Header */}
                <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight mb-2">
                            Welcome back, <span className="gradient-text">{user.name || user.email.split('@')[0]}</span>
                        </h1>
                        <p className="text-muted flex items-center gap-2 text-sm">
                            {role === 'manager' && <BriefcaseIcon size={14} className="text-secondary" />}
                            {role === 'admin' && <ShieldIcon size={14} className="text-accent" />}
                            {role === 'analyst' && <ChartIcon size={14} className="text-primary" />}
                            <span className="capitalize tracking-wide">{role} Control Center</span>
                            <span className="w-1 h-1 bg-white/20 rounded-full"></span>
                            <span>Jan 13, 2026</span>
                        </p>
                    </div>
                    <div className="flex items-center gap-3">
                        <Button variant="secondary" size="sm">Download Report</Button>
                        <Button variant="primary" size="sm">New Trade</Button>
                    </div>
                </div>

                {/* Stats Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    <Card glass className="overflow-hidden group">
                        <div className="absolute top-0 right-0 w-24 h-24 bg-primary/10 rounded-bl-full -mr-12 -mt-12 transition-all group-hover:bg-primary/20"></div>
                        <div className="relative">
                            <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-1">Total Net Worth</p>
                            <h3 className="text-2xl font-bold tracking-tight">${portfolioValue.toLocaleString()}</h3>
                            <div className="flex items-center gap-1.5 mt-2">
                                <div className="flex items-center text-success bg-success/10 px-1.5 py-0.5 rounded text-[10px] font-bold">
                                    <TrendingUpIcon size={10} className="mr-0.5" />
                                    {dailyChangePercent}%
                                </div>
                                <span className="text-xs text-muted">since yesterday</span>
                            </div>
                        </div>
                    </Card>

                    <Card glass className="group">
                        <div className="flex items-start justify-between">
                            <div>
                                <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-1">Active Positions</p>
                                <h3 className="text-2xl font-bold tracking-tight">24 <span className="text-sm font-normal text-muted">assets</span></h3>
                                <p className="text-[10px] text-muted mt-2 tracking-wide">Diversified in 8 sectors</p>
                            </div>
                            <div className="w-10 h-10 bg-secondary/10 rounded-lg flex items-center justify-center text-secondary">
                                <BriefcaseIcon size={20} />
                            </div>
                        </div>
                    </Card>

                    <Card glass className="group">
                        <div className="flex items-start justify-between">
                            <div>
                                <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-1">Weekly Volume</p>
                                <h3 className="text-2xl font-bold tracking-tight">$84.2K</h3>
                                <div className="flex items-center gap-1 mt-2">
                                    <span className="w-1.5 h-1.5 rounded-full bg-success"></span>
                                    <span className="text-[10px] text-muted tracking-wide">Above average</span>
                                </div>
                            </div>
                            <div className="w-10 h-10 bg-accent/10 rounded-lg flex items-center justify-center text-accent">
                                <ActivityIcon size={20} />
                            </div>
                        </div>
                    </Card>

                    <Card glass className="group">
                        <div className="flex items-start justify-between">
                            <div>
                                <p className="text-xs font-semibold text-muted uppercase tracking-wider mb-1">Efficiency Rate</p>
                                <h3 className="text-2xl font-bold tracking-tight">92.4%</h3>
                                <div className="flex items-center gap-1 mt-2">
                                    <span className="text-[10px] text-success font-bold">+4.1%</span>
                                    <span className="text-[10px] text-muted tracking-wide">this month</span>
                                </div>
                            </div>
                            <div className="w-10 h-10 bg-info/10 rounded-lg flex items-center justify-center text-info">
                                <ShieldIcon size={20} />
                            </div>
                        </div>
                    </Card>
                </div>

                {/* Role-specific content */}
                {role === 'admin' && (
                    <div className="mb-8 animate-fadeIn">
                        <Card glass className="border-accent/20">
                            <CardHeader className="flex flex-row items-center justify-between pb-2">
                                <div>
                                    <CardTitle className="text-lg">System Administration</CardTitle>
                                    <CardDescription>Real-time security monitoring and user access management</CardDescription>
                                </div>
                                <Badge variant="info">Active System</Badge>
                            </CardHeader>
                            <CardContent className="pt-4">
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    <Button variant="secondary" className="flex items-center justify-start h-12">
                                        <UserIcon size={18} className="mr-3 text-primary" />
                                        User Directory
                                    </Button>
                                    <Button variant="secondary" className="flex items-center justify-start h-12">
                                        <ShieldIcon size={18} className="mr-3 text-accent" />
                                        Security Logs
                                    </Button>
                                    <Button variant="secondary" className="flex items-center justify-start h-12">
                                        <SettingsIcon size={18} className="mr-3 text-info" />
                                        Global Settings
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                )}

                {role === 'manager' && (
                    <div className="mb-8 animate-fadeIn">
                        <Card glass className="border-secondary/20">
                            <CardHeader className="flex flex-row items-center justify-between pb-2">
                                <div>
                                    <CardTitle className="text-lg">Portfolio Management</CardTitle>
                                    <CardDescription>Overview of managed accounts and team allocations</CardDescription>
                                </div>
                                <div className="flex gap-2">
                                    <Badge variant="success">Market Open</Badge>
                                </div>
                            </CardHeader>
                            <CardContent className="pt-4">
                                <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
                                    <div className="p-4 bg-white/5 rounded-xl border border-white/5 hover:border-secondary/30 transition-colors">
                                        <div className="text-2xl font-bold tracking-tight">$42.4M</div>
                                        <div className="text-[10px] text-muted uppercase tracking-widest mt-1">Total Assets Under Mgmt</div>
                                    </div>
                                    <div className="p-4 bg-white/5 rounded-xl border border-white/5">
                                        <div className="text-2xl font-bold tracking-tight">12</div>
                                        <div className="text-[10px] text-muted uppercase tracking-widest mt-1">Active Portfolio Managers</div>
                                    </div>
                                    <div className="p-4 bg-white/5 rounded-xl border border-white/5">
                                        <div className="text-2xl font-bold tracking-tight">642</div>
                                        <div className="text-[10px] text-muted uppercase tracking-widest mt-1">Total Client Base</div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Top Stocks */}
                    <Card glass className="flex flex-col">
                        <CardHeader>
                            <CardTitle className="text-lg">Market Performers</CardTitle>
                            <CardDescription>High-impact stocks in your watchlist</CardDescription>
                        </CardHeader>
                        <CardContent className="flex-1">
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>Asset</TableHead>
                                        <TableHead>Price</TableHead>
                                        <TableHead className="text-right">Change</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {topStocks.map((stock) => (
                                        <TableRow key={stock.symbol}>
                                            <TableCell>
                                                <div className="flex items-center gap-3">
                                                    <div className="w-8 h-8 rounded-lg bg-surface-elevated flex items-center justify-center font-bold text-xs ring-1 ring-white/10">
                                                        {stock.symbol[0]}
                                                    </div>
                                                    <div>
                                                        <div className="font-semibold text-sm">{stock.symbol}</div>
                                                        <div className="text-[10px] text-muted">{stock.name}</div>
                                                    </div>
                                                </div>
                                            </TableCell>
                                            <TableCell className="font-medium text-sm">${stock.price}</TableCell>
                                            <TableCell className={`text-right text-sm font-semibold ${stock.change >= 0 ? 'text-success' : 'text-error'}`}>
                                                {stock.change >= 0 ? '+' : ''}{stock.changePercent}%
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </CardContent>
                    </Card>

                    {/* Recent Activity */}
                    <Card glass className="flex flex-col">
                        <CardHeader>
                            <CardTitle className="text-lg">Trade History</CardTitle>
                            <CardDescription>Your most recent ledger entries</CardDescription>
                        </CardHeader>
                        <CardContent className="flex-1">
                            <div className="space-y-4">
                                {recentActivity.map((activity, index) => (
                                    <div key={index} className="flex items-center justify-between p-4 bg-white/5 rounded-xl border border-white/5 hover:bg-white/[0.08] transition-colors group">
                                        <div className="flex items-center gap-4">
                                            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${activity.action === 'Buy'
                                                ? 'bg-success/10 text-success'
                                                : 'bg-error/10 text-error'
                                                }`}>
                                                {activity.action === 'Buy' ? <TrendingUpIcon size={18} /> : <TrendingDownIcon size={18} />}
                                            </div>
                                            <div>
                                                <div className="font-bold text-sm group-hover:text-primary transition-colors">{activity.action} {activity.stock}</div>
                                                <div className="text-xs text-muted mt-0.5">{activity.shares} shares @ ${activity.price}</div>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <Badge variant={activity.status === 'Completed' ? 'success' : 'warning'} className="mb-1">
                                                {activity.status}
                                            </Badge>
                                            <div className="text-[10px] text-muted uppercase tracking-tighter">{activity.time}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                            <Button variant="ghost" className="w-full mt-6 text-xs text-muted hover:text-foreground">
                                View All Activity
                            </Button>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
}
