'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Card, { CardHeader, CardTitle, CardDescription, CardContent } from '@/components/Card';
import Button from '@/components/Button';
import Badge from '@/components/Badge';
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from '@/components/Table';
import {
    TrendingUpIcon,
    ChartIcon,
    UserIcon,
    SettingsIcon,
    LogoutIcon,
    BellIcon,
    ShieldIcon,
    UploadIcon,
    DatabaseIcon,
    ActivityIcon,
    AlertIcon,
    CheckIcon,
    ClockIcon,
    RefreshIcon,
} from '@/components/Icons';

interface User {
    id?: number;
    name?: string;
    email: string;
    role: string;
}

// Mock data for demo
const mockStagingQueue = [
    { id: 1, filename: 'transactions_jan.csv', rows: 45230, status: 'validating', uploadedAt: '10 min ago' },
    { id: 2, filename: 'inventory_update.csv', rows: 1200, status: 'valid', uploadedAt: '1 hour ago' },
    { id: 3, filename: 'transactions_feb.csv', rows: 51000, status: 'error', errors: 12, uploadedAt: '2 hours ago' },
];

const mockPipelineStatus = [
    { name: 'Forecasting', status: 'completed', lastRun: '2 hours ago', duration: '4m 32s' },
    { name: 'GNN Build', status: 'completed', lastRun: '1 day ago', duration: '1m 15s' },
    { name: 'Adversarial Test', status: 'completed', lastRun: '2 hours ago', duration: '2m 45s' },
    { name: 'Data ETL', status: 'idle', lastRun: '3 hours ago', duration: '45s' },
];

const mockHighRiskSKUs = [
    { sku: 'SKU_042', store: 'Store_1', riskScore: 0.95, daysCover: 1.2 },
    { sku: 'SKU_108', store: 'Store_2', riskScore: 0.89, daysCover: 2.1 },
    { sku: 'SKU_156', store: 'Store_1', riskScore: 0.85, daysCover: 2.5 },
];

const mockAuditLogs = [
    { time: '14:30:22', user: 'admin', action: 'Triggered forecast pipeline' },
    { time: '14:28:15', user: 'admin', action: 'Approved data upload (trans_jan)' },
    { time: '14:15:00', user: 'system', action: 'Scheduled adversarial test completed' },
    { time: '13:45:30', user: 'jane@...', action: 'Created purchase order #1234' },
];

const mockUsers = [
    { id: 1, name: 'John Smith', email: 'john@company.com', role: 'admin' },
    { id: 2, name: 'Jane Doe', email: 'jane@company.com', role: 'manager' },
    { id: 3, name: 'Bob Wilson', email: 'bob@company.com', role: 'analyst' },
];

export default function AdminDashboard() {
    const router = useRouter();
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('overview');

    useEffect(() => {
        const userData = localStorage.getItem('user');
        if (!userData) {
            router.push('/auth/login');
            return;
        }
        const parsed = JSON.parse(userData);
        if (parsed.role !== 'admin') {
            router.push('/dashboard');
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
                <div className="animate-pulse text-2xl font-bold gradient-text">Loading Admin Dashboard...</div>
            </div>
        );
    }

    if (!user) return null;

    const getStatusBadge = (status: string) => {
        switch (status) {
            case 'completed': return <Badge variant="success">Completed</Badge>;
            case 'running': return <Badge variant="info">Running</Badge>;
            case 'idle': return <Badge variant="default">Idle</Badge>;
            case 'valid': return <Badge variant="success">Valid</Badge>;
            case 'validating': return <Badge variant="warning">Validating</Badge>;
            case 'error': return <Badge variant="error">Error</Badge>;
            default: return <Badge>{status}</Badge>;
        }
    };

    const getRiskColor = (score: number) => {
        if (score >= 0.8) return 'text-error';
        if (score >= 0.5) return 'text-warning';
        return 'text-success';
    };

    return (
        <div className="min-h-screen bg-background text-foreground">
            {/* Navigation */}
            <nav className="glass border-b border-white/10 sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-8">
                            <div className="flex items-center gap-2">
                                <div className="w-10 h-10 bg-gradient-to-br from-indigo-600 to-purple-600 rounded-lg flex items-center justify-center shadow-lg shadow-primary/20">
                                    <TrendingUpIcon className="text-white" size={24} />
                                </div>
                                <span className="text-xl font-bold gradient-text">StockSensePro</span>
                            </div>
                            <div className="hidden md:flex items-center gap-1">
                                {['overview', 'data', 'ml', 'testing', 'users', 'logs'].map((tab) => (
                                    <button
                                        key={tab}
                                        onClick={() => setActiveTab(tab)}
                                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === tab
                                            ? 'bg-primary/20 text-primary'
                                            : 'text-muted hover:text-foreground hover:bg-white/5'
                                            }`}
                                    >
                                        {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                    </button>
                                ))}
                            </div>
                        </div>
                        <div className="flex items-center gap-4">
                            <button className="p-2 hover:bg-white/5 rounded-lg transition-colors relative">
                                <BellIcon className="text-muted" size={20} />
                                <span className="absolute top-2 right-2 w-1.5 h-1.5 bg-error rounded-full"></span>
                            </button>
                            <div className="flex items-center gap-2">
                                <div className="w-8 h-8 rounded-full bg-gradient-to-r from-pink-500 to-rose-500 flex items-center justify-center text-xs font-bold text-white">
                                    {user.name ? user.name[0] : user.email[0].toUpperCase()}
                                </div>
                                <div className="hidden sm:block">
                                    <div className="text-sm font-medium">{user.name || user.email}</div>
                                    <div className="text-xs text-accent flex items-center gap-1">
                                        <ShieldIcon size={10} /> Admin
                                    </div>
                                </div>
                            </div>
                            <button onClick={handleLogout} className="p-2 hover:bg-white/5 rounded-lg transition-colors">
                                <LogoutIcon className="text-muted hover:text-error" size={18} />
                            </button>
                        </div>
                    </div>
                </div>
            </nav>

            <div className="max-w-7xl mx-auto px-6 py-8">
                {/* System Status Bar */}
                <div className="glass-strong rounded-xl p-4 mb-8 border border-white/10">
                    <div className="flex flex-wrap items-center justify-between gap-4">
                        <div className="flex items-center gap-6">
                            <div className="flex items-center gap-2">
                                <span className="w-2 h-2 bg-success rounded-full animate-pulse"></span>
                                <span className="text-sm text-muted">Database: <span className="text-success font-medium">Online</span></span>
                            </div>
                            <div className="flex items-center gap-2">
                                <span className="w-2 h-2 bg-success rounded-full"></span>
                                <span className="text-sm text-muted">ML Pipeline: <span className="text-foreground font-medium">Idle</span></span>
                            </div>
                            <div className="flex items-center gap-2">
                                <ClockIcon size={14} className="text-muted" />
                                <span className="text-sm text-muted">Last Forecast: <span className="text-foreground font-medium">2h ago</span></span>
                            </div>
                        </div>
                        <div className="text-sm text-muted">
                            {new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}
                        </div>
                    </div>
                </div>

                {/* Quick Stats */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                    <Card glass className="group">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs text-muted uppercase tracking-wider">Total SKUs</p>
                                <h3 className="text-2xl font-bold mt-1">248</h3>
                            </div>
                            <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center text-primary">
                                <DatabaseIcon size={20} />
                            </div>
                        </div>
                    </Card>

                    <Card glass className="group">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs text-muted uppercase tracking-wider">Stores</p>
                                <h3 className="text-2xl font-bold mt-1">3</h3>
                            </div>
                            <div className="w-10 h-10 bg-secondary/10 rounded-lg flex items-center justify-center text-secondary">
                                <ChartIcon size={20} />
                            </div>
                        </div>
                    </Card>

                    <Card glass className="group">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs text-muted uppercase tracking-wider">High Risk SKUs</p>
                                <h3 className="text-2xl font-bold mt-1 text-error">23</h3>
                            </div>
                            <div className="w-10 h-10 bg-error/10 rounded-lg flex items-center justify-center text-error">
                                <AlertIcon size={20} />
                            </div>
                        </div>
                    </Card>

                    <Card glass className="group">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs text-muted uppercase tracking-wider">Active Users</p>
                                <h3 className="text-2xl font-bold mt-1">8</h3>
                            </div>
                            <div className="w-10 h-10 bg-accent/10 rounded-lg flex items-center justify-center text-accent">
                                <UserIcon size={20} />
                            </div>
                        </div>
                    </Card>
                </div>

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                    {/* Data Management */}
                    <Card glass>
                        <CardHeader>
                            <div className="flex items-center justify-between">
                                <div>
                                    <CardTitle className="flex items-center gap-2">
                                        <UploadIcon size={18} className="text-primary" />
                                        Data Management
                                    </CardTitle>
                                    <CardDescription>Upload and manage transaction data</CardDescription>
                                </div>
                                <Button variant="primary" size="sm">
                                    <UploadIcon size={14} />
                                    Upload CSV
                                </Button>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-3">
                                <p className="text-xs text-muted uppercase tracking-wider mb-2">Staging Queue</p>
                                {mockStagingQueue.map((item) => (
                                    <div key={item.id} className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                                        <div className="flex items-center gap-3">
                                            <div className="w-8 h-8 bg-surface-elevated rounded flex items-center justify-center">
                                                <DatabaseIcon size={14} className="text-muted" />
                                            </div>
                                            <div>
                                                <div className="text-sm font-medium">{item.filename}</div>
                                                <div className="text-xs text-muted">{item.rows.toLocaleString()} rows • {item.uploadedAt}</div>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            {getStatusBadge(item.status)}
                                            {item.status === 'valid' && (
                                                <Button variant="ghost" size="sm">Approve</Button>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>

                    {/* ML Operations */}
                    <Card glass>
                        <CardHeader>
                            <div className="flex items-center justify-between">
                                <div>
                                    <CardTitle className="flex items-center gap-2">
                                        <ActivityIcon size={18} className="text-secondary" />
                                        ML Operations
                                    </CardTitle>
                                    <CardDescription>Manage forecasting models</CardDescription>
                                </div>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-2 gap-3 mb-4">
                                <div className="p-4 bg-white/5 rounded-lg border border-white/5">
                                    <p className="text-xs text-muted uppercase tracking-wider mb-2">Active Model</p>
                                    <p className="text-lg font-bold text-primary">TFT v2.1</p>
                                    <p className="text-xs text-muted mt-1">MAE: 4.23 | MAPE: 8.5%</p>
                                </div>
                                <div className="p-4 bg-white/5 rounded-lg border border-white/5">
                                    <p className="text-xs text-muted uppercase tracking-wider mb-2">GNN Graph</p>
                                    <p className="text-lg font-bold">248 nodes</p>
                                    <p className="text-xs text-muted mt-1">1,234 edges</p>
                                </div>
                            </div>
                            <div className="flex flex-wrap gap-2">
                                <Button variant="primary" size="sm">
                                    <ActivityIcon size={14} />
                                    Run Forecast
                                </Button>
                                <Button variant="secondary" size="sm">
                                    <RefreshIcon size={14} />
                                    Retrain Model
                                </Button>
                                <Button variant="ghost" size="sm">
                                    Generate GNN
                                </Button>
                            </div>
                        </CardContent>
                    </Card>

                    {/* Adversarial Testing */}
                    <Card glass>
                        <CardHeader>
                            <div className="flex items-center justify-between">
                                <div>
                                    <CardTitle className="flex items-center gap-2">
                                        <ShieldIcon size={18} className="text-accent" />
                                        Adversarial Testing
                                    </CardTitle>
                                    <CardDescription>Stress test forecasts</CardDescription>
                                </div>
                                <Button variant="secondary" size="sm">
                                    Run Test
                                </Button>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-3 gap-3 mb-4">
                                <div className="text-center p-3 bg-error/10 rounded-lg">
                                    <p className="text-2xl font-bold text-error">23</p>
                                    <p className="text-xs text-muted">High Risk</p>
                                </div>
                                <div className="text-center p-3 bg-warning/10 rounded-lg">
                                    <p className="text-2xl font-bold text-warning">45</p>
                                    <p className="text-xs text-muted">Medium</p>
                                </div>
                                <div className="text-center p-3 bg-success/10 rounded-lg">
                                    <p className="text-2xl font-bold text-success">180</p>
                                    <p className="text-xs text-muted">Low Risk</p>
                                </div>
                            </div>
                            <p className="text-xs text-muted uppercase tracking-wider mb-2">High Risk SKUs</p>
                            <div className="space-y-2">
                                {mockHighRiskSKUs.map((item, idx) => (
                                    <div key={idx} className="flex items-center justify-between p-2 bg-white/5 rounded-lg">
                                        <div className="flex items-center gap-2">
                                            <span className="font-mono text-sm">{item.sku}</span>
                                            <span className="text-xs text-muted">{item.store}</span>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <span className={`text-sm font-bold ${getRiskColor(item.riskScore)}`}>
                                                {(item.riskScore * 100).toFixed(0)}%
                                            </span>
                                            <span className="text-xs text-muted">{item.daysCover}d cover</span>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>

                    {/* User Management */}
                    <Card glass>
                        <CardHeader>
                            <div className="flex items-center justify-between">
                                <div>
                                    <CardTitle className="flex items-center gap-2">
                                        <UserIcon size={18} className="text-info" />
                                        User Management
                                    </CardTitle>
                                    <CardDescription>Manage roles and permissions</CardDescription>
                                </div>
                                <Button variant="primary" size="sm">
                                    Add User
                                </Button>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>User</TableHead>
                                        <TableHead>Role</TableHead>
                                        <TableHead className="text-right">Actions</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {mockUsers.map((u) => (
                                        <TableRow key={u.id}>
                                            <TableCell>
                                                <div>
                                                    <div className="font-medium text-sm">{u.name}</div>
                                                    <div className="text-xs text-muted">{u.email}</div>
                                                </div>
                                            </TableCell>
                                            <TableCell>
                                                <Badge variant={u.role === 'admin' ? 'info' : u.role === 'manager' ? 'warning' : 'default'}>
                                                    {u.role}
                                                </Badge>
                                            </TableCell>
                                            <TableCell className="text-right">
                                                <Button variant="ghost" size="sm">Edit</Button>
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </CardContent>
                    </Card>
                </div>

                {/* System Monitoring & Audit Logs */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Pipeline Status */}
                    <Card glass>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <SettingsIcon size={18} className="text-primary" />
                                Pipeline Status
                            </CardTitle>
                            <CardDescription>Monitor running jobs</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-3">
                                {mockPipelineStatus.map((pipeline, idx) => (
                                    <div key={idx} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                                        <div className="flex items-center gap-3">
                                            <div className={`w-2 h-2 rounded-full ${pipeline.status === 'completed' ? 'bg-success' :
                                                pipeline.status === 'running' ? 'bg-info animate-pulse' : 'bg-muted'
                                                }`}></div>
                                            <span className="font-medium text-sm">{pipeline.name}</span>
                                        </div>
                                        <div className="flex items-center gap-4 text-xs text-muted">
                                            <span>{pipeline.lastRun}</span>
                                            <span>{pipeline.duration}</span>
                                            {getStatusBadge(pipeline.status)}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>

                    {/* Audit Logs */}
                    <Card glass>
                        <CardHeader>
                            <div className="flex items-center justify-between">
                                <div>
                                    <CardTitle className="flex items-center gap-2">
                                        <ClockIcon size={18} className="text-secondary" />
                                        Recent Activity
                                    </CardTitle>
                                    <CardDescription>System audit logs</CardDescription>
                                </div>
                                <Button variant="ghost" size="sm">Export</Button>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-3">
                                {mockAuditLogs.map((log, idx) => (
                                    <div key={idx} className="flex items-start gap-3 p-3 bg-white/5 rounded-lg">
                                        <div className="w-8 h-8 bg-surface-elevated rounded-full flex items-center justify-center flex-shrink-0">
                                            <CheckIcon size={12} className="text-success" />
                                        </div>
                                        <div className="flex-1 min-w-0">
                                            <p className="text-sm">{log.action}</p>
                                            <p className="text-xs text-muted mt-1">
                                                <span className="font-medium">{log.user}</span> • {log.time}
                                            </p>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>
                </div>
            </div>
        </div>
    );
}
