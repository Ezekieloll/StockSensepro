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
    ChartIcon,
    UserIcon,
    LogoutIcon,
    BellIcon,
    BriefcaseIcon,
    SearchIcon,
    AlertIcon,
    CheckIcon,
    ClockIcon,
    DatabaseIcon,
    RefreshIcon,
} from '@/components/Icons';
import InsightAssistant from '@/components/InsightAssistant';

interface User {
    id?: number;
    name?: string;
    email: string;
    role: string;
}

interface ForecastItem {
    sku: string;
    product_name: string;
    category: string;
    store_id: string;
    current_stock: number;
    seven_day_forecast: number;
    stock_status: string;
    daily_forecasts: Array<{ date: string; predicted_demand: number }>;
}

interface ForecastAlert {
    id: string;
    type: string;
    severity: string;
    product_name: string;
    sku: string;
    store_id: string;
    current_stock: number;
    predicted_demand: number;
    message: string;
}

interface ForecastSummary {
    total_products: number;
    critical_stock_count: number;
    low_stock_count: number;
    avg_confidence: number;
}

// Fallback mock data
const mockPurchaseOrders = [
    { id: 'PO-2024-001', supplier: 'Dairy Fresh Co.', items: 5, total: 12500, status: 'pending', createdAt: '2 hours ago' },
    { id: 'PO-2024-002', supplier: 'Farm Produce Ltd.', items: 8, total: 8750, status: 'approved', createdAt: '1 day ago' },
    { id: 'PO-2024-003', supplier: 'Bakery Supplies', items: 3, total: 3200, status: 'delivered', createdAt: '3 days ago' },
];

const mockInventorySummary = [
    { store: 'S1', totalSKUs: 86, lowStock: 12, criticalStock: 3, value: 245000 },
    { store: 'S2', totalSKUs: 82, lowStock: 8, criticalStock: 5, value: 198000 },
    { store: 'S3', totalSKUs: 80, lowStock: 15, criticalStock: 2, value: 178000 },
];

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function ManagerDashboard() {
    const router = useRouter();
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('overview');

    // Forecast data state
    const [forecasts, setForecasts] = useState<ForecastItem[]>([]);
    const [alerts, setAlerts] = useState<ForecastAlert[]>([]);
    const [summary, setSummary] = useState<ForecastSummary | null>(null);
    const [forecastLoading, setForecastLoading] = useState(false);
    const [selectedStore, setSelectedStore] = useState<string>('');

    useEffect(() => {
        const userData = localStorage.getItem('user');
        if (!userData) {
            router.push('/auth/login');
            return;
        }
        const parsed = JSON.parse(userData);
        if (parsed.role !== 'manager') {
            if (parsed.role === 'admin') {
                router.push('/admin');
            } else {
                router.push('/dashboard');
            }
            return;
        }
        setUser(parsed);
        setLoading(false);

        // Fetch forecast data
        fetchForecastData();
    }, [router]);

    const fetchForecastData = async (storeId?: string) => {
        setForecastLoading(true);
        try {
            const storeParam = storeId ? `?store_id=${storeId}` : '';

            // Fetch forecasts by product
            const forecastRes = await fetch(`${API_URL}/forecast/by-product${storeParam}`);
            if (forecastRes.ok) {
                const forecastData = await forecastRes.json();
                setForecasts(forecastData);
            }

            // Fetch alerts
            const alertsRes = await fetch(`${API_URL}/forecast/alerts${storeParam}`);
            if (alertsRes.ok) {
                const alertsData = await alertsRes.json();
                setAlerts(alertsData);
            }

            // Fetch summary
            const summaryRes = await fetch(`${API_URL}/forecast/summary${storeParam}`);
            if (summaryRes.ok) {
                const summaryData = await summaryRes.json();
                setSummary(summaryData);
            }
        } catch (error) {
            console.error('Error fetching forecast data:', error);
        } finally {
            setForecastLoading(false);
        }
    };

    const handleStoreChange = (storeId: string) => {
        setSelectedStore(storeId);
        fetchForecastData(storeId || undefined);
    };

    const handleLogout = () => {
        localStorage.removeItem('user');
        localStorage.removeItem('token');
        router.push('/');
    };

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-background">
                <div className="animate-pulse text-2xl font-bold gradient-text">Loading Manager Dashboard...</div>
            </div>
        );
    }

    if (!user) return null;

    const getStatusBadge = (status: string) => {
        switch (status) {
            case 'ok': return <Badge variant="success">OK</Badge>;
            case 'low': return <Badge variant="warning">Low</Badge>;
            case 'critical': return <Badge variant="error">Critical</Badge>;
            case 'pending': return <Badge variant="warning">Pending</Badge>;
            case 'approved': return <Badge variant="info">Approved</Badge>;
            case 'delivered': return <Badge variant="success">Delivered</Badge>;
            default: return <Badge>{status}</Badge>;
        }
    };

    const getSeverityColor = (severity: string) => {
        switch (severity) {
            case 'high': return 'border-l-error bg-error/5';
            case 'medium': return 'border-l-warning bg-warning/5';
            case 'low': return 'border-l-info bg-info/5';
            default: return 'border-l-muted';
        }
    };

    return (
        <div className="min-h-screen bg-background text-foreground">
            {/* Navigation */}
            <nav className="glass border-b border-white/10 sticky top-0 z-50">
                <div className="max-w-7xl mx-auto px-6 py-4">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-8">
                            <div className="flex items-center gap-2">
                                <div className="w-10 h-10 bg-gradient-to-br from-purple-600 to-pink-600 rounded-lg flex items-center justify-center shadow-lg shadow-secondary/20">
                                    <TrendingUpIcon className="text-white" size={24} />
                                </div>
                                <span className="text-xl font-bold gradient-text">StockSensePro</span>
                            </div>
                            <div className="hidden md:flex items-center gap-1">
                                {['overview', 'forecasts', 'orders', 'inventory', 'alerts'].map((tab) => (
                                    <button
                                        key={tab}
                                        onClick={() => tab === 'forecasts' ? router.push('/forecasts') : setActiveTab(tab)}
                                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === tab
                                            ? 'bg-secondary/20 text-secondary'
                                            : 'text-muted hover:text-foreground hover:bg-white/5'
                                            }`}
                                    >
                                        {tab.charAt(0).toUpperCase() + tab.slice(1)}
                                    </button>
                                ))}
                            </div>
                        </div>
                        <div className="flex items-center gap-4">
                            {/* Store Filter */}
                            <select
                                value={selectedStore}
                                onChange={(e) => handleStoreChange(e.target.value)}
                                className="bg-slate-900 text-white border border-white/20 rounded-lg py-1.5 px-3 text-xs focus:ring-1 focus:ring-secondary outline-none"
                            >
                                <option value="">All Stores</option>
                                <option value="S1">Store S1</option>
                                <option value="S2">Store S2</option>
                                <option value="S3">Store S3</option>
                            </select>
                            <button
                                onClick={() => fetchForecastData(selectedStore || undefined)}
                                className="p-2 hover:bg-white/5 rounded-lg transition-colors"
                                title="Refresh data"
                            >
                                <RefreshIcon className={`text-muted ${forecastLoading ? 'animate-spin' : ''}`} size={18} />
                            </button>
                            <div className="relative hidden sm:block">
                                <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" size={16} />
                                <input
                                    type="text"
                                    placeholder="Search SKU..."
                                    className="bg-slate-900 text-white border border-white/20 rounded-full py-1.5 pl-9 pr-4 text-xs focus:ring-1 focus:ring-secondary outline-none transition-all w-40 focus:w-56"
                                />
                            </div>
                            <button className="p-2 hover:bg-white/5 rounded-lg transition-colors relative">
                                <BellIcon className="text-muted" size={20} />
                                {alerts.length > 0 && (
                                    <span className="absolute top-1 right-1 w-2 h-2 bg-error rounded-full animate-pulse"></span>
                                )}
                            </button>
                            <div className="flex items-center gap-2">
                                <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center text-xs font-bold text-white">
                                    {user.name ? user.name[0] : user.email[0].toUpperCase()}
                                </div>
                                <div className="hidden sm:block">
                                    <div className="text-sm font-medium">{user.name || user.email}</div>
                                    <div className="text-xs text-secondary flex items-center gap-1">
                                        <BriefcaseIcon size={10} /> Manager
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
                {/* Header */}
                <div className="flex flex-col md:flex-row md:items-end justify-between gap-4 mb-8">
                    <div>
                        <h1 className="text-3xl font-bold tracking-tight mb-2">
                            Welcome back, <span className="gradient-text">{user.name || user.email.split('@')[0]}</span>
                        </h1>
                        <p className="text-muted text-sm flex items-center gap-2">
                            <BriefcaseIcon size={14} className="text-secondary" />
                            Store Manager Dashboard
                            <span className="w-1 h-1 bg-white/20 rounded-full"></span>
                            <span>{new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}</span>
                            {summary && (
                                <>
                                    <span className="w-1 h-1 bg-white/20 rounded-full"></span>
                                    <span className="text-success">Model Confidence: {(summary.avg_confidence * 100).toFixed(0)}%</span>
                                </>
                            )}
                        </p>
                    </div>
                    <div className="flex items-center gap-3">
                        <Button variant="secondary" size="sm">
                            <ChartIcon size={14} />
                            View Reports
                        </Button>
                        <Button variant="primary" size="sm">
                            Create PO
                        </Button>
                    </div>
                </div>

                {/* Quick Stats */}
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                    <Card glass className="group hover:border-success/30 transition-all">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs text-muted uppercase tracking-wider">Total Inventory Value</p>
                                <h3 className="text-2xl font-bold mt-1">$621K</h3>
                                <div className="flex items-center gap-1 mt-1">
                                    <TrendingUpIcon size={12} className="text-success" />
                                    <span className="text-xs text-success">+2.3%</span>
                                </div>
                            </div>
                            <div className="w-10 h-10 bg-success/10 rounded-lg flex items-center justify-center text-success">
                                <DatabaseIcon size={20} />
                            </div>
                        </div>
                    </Card>

                    <Card glass className="group hover:border-warning/30 transition-all">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs text-muted uppercase tracking-wider">Low Stock Items</p>
                                <h3 className="text-2xl font-bold mt-1 text-warning">{summary?.low_stock_count || 0}</h3>
                                <p className="text-xs text-muted mt-1">From ML predictions</p>
                            </div>
                            <div className="w-10 h-10 bg-warning/10 rounded-lg flex items-center justify-center text-warning">
                                <AlertIcon size={20} />
                            </div>
                        </div>
                    </Card>

                    <Card glass className="group hover:border-error/30 transition-all">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs text-muted uppercase tracking-wider">Critical Stock</p>
                                <h3 className="text-2xl font-bold mt-1 text-error">{summary?.critical_stock_count || 0}</h3>
                                <p className="text-xs text-muted mt-1">Needs immediate action</p>
                            </div>
                            <div className="w-10 h-10 bg-error/10 rounded-lg flex items-center justify-center text-error">
                                <TrendingDownIcon size={20} />
                            </div>
                        </div>
                    </Card>

                    <Card glass className="group hover:border-info/30 transition-all">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-xs text-muted uppercase tracking-wider">Products Tracked</p>
                                <h3 className="text-2xl font-bold mt-1">{summary?.total_products || 0}</h3>
                                <p className="text-xs text-muted mt-1">TFT+GNN Model</p>
                            </div>
                            <div className="w-10 h-10 bg-info/10 rounded-lg flex items-center justify-center text-info">
                                <ChartIcon size={20} />
                            </div>
                        </div>
                    </Card>
                </div>

                {/* Main Content Grid */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
                    {/* Alerts Panel */}
                    <Card glass className="lg:col-span-1">
                        <CardHeader>
                            <div className="flex items-center justify-between">
                                <CardTitle className="flex items-center gap-2 text-lg">
                                    <BellIcon size={18} className="text-error" />
                                    ML-Generated Alerts
                                    <span className="ml-1 px-2 py-0.5 bg-error/20 text-error text-xs rounded-full">{alerts.length}</span>
                                </CardTitle>
                            </div>
                        </CardHeader>
                        <CardContent>
                            {forecastLoading ? (
                                <div className="text-center py-8 text-muted">Loading alerts...</div>
                            ) : alerts.length === 0 ? (
                                <div className="text-center py-8 text-muted">
                                    <CheckIcon size={32} className="mx-auto mb-2 text-success" />
                                    <p>No alerts - all stock levels healthy!</p>
                                </div>
                            ) : (
                                <div className="space-y-3">
                                    {alerts.slice(0, 5).map((alert) => (
                                        <div
                                            key={alert.id}
                                            className={`p-3 rounded-lg border-l-4 ${getSeverityColor(alert.severity)}`}
                                        >
                                            <p className="text-sm font-medium">{alert.message}</p>
                                            <p className="text-xs text-muted mt-1">
                                                Store: {alert.store_id} • Stock: {alert.current_stock} • Forecast: {alert.predicted_demand.toFixed(0)}
                                            </p>
                                        </div>
                                    ))}
                                </div>
                            )}
                            <Button variant="ghost" className="w-full mt-4 text-xs">
                                View All Alerts
                            </Button>
                        </CardContent>
                    </Card>

                    {/* Forecasts Table */}
                    <Card glass className="lg:col-span-2">
                        <CardHeader>
                            <div className="flex items-center justify-between">
                                <div>
                                    <CardTitle className="flex items-center gap-2 text-lg">
                                        <ChartIcon size={18} className="text-primary" />
                                        ML Demand Forecasts
                                    </CardTitle>
                                    <CardDescription>7-day demand predictions from TFT+GNN model</CardDescription>
                                </div>
                                <div className="flex items-center gap-2">
                                    <Badge variant="success">Live</Badge>
                                    <Button variant="ghost" size="sm" onClick={() => router.push('/forecasts')}>View All</Button>
                                </div>
                            </div>
                        </CardHeader>
                        <CardContent>
                            {forecastLoading ? (
                                <div className="text-center py-8 text-muted">Loading forecasts...</div>
                            ) : (
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead>Product</TableHead>
                                            <TableHead>Store</TableHead>
                                            <TableHead className="text-right">Stock</TableHead>
                                            <TableHead className="text-right">7d Forecast</TableHead>
                                            <TableHead className="text-right">Status</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {forecasts.slice(0, 8).map((item) => (
                                            <TableRow key={`${item.sku}-${item.store_id}`} className="hover:bg-white/5">
                                                <TableCell>
                                                    <div>
                                                        <div className="font-medium text-sm">{item.product_name}</div>
                                                        <div className="text-xs text-muted font-mono">{item.sku}</div>
                                                    </div>
                                                </TableCell>
                                                <TableCell className="text-sm">{item.store_id}</TableCell>
                                                <TableCell className="text-right font-medium">{item.current_stock}</TableCell>
                                                <TableCell className="text-right">
                                                    <span className={item.seven_day_forecast > item.current_stock ? 'text-error' : 'text-foreground'}>
                                                        {item.seven_day_forecast.toFixed(0)}
                                                    </span>
                                                </TableCell>
                                                <TableCell className="text-right">
                                                    {getStatusBadge(item.stock_status)}
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            )}
                        </CardContent>
                    </Card>
                </div>

                {/* Bottom Section */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                    {/* Purchase Orders */}
                    <Card glass>
                        <CardHeader>
                            <div className="flex items-center justify-between">
                                <div>
                                    <CardTitle className="flex items-center gap-2 text-lg">
                                        <BriefcaseIcon size={18} className="text-secondary" />
                                        Purchase Orders
                                    </CardTitle>
                                    <CardDescription>Recent orders and their status</CardDescription>
                                </div>
                                <Button variant="primary" size="sm">+ New PO</Button>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-3">
                                {mockPurchaseOrders.map((po) => (
                                    <div key={po.id} className="flex items-center justify-between p-4 bg-white/5 rounded-lg border border-white/5 hover:border-white/10 transition-all">
                                        <div className="flex items-center gap-4">
                                            <div className="w-10 h-10 bg-surface-elevated rounded-lg flex items-center justify-center">
                                                <BriefcaseIcon size={16} className="text-muted" />
                                            </div>
                                            <div>
                                                <div className="font-medium text-sm">{po.id}</div>
                                                <div className="text-xs text-muted">{po.supplier} • {po.items} items</div>
                                            </div>
                                        </div>
                                        <div className="text-right">
                                            <div className="font-bold text-sm">${po.total.toLocaleString()}</div>
                                            <div className="mt-1">{getStatusBadge(po.status)}</div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>

                    {/* Inventory by Store */}
                    <Card glass>
                        <CardHeader>
                            <div className="flex items-center justify-between">
                                <div>
                                    <CardTitle className="flex items-center gap-2 text-lg">
                                        <DatabaseIcon size={18} className="text-info" />
                                        Inventory by Store
                                    </CardTitle>
                                    <CardDescription>Stock health across locations</CardDescription>
                                </div>
                                <Button variant="ghost" size="sm">Manage</Button>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-4">
                                {mockInventorySummary.map((store) => (
                                    <div key={store.store} className="p-4 bg-white/5 rounded-lg border border-white/5">
                                        <div className="flex items-center justify-between mb-3">
                                            <div className="font-medium">{store.store}</div>
                                            <div className="text-sm text-muted">${(store.value / 1000).toFixed(0)}K value</div>
                                        </div>
                                        <div className="flex items-center gap-4 text-sm">
                                            <div className="flex items-center gap-2">
                                                <span className="w-2 h-2 bg-success rounded-full"></span>
                                                <span className="text-muted">{store.totalSKUs - store.lowStock - store.criticalStock} OK</span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <span className="w-2 h-2 bg-warning rounded-full"></span>
                                                <span className="text-muted">{store.lowStock} Low</span>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                <span className="w-2 h-2 bg-error rounded-full"></span>
                                                <span className="text-muted">{store.criticalStock} Critical</span>
                                            </div>
                                        </div>
                                        {/* Progress bar */}
                                        <div className="mt-3 h-2 bg-surface-elevated rounded-full overflow-hidden flex">
                                            <div
                                                className="bg-success h-full"
                                                style={{ width: `${((store.totalSKUs - store.lowStock - store.criticalStock) / store.totalSKUs) * 100}%` }}
                                            ></div>
                                            <div
                                                className="bg-warning h-full"
                                                style={{ width: `${(store.lowStock / store.totalSKUs) * 100}%` }}
                                            ></div>
                                            <div
                                                className="bg-error h-full"
                                                style={{ width: `${(store.criticalStock / store.totalSKUs) * 100}%` }}
                                            ></div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </CardContent>
                    </Card>
                </div>

                {/* LLM Assistant */}
                <InsightAssistant
                    forecasts={forecasts}
                    alerts={alerts}
                    summary={summary}
                />
            </div>
        </div>
    );
}
