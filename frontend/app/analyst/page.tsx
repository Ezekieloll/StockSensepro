'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Card, { CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import Toast from '@/components/ui/Toast';
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from '@/components/ui/Table';
import {
    TrendingUpIcon,
    TrendingDownIcon,
    ChartIcon,
    UserIcon,
    LogoutIcon,
    BellIcon,
    SearchIcon,
    ActivityIcon,
    AlertIcon,
    CheckIcon,
    DatabaseIcon,
    RefreshIcon,
    CalendarIcon,
} from '@/components/ui/Icons';
import AnalysisView from './components/AnalysisView';
import ProductAccuracyTable from './components/ProductAccuracyTable';
import AccuracySummaryCard from './components/AccuracySummaryCard';
import GNN3DVisualizer from './components/GNN3DVisualizer';
import ChatPanel from '@/components/ui/ChatPanel';

interface User {
    id?: number;
    name?: string;
    email: string;
    role: string;
}

interface ModelMetric {
    model: string;
    type: string;
    mae: number;
    mape: number;
    wape: number;
    status: string;
    trained_at: string;
    epochs: number;
}

interface ModelMetricsResponse {
    models: ModelMetric[];
    active_model: ModelMetric | null;
    total_models: number;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Mock data for analyst view (fallback)
const mockModelMetrics = [
    { model: 'TFT v2.1', type: 'Temporal Fusion Transformer', mae: 4.23, mape: 8.5, wape: 7.2, status: 'active', trained_at: '2026-01-20', epochs: 50 },
    { model: 'LSTM v1.8', type: 'LSTM', mae: 5.67, mape: 11.2, wape: 9.8, status: 'standby', trained_at: '2026-01-15', epochs: 40 },
    { model: 'LSTM+GNN v1.2', type: 'LSTM+GNN', mae: 4.89, mape: 9.8, wape: 8.4, status: 'standby', trained_at: '2026-01-10', epochs: 45 },
    { model: 'Transformer v1.0', type: 'Transformer', mae: 6.12, mape: 12.5, wape: 10.2, status: 'archived', trained_at: '2025-12-28', epochs: 30 },
];

const mockForecastAccuracy = [
    { sku: 'SKU_001', name: 'Organic Milk 1L', predicted: 245, actual: 238, error: 2.9, trend: 'up' },
    { sku: 'SKU_015', name: 'Whole Wheat Bread', predicted: 180, actual: 195, error: 7.7, trend: 'down' },
    { sku: 'SKU_042', name: 'Fresh Eggs 12pk', predicted: 312, actual: 298, error: 4.7, trend: 'up' },
    { sku: 'SKU_078', name: 'Orange Juice 2L', predicted: 156, actual: 162, error: 3.7, trend: 'down' },
    { sku: 'SKU_103', name: 'Greek Yogurt', predicted: 88, actual: 92, error: 4.3, trend: 'up' },
];

const mockAnomalies = [
    { id: 1, sku: 'SKU_156', type: 'Demand Spike', description: 'Unusual 3x demand increase detected', date: 'Jan 15', status: 'new' },
    { id: 2, sku: 'SKU_089', type: 'Pattern Break', description: 'Weekly seasonality pattern disrupted', date: 'Jan 14', status: 'investigating' },
    { id: 3, sku: 'SKU_201', type: 'Forecast Drift', description: 'Model consistently over-predicting by 15%', date: 'Jan 13', status: 'resolved' },
];

const mockGNNInsights = [
    { sku: 'SKU_001', influencedBy: ['SKU_015', 'SKU_042'], influenceStrength: 0.82, category: 'Dairy' },
    { sku: 'SKU_015', influencedBy: ['SKU_001', 'SKU_078'], influenceStrength: 0.65, category: 'Bakery' },
    { sku: 'SKU_042', influencedBy: ['SKU_001', 'SKU_103'], influenceStrength: 0.78, category: 'Dairy' },
];

const mockSimulationResults = [
    { scenario: 'Baseline', demand: 1245, risk: 'low', confidence: 95 },
    { scenario: 'Demand Spike +50%', demand: 1868, risk: 'medium', confidence: 88 },
    { scenario: 'Holiday Season', demand: 2156, risk: 'high', confidence: 82 },
    { scenario: 'Weather Shock', demand: 1456, risk: 'medium', confidence: 85 },
];

export default function AnalystDashboard() {
    const router = useRouter();
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('overview');
    const [selectedModel, setSelectedModel] = useState('TFT v2.1');
    const [modelMetrics, setModelMetrics] = useState<ModelMetric[]>(mockModelMetrics);
    const [metricsLoading, setMetricsLoading] = useState(false);
    const [showExportMenu, setShowExportMenu] = useState(false);
    const [notification, setNotification] = useState<{ message: string; type: 'success' | 'error' | 'warning' | 'info' } | null>(null);
    const [simulationResults, setSimulationResults] = useState(mockSimulationResults);
    const [simulationLoading, setSimulationLoading] = useState(false);
    const [customScenario, setCustomScenario] = useState('');
    const [showCustomInput, setShowCustomInput] = useState(false);
    const [isChatOpen, setIsChatOpen] = useState(false);

    useEffect(() => {
        const userData = localStorage.getItem('user');
        if (!userData) {
            router.push('/auth/login');
            return;
        }
        const parsed = JSON.parse(userData);
        if (parsed.role !== 'analyst') {
            if (parsed.role === 'admin') {
                router.push('/admin');
            } else if (parsed.role === 'manager') {
                router.push('/manager');
            }
            return;
        }
        setUser(parsed);
        setLoading(false);
        
        // Fetch model metrics
        fetchModelMetrics();
    }, [router]);

    // Close export menu when clicking outside
    useEffect(() => {
        const handleClickOutside = (event: MouseEvent) => {
            if (showExportMenu) {
                const target = event.target as HTMLElement;
                if (!target.closest('.export-menu-container')) {
                    setShowExportMenu(false);
                }
            }
        };

        document.addEventListener('mousedown', handleClickOutside);
        return () => document.removeEventListener('mousedown', handleClickOutside);
    }, [showExportMenu]);

    const fetchModelMetrics = async () => {
        setMetricsLoading(true);
        try {
            const response = await fetch(`${API_URL}/analytics/model-metrics`);
            if (response.ok) {
                const data: ModelMetricsResponse = await response.json();
                setModelMetrics(data.models);
                if (data.active_model) {
                    setSelectedModel(data.active_model.model);
                }
            }
        } catch (error) {
            console.error('Error fetching model metrics:', error);
            // Fall back to mock data (already set)
        } finally {
            setMetricsLoading(false);
        }
    };

    const handleExportReport = async () => {
        try {
            // Download full analysis report
            const response = await fetch(`${API_URL}/analytics/export/full-analysis`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `analysis_report_${new Date().toISOString().split('T')[0]}.csv`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                setShowExportMenu(false);
                setNotification({ message: 'Full analysis report exported successfully', type: 'success' });
            } else {
                setNotification({ message: 'Failed to export report. Please try again.', type: 'error' });
            }
        } catch (error) {
            console.error('Error exporting report:', error);
            setNotification({ message: 'Failed to export report. Check console for details.', type: 'error' });
        }
    };

    const handleExportModelPerformance = async () => {
        try {
            const response = await fetch(`${API_URL}/analytics/export/model-performance`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `model_performance_${new Date().toISOString().split('T')[0]}.csv`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                setShowExportMenu(false);
                setNotification({ message: 'Model performance data exported successfully', type: 'success' });
            }
        } catch (error) {
            console.error('Error exporting model performance:', error);
            setNotification({ message: 'Failed to export. Please try again.', type: 'error' });
        }
    };

    const handleExportVolumeStats = async () => {
        try {
            const response = await fetch(`${API_URL}/analytics/export/sku-volume-stats`);
            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `sku_volume_stats_${new Date().toISOString().split('T')[0]}.csv`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                setShowExportMenu(false);
                setNotification({ message: 'SKU volume statistics exported successfully', type: 'success' });
            }
        } catch (error) {
            console.error('Error exporting volume stats:', error);
            setNotification({ message: 'Failed to export. Please try again.', type: 'error' });
        }
    };

    const runSimulation = async () => {
        setSimulationLoading(true);
        try {
            const response = await fetch(`${API_URL}/simulations/run`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
            });
            
            if (response.ok) {
                const data = await response.json();
                setSimulationResults(data);
                setNotification({ message: 'Simulation completed successfully', type: 'success' });
            } else {
                setNotification({ message: 'Simulation failed. Using cached results.', type: 'warning' });
            }
        } catch (error) {
            console.error('Error running simulation:', error);
            setNotification({ message: 'Simulation error. Using cached results.', type: 'warning' });
        } finally {
            setSimulationLoading(false);
        }
    };

    const runCustomScenario = async () => {
        if (!customScenario.trim()) {
            setNotification({ message: 'Please enter a scenario description', type: 'warning' });
            return;
        }
        
        setSimulationLoading(true);
        try {
            const response = await fetch(`${API_URL}/simulations/custom?scenario_text=${encodeURIComponent(customScenario)}`, {
                method: 'POST',
            });
            
            if (response.ok) {
                const result = await response.json();
                
                // Add AI result to the simulation results
                setSimulationResults(prev => [
                    result,
                    ...prev.slice(0, 3)  // Keep top 3 default scenarios
                ]);
                
                const reasoning = result.ai_reasoning ? ` (${result.ai_reasoning})` : '';
                setNotification({ 
                    message: `AI analysis complete: ${result.demand.toLocaleString()} units projected${reasoning}`, 
                    type: 'success' 
                });
                
                setCustomScenario('');
                setShowCustomInput(false);
            } else {
                setNotification({ message: 'AI analysis failed. Try again.', type: 'error' });
            }
        } catch (error) {
            console.error('Error running custom scenario:', error);
            setNotification({ message: 'Failed to analyze scenario.', type: 'error' });
        } finally {
            setSimulationLoading(false);
        }
    };

    const handleChatScenario = (result: any) => {
        // Add chat-analyzed scenario to simulation results
        setSimulationResults(prev => [
            result,
            ...prev.slice(0, 3)  // Keep top 3 default scenarios
        ]);
    };

    const handleLogout = () => {
        localStorage.removeItem('user');
        localStorage.removeItem('token');
        router.push('/');
    };

    if (loading) {
        return (
            <div className="min-h-screen flex items-center justify-center bg-background">
                <div className="animate-pulse text-2xl font-bold gradient-text">Loading Analyst Dashboard...</div>
            </div>
        );
    }

    if (!user) return null;

    const getStatusBadge = (status: string) => {
        switch (status) {
            case 'active': return <Badge variant="success">Active</Badge>;
            case 'standby': return <Badge variant="info">Standby</Badge>;
            case 'archived': return <Badge variant="default">Archived</Badge>;
            case 'new': return <Badge variant="error">New</Badge>;
            case 'investigating': return <Badge variant="warning">Investigating</Badge>;
            case 'resolved': return <Badge variant="success">Resolved</Badge>;
            default: return <Badge>{status}</Badge>;
        }
    };

    const getRiskBadge = (risk: string) => {
        switch (risk) {
            case 'low': return <Badge variant="success">Low Risk</Badge>;
            case 'medium': return <Badge variant="warning">Medium</Badge>;
            case 'high': return <Badge variant="error">High Risk</Badge>;
            default: return <Badge>{risk}</Badge>;
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
                                <div className="w-10 h-10 bg-gradient-to-br from-cyan-600 to-blue-600 rounded-lg flex items-center justify-center shadow-lg shadow-info/20">
                                    <TrendingUpIcon className="text-white" size={24} />
                                </div>
                                <span className="text-xl font-bold gradient-text">StockSensePro</span>
                            </div>
                            <div className="hidden md:flex items-center gap-1">
                                {['overview', 'analysis', 'forecasts', 'gnn'].map((tab) => (
                                    <button
                                        key={tab}
                                        onClick={() => setActiveTab(tab)}
                                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === tab
                                            ? 'bg-info/20 text-info'
                                            : 'text-muted hover:text-foreground hover:bg-white/5'
                                            }`}
                                    >
                                        {tab === 'gnn' ? 'GNN Insights' : tab.charAt(0).toUpperCase() + tab.slice(1)}
                                    </button>
                                ))}
                            </div>
                        </div>
                        <div className="flex items-center gap-4">
                            <div className="relative hidden sm:block">
                                <SearchIcon className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" size={16} />
                                <input
                                    type="text"
                                    placeholder="Search forecasts..."
                                    className="bg-surface-elevated border border-white/10 rounded-full py-1.5 pl-9 pr-4 text-xs focus:ring-1 focus:ring-info outline-none transition-all w-40 focus:w-56"
                                />
                            </div>
                            <button className="p-2 hover:bg-white/5 rounded-lg transition-colors relative">
                                <BellIcon className="text-muted" size={20} />
                            </button>
                            <div className="flex items-center gap-2">
                                <div className="w-8 h-8 rounded-full bg-gradient-to-r from-cyan-500 to-blue-500 flex items-center justify-center text-xs font-bold text-white">
                                    {user.name ? user.name[0] : user.email[0].toUpperCase()}
                                </div>
                                <div className="hidden sm:block">
                                    <div className="text-sm font-medium">{user.name || user.email}</div>
                                    <div className="text-xs text-info flex items-center gap-1">
                                        <ChartIcon size={10} /> Analyst
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
                            Analytics <span className="gradient-text">Dashboard</span>
                        </h1>
                        <p className="text-muted text-sm flex items-center gap-2">
                            <ChartIcon size={14} className="text-info" />
                            Forecast Analysis & Model Insights
                            <span className="w-1 h-1 bg-white/20 rounded-full"></span>
                            <span>{new Date().toLocaleDateString('en-US', { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' })}</span>
                        </p>
                    </div>
                    <div className="flex items-center gap-3">
                        <Button variant="ghost" size="sm" onClick={fetchModelMetrics} disabled={metricsLoading}>
                            <RefreshIcon size={14} className={metricsLoading ? 'animate-spin' : ''} />
                            Refresh Data
                        </Button>
                        <div className="relative export-menu-container">
                            <Button 
                                variant="primary" 
                                size="sm" 
                                onClick={() => setShowExportMenu(!showExportMenu)}
                            >
                                <DatabaseIcon size={14} />
                                Export Report
                            </Button>
                            {showExportMenu && (
                                <div className="absolute right-0 mt-2 w-64 glass border border-white/10 rounded-lg shadow-xl z-50">
                                    <div className="p-2">
                                        <button
                                            onClick={handleExportReport}
                                            className="w-full text-left px-3 py-2 text-sm rounded-lg hover:bg-white/5 transition-colors"
                                        >
                                            <div className="font-medium">Full Analysis Report</div>
                                            <div className="text-xs text-muted">All metrics + volume stats</div>
                                        </button>
                                        <button
                                            onClick={handleExportModelPerformance}
                                            className="w-full text-left px-3 py-2 text-sm rounded-lg hover:bg-white/5 transition-colors"
                                        >
                                            <div className="font-medium">Model Performance</div>
                                            <div className="text-xs text-muted">MAE, MAPE, WAPE by SKU</div>
                                        </button>
                                        <button
                                            onClick={handleExportVolumeStats}
                                            className="w-full text-left px-3 py-2 text-sm rounded-lg hover:bg-white/5 transition-colors"
                                        >
                                            <div className="font-medium">Volume Statistics</div>
                                            <div className="text-xs text-muted">SKU demand patterns</div>
                                        </button>
                                    </div>
                                </div>
                            )}
                        </div>
                    </div>
                </div>

                {/* Analysis View Content */}
                {activeTab === 'analysis' && (
                    <div className="animate-in fade-in duration-300">
                        <AnalysisView />
                    </div>
                )}

                {/* GNN 3D View Content */}
                {activeTab === 'gnn' && (
                    <div className="animate-in fade-in duration-300">
                        <GNN3DVisualizer />
                    </div>
                )}

                {/* Main Dashboard Content (Only show if NOT analysis or gnn tab) */}
                {activeTab !== 'analysis' && activeTab !== 'gnn' && (
                    <>
                        {/* Model Performance Cards */}
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                            <Card glass className="group hover:border-info/30 transition-all">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-xs text-muted uppercase tracking-wider">Active Model</p>
                                        <h3 className="text-xl font-bold mt-1 text-info">
                                            {modelMetrics.find(m => m.status === 'active')?.model.split(' ')[0] || 'TFT'}
                                        </h3>
                                        <p className="text-xs text-muted mt-1">
                                            {modelMetrics.find(m => m.status === 'active')?.type.split(' ')[0] || 'Loading...'}
                                        </p>
                                    </div>
                                    <div className="w-10 h-10 bg-info/10 rounded-lg flex items-center justify-center text-info">
                                        <ActivityIcon size={20} />
                                    </div>
                                </div>
                            </Card>

                            <Card glass className="group hover:border-success/30 transition-all">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-xs text-muted uppercase tracking-wider">Avg. MAPE</p>
                                        <h3 className="text-2xl font-bold mt-1 text-success">
                                            {modelMetrics.find(m => m.status === 'active')?.mape.toFixed(1) || '8.5'}%
                                        </h3>
                                        <div className="flex items-center gap-1 mt-1">
                                            {metricsLoading ? (
                                                <span className="text-xs text-muted">Loading...</span>
                                            ) : (
                                                <>
                                                    <TrendingDownIcon size={12} className="text-success" />
                                                    <span className="text-xs text-success">Live Data</span>
                                                </>
                                            )}
                                        </div>
                                    </div>
                                    <div className="w-10 h-10 bg-success/10 rounded-lg flex items-center justify-center text-success">
                                        <CheckIcon size={20} />
                                    </div>
                                </div>
                            </Card>

                            <Card glass className="group hover:border-warning/30 transition-all">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-xs text-muted uppercase tracking-wider">Anomalies</p>
                                        <h3 className="text-2xl font-bold mt-1 text-warning">3</h3>
                                        <p className="text-xs text-muted mt-1">1 new today</p>
                                    </div>
                                    <div className="w-10 h-10 bg-warning/10 rounded-lg flex items-center justify-center text-warning">
                                        <AlertIcon size={20} />
                                    </div>
                                </div>
                            </Card>

                            <Card glass className="group hover:border-primary/30 transition-all">
                                <div className="flex items-center justify-between">
                                    <div>
                                        <p className="text-xs text-muted uppercase tracking-wider">GNN Nodes</p>
                                        <h3 className="text-2xl font-bold mt-1">248</h3>
                                        <p className="text-xs text-muted mt-1">1,234 edges</p>
                                    </div>
                                    <div className="w-10 h-10 bg-primary/10 rounded-lg flex items-center justify-center text-primary">
                                        <DatabaseIcon size={20} />
                                    </div>
                                </div>
                            </Card>
                        </div>

                        {/* Main Content Grid */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
                            {/* Model Comparison */}
                            <Card glass className="lg:col-span-2">
                                <CardHeader>
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <CardTitle className="flex items-center gap-2 text-lg">
                                                <ActivityIcon size={18} className="text-info" />
                                                Model Performance Comparison
                                            </CardTitle>
                                            <CardDescription>Compare accuracy metrics across models</CardDescription>
                                        </div>
                                    </div>
                                </CardHeader>
                                <CardContent>
                                    {metricsLoading ? (
                                        <div className="text-center py-8 text-muted">
                                            <RefreshIcon size={24} className="mx-auto mb-2 animate-spin" />
                                            Loading model metrics...
                                        </div>
                                    ) : (
                                        <Table>
                                            <TableHeader>
                                                <TableRow>
                                                    <TableHead>Model</TableHead>
                                                    <TableHead className="text-right">MAE</TableHead>
                                                    <TableHead className="text-right">MAPE</TableHead>
                                                    <TableHead className="text-right">WAPE</TableHead>
                                                    <TableHead className="text-right">Status</TableHead>
                                                </TableRow>
                                            </TableHeader>
                                            <TableBody>
                                                {modelMetrics.map((model) => (
                                                    <TableRow
                                                        key={model.model}
                                                        className={`hover:bg-white/5 cursor-pointer ${selectedModel === model.model ? 'bg-info/10' : ''}`}
                                                        onClick={() => setSelectedModel(model.model)}
                                                    >
                                                        <TableCell>
                                                            <div>
                                                                <div className="font-medium text-sm">{model.model}</div>
                                                                <div className="text-xs text-muted">{model.type}</div>
                                                            </div>
                                                        </TableCell>
                                                        <TableCell className="text-right font-mono">{model.mae.toFixed(2)}</TableCell>
                                                        <TableCell className="text-right font-mono">{model.mape.toFixed(1)}%</TableCell>
                                                        <TableCell className="text-right font-mono">{model.wape.toFixed(1)}%</TableCell>
                                                        <TableCell className="text-right">
                                                            {getStatusBadge(model.status)}
                                                        </TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    )}
                                </CardContent>
                            </Card>

                            {/* Anomalies Panel */}
                            <Card glass>
                                <CardHeader>
                                    <div className="flex items-center justify-between">
                                        <CardTitle className="flex items-center gap-2 text-lg">
                                            <AlertIcon size={18} className="text-warning" />
                                            Detected Anomalies
                                        </CardTitle>
                                    </div>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        {mockAnomalies.map((anomaly) => (
                                            <div
                                                key={anomaly.id}
                                                className="p-3 bg-white/5 rounded-lg border border-white/5 hover:border-warning/30 transition-all cursor-pointer"
                                            >
                                                <div className="flex items-start justify-between mb-2">
                                                    <span className="font-mono text-xs text-muted">{anomaly.sku}</span>
                                                    {getStatusBadge(anomaly.status)}
                                                </div>
                                                <p className="text-sm font-medium text-warning">{anomaly.type}</p>
                                                <p className="text-xs text-muted mt-1">{anomaly.description}</p>
                                                <p className="text-xs text-muted mt-2">{anomaly.date}</p>
                                            </div>
                                        ))}
                                    </div>
                                    <Button variant="ghost" className="w-full mt-4 text-xs">
                                        Flag New Anomaly
                                    </Button>
                                </CardContent>
                            </Card>
                        </div>

                        {/* Second Row */}
                        <div className="grid grid-cols-1 gap-8 mb-8">
                            {/* Product Accuracy - Full Width */}
                            <ProductAccuracyTable />
                        </div>

                        {/* Third Row */}
                        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8 mb-8">
                            {/* Accuracy Summary - Takes 1 column */}
                            <AccuracySummaryCard />

                            {/* Forecast Accuracy - Takes 2 columns */}
                            <div className="lg:col-span-2">
                                <Card glass>
                                    <CardHeader>
                                        <div className="flex items-center justify-between">
                                            <div>
                                                <CardTitle className="flex items-center gap-2 text-lg">
                                                    <ChartIcon size={18} className="text-success" />
                                                    Forecast Accuracy Review
                                                </CardTitle>
                                                <CardDescription>Compare predictions vs actuals</CardDescription>
                                            </div>
                                            <Button variant="ghost" size="sm">View All</Button>
                                        </div>
                                    </CardHeader>
                                    <CardContent>
                                        <Table>
                                            <TableHeader>
                                                <TableRow>
                                                    <TableHead>Product</TableHead>
                                                    <TableHead className="text-right">Predicted</TableHead>
                                                    <TableHead className="text-right">Actual</TableHead>
                                                    <TableHead className="text-right">Error %</TableHead>
                                                </TableRow>
                                            </TableHeader>
                                            <TableBody>
                                                {mockForecastAccuracy.map((item) => (
                                                    <TableRow key={item.sku} className="hover:bg-white/5">
                                                        <TableCell>
                                                            <div>
                                                                <div className="font-medium text-sm">{item.name}</div>
                                                                <div className="text-xs text-muted font-mono">{item.sku}</div>
                                                            </div>
                                                        </TableCell>
                                                        <TableCell className="text-right font-mono">{item.predicted}</TableCell>
                                                        <TableCell className="text-right font-mono">{item.actual}</TableCell>
                                                        <TableCell className="text-right">
                                                            <span className={`flex items-center justify-end gap-1 ${item.error > 5 ? 'text-warning' : 'text-success'}`}>
                                                                {item.trend === 'up' ? <TrendingUpIcon size={12} /> : <TrendingDownIcon size={12} />}
                                                                {item.error}%
                                                            </span>
                                                        </TableCell>
                                                    </TableRow>
                                                ))}
                                            </TableBody>
                                        </Table>
                                    </CardContent>
                                </Card>
                            </div>
                        </div>

                        {/* Fourth Row */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-8">
                            {/* What-If Simulation */}
                            <Card glass>
                                <CardHeader>
                                    <div className="flex items-center justify-between">
                                        <div>
                                            <CardTitle className="flex items-center gap-2 text-lg">
                                                <RefreshIcon size={18} className="text-primary" />
                                                What-If Simulation
                                            </CardTitle>
                                            <CardDescription>Test different demand scenarios</CardDescription>
                                        </div>
                                        <Button variant="primary" size="sm" onClick={runSimulation} disabled={simulationLoading}>
                                            <RefreshIcon size={14} className={simulationLoading ? 'animate-spin' : ''} />
                                            {simulationLoading ? 'Running...' : 'Run Simulation'}
                                        </Button>
                                    </div>
                                </CardHeader>
                                <CardContent>
                                    {/* Info message about chat */}
                                    <div className="mb-4 p-3 bg-info/10 border border-info/30 rounded-lg">
                                        <p className="text-sm text-info flex items-center gap-2">
                                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                                            </svg>
                                            Use the chat assistant (bottom right) to analyze custom scenarios!
                                        </p>
                                    </div>

                                    {/* Simulation Results */}
                                    <div className="space-y-3">
                                        {simulationResults.map((sim: any, idx) => (
                                            <div key={idx} className={`p-4 rounded-lg border ${sim.ai_reasoning ? 'bg-primary/5 border-primary/30' : 'bg-white/5 border-white/5'}`}>
                                                <div className="flex items-start justify-between mb-2">
                                                    <div className="flex-1">
                                                        <div className="flex items-center gap-2">
                                                            <div className="font-medium text-sm">{sim.scenario}</div>
                                                            {sim.ai_reasoning && (
                                                                <span className="px-2 py-0.5 bg-primary/20 text-primary text-xs rounded-full">AI</span>
                                                            )}
                                                        </div>
                                                        {sim.ai_reasoning && (
                                                            <div className="text-xs text-muted mt-1 italic">{sim.ai_reasoning}</div>
                                                        )}
                                                        <div className="text-xs text-muted mt-1">Confidence: {sim.confidence}%</div>
                                                    </div>
                                                    <div className="text-right ml-4">
                                                        <div className="font-bold text-lg">{sim.demand.toLocaleString()}</div>
                                                        <div className="mt-1">{getRiskBadge(sim.risk)}</div>
                                                    </div>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                </CardContent>
                            </Card>

                            {/* GNN Insights Preview */}
                            <Card glass>
                                <CardHeader>
                                    <CardTitle className="flex items-center gap-2 text-lg">
                                        <DatabaseIcon size={18} className="text-info" />
                                        GNN Insights Sample
                                    </CardTitle>
                                    <CardDescription>Product relationship preview</CardDescription>
                                </CardHeader>
                                <CardContent>
                                    <div className="space-y-3">
                                        {mockGNNInsights.slice(0, 2).map((insight, idx) => (
                                            <div key={idx} className="p-3 bg-white/5 rounded-lg border border-white/5">
                                                <div className="flex items-center justify-between mb-2">
                                                    <span className="font-mono font-bold text-primary text-sm">{insight.sku}</span>
                                                    <Badge variant="default">{insight.category}</Badge>
                                                </div>
                                                <p className="text-xs text-muted mb-2">Influenced by: {insight.influencedBy.join(', ')}</p>
                                                <div className="flex items-center justify-between text-sm">
                                                    <span className="text-muted">Strength</span>
                                                    <span className="font-bold text-info">{(insight.influenceStrength * 100).toFixed(0)}%</span>
                                                </div>
                                            </div>
                                        ))}
                                    </div>
                                    <Button variant="ghost" className="w-full mt-3 text-xs">View Full Graph</Button>
                                </CardContent>
                            </Card>
                        </div>

                        {/* GNN Insights */}
                        <Card glass>
                            <CardHeader>
                                <div className="flex items-center justify-between">
                                    <div>
                                        <CardTitle className="flex items-center gap-2 text-lg">
                                            <DatabaseIcon size={18} className="text-primary" />
                                            GNN Product Influence Graph
                                        </CardTitle>
                                        <CardDescription>Understand product relationships and cross-influences</CardDescription>
                                    </div>
                                    <Button variant="ghost" size="sm">View Full Graph</Button>
                                </div>
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                                    {mockGNNInsights.map((insight, idx) => (
                                        <div key={idx} className="p-4 bg-white/5 rounded-lg border border-white/5 hover:border-primary/30 transition-all">
                                            <div className="flex items-center justify-between mb-3">
                                                <span className="font-mono font-bold text-primary">{insight.sku}</span>
                                                <Badge variant="default">{insight.category}</Badge>
                                            </div>
                                            <p className="text-xs text-muted mb-2">Influenced by:</p>
                                            <div className="flex flex-wrap gap-2 mb-3">
                                                {insight.influencedBy.map((sku) => (
                                                    <span key={sku} className="px-2 py-1 bg-surface-elevated rounded text-xs font-mono">
                                                        {sku}
                                                    </span>
                                                ))}
                                            </div>
                                            <div className="flex items-center justify-between">
                                                <span className="text-xs text-muted">Influence Strength</span>
                                                <span className="text-sm font-bold text-info">{(insight.influenceStrength * 100).toFixed(0)}%</span>
                                            </div>
                                            {/* Influence bar */}
                                            <div className="mt-2 h-2 bg-surface-elevated rounded-full overflow-hidden">
                                                <div
                                                    className="h-full bg-gradient-to-r from-info to-primary rounded-full transition-all"
                                                    style={{ width: `${insight.influenceStrength * 100}%` }}
                                                ></div>
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>
                    </>
                )}

                {/* LLM Assistant Floating Button */}
                <div className="fixed bottom-6 right-6 z-40">
                    <button 
                        onClick={() => setIsChatOpen(!isChatOpen)}
                        className="w-14 h-14 bg-gradient-to-r from-cyan-600 to-blue-600 rounded-full shadow-lg shadow-info/30 flex items-center justify-center hover:scale-110 transition-transform group"
                    >
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                        </svg>
                    </button>
                    <div className="absolute -top-10 right-0 bg-surface-elevated text-xs px-3 py-1.5 rounded-lg shadow-lg opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap border border-white/10">
                        💬 Ask AI Assistant
                    </div>
                </div>
            </div>

            {/* Chat Panel */}
            <ChatPanel 
                isOpen={isChatOpen} 
                onClose={() => setIsChatOpen(false)}
                onScenarioAnalyzed={handleChatScenario}
            />

            {/* Toast Notification */}
            {notification && (
                <Toast
                    message={notification.message}
                    type={notification.type}
                    onClose={() => setNotification(null)}
                />
            )}
        </div >
    );
}
