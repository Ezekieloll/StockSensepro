'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import Card, { CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import Input from '@/components/ui/Input';
import Toast from '@/components/ui/Toast';
import ConfirmDialog from '@/components/ui/ConfirmDialog';
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from '@/components/ui/Table';
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
} from '@/components/ui/Icons';

interface User {
    id?: number;
    name?: string;
    email: string;
    role: string;
}

interface ModelMetrics {
    mae: number;
    mape: number;
    wape: number;
}

interface GraphStats {
    num_nodes: number;
    num_edges: number;
}

interface HighRiskSKU {
    sku: string;
    store_id: string;
    risk_score: number;
    predicted_demand: number;
    current_inventory: number;
    days_of_cover: number;
    stockout: boolean;
}

interface PurchaseOrder {
    id: number;
    po_number: string;
    store_id: string;
    status: string;
    total_items: number;
    total_quantity: number;
    total_amount: number | null;
    created_at: string;
    created_by?: { name: string; email: string };
    expected_delivery_date?: string;
    actual_delivery_date?: string;
    notes?: string;
    items?: PurchaseOrderItem[];
}

interface PurchaseOrderItem {
    id: number;
    sku: string;
    product_category: string | null;
    quantity_requested: number;
    quantity_delivered: number;
    unit_price: number | null;
    line_total: number | null;
}

interface StagingUpload {
    id: number;
    filename: string;
    uploaded_by: string;
    uploaded_at: string;
    status: string;
    row_count: number;
    valid_rows: number;
    invalid_rows: number;
    date_range: {
        min: string | null;
        max: string | null;
    };
    error_message: string | null;
}

// Mock data for demo (features not yet implemented)

const mockPipelineStatus = [
    { name: 'Forecasting', status: 'completed', lastRun: '2 hours ago', duration: '4m 32s' },
    { name: 'GNN Build', status: 'completed', lastRun: '1 day ago', duration: '1m 15s' },
    { name: 'Adversarial Test', status: 'completed', lastRun: '2 hours ago', duration: '2m 45s' },
    { name: 'Data ETL', status: 'idle', lastRun: '3 hours ago', duration: '45s' },
];

const mockAuditLogs = [
    { time: '14:30:22', user: 'admin', action: 'Triggered forecast pipeline' },
    { time: '14:28:15', user: 'admin', action: 'Approved data upload (trans_jan)' },
    { time: '14:15:00', user: 'system', action: 'Scheduled adversarial test completed' },
    { time: '13:45:30', user: 'jane@...', action: 'Created purchase order #1234' },
];

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function AdminDashboard() {
    const router = useRouter();
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);
    const [activeTab, setActiveTab] = useState('overview');
    
    // Real data states
    const [modelMetrics, setModelMetrics] = useState<ModelMetrics | null>(null);
    const [graphStats, setGraphStats] = useState<GraphStats | null>(null);
    const [highRiskSKUs, setHighRiskSKUs] = useState<HighRiskSKU[]>([]);
    const [totalSKUs, setTotalSKUs] = useState<number>(0);
    const [users, setUsers] = useState<User[]>([]);
    const [showUserModal, setShowUserModal] = useState(false);
    const [editingUser, setEditingUser] = useState<User | null>(null);
    const [userForm, setUserForm] = useState({ name: '', email: '', password: '', role: 'analyst' });
    
    // Purchase Orders
    const [purchaseOrders, setPurchaseOrders] = useState<PurchaseOrder[]>([]);
    const [selectedPO, setSelectedPO] = useState<PurchaseOrder | null>(null);
    const [showDeliverModal, setShowDeliverModal] = useState(false);
    const [deliveryDate, setDeliveryDate] = useState(new Date().toISOString().split('T')[0]);
    
    // CSV Upload
    const [stagingUploads, setStagingUploads] = useState<StagingUpload[]>([]);
    const [uploadingCSV, setUploadingCSV] = useState(false);
    const [selectedFile, setSelectedFile] = useState<File | null>(null);
    
    // Adversarial Testing
    const [runningTest, setRunningTest] = useState(false);
    
    // Notifications
    const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' | 'warning' | 'info' } | null>(null);
    const [confirmDialog, setConfirmDialog] = useState<{ title: string; message: string; onConfirm: () => void; type?: 'danger' | 'warning' | 'info' } | null>(null);

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

    // Fetch real data from backend
    useEffect(() => {
        const fetchAdminData = async () => {
            try {
                // Fetch model metrics
                const metricsRes = await fetch(`${API_BASE}/analytics/model-metrics`);
                if (metricsRes.ok) {
                    const data = await metricsRes.json();
                    setModelMetrics({
                        mae: data.mae,
                        mape: data.mape,
                        wape: data.wape
                    });
                }

                // Fetch GNN graph statistics
                const graphRes = await fetch(`${API_BASE}/gnn/graph-statistics`);
                if (graphRes.ok) {
                    const data = await graphRes.json();
                    setGraphStats({
                        num_nodes: data.num_nodes,
                        num_edges: data.num_edges
                    });
                    setTotalSKUs(data.num_nodes);
                }

                // Fetch high-risk SKUs from adversarial testing
                const riskRes = await fetch(`${API_BASE}/adversarial/?high_risk_only=true`);
                if (riskRes.ok) {
                    const data = await riskRes.json();
                    setHighRiskSKUs(data.slice(0, 5)); // Top 5 high-risk SKUs
                }

                // Fetch users
                const usersRes = await fetch(`${API_BASE}/api/users/`);
                if (usersRes.ok) {
                    const data = await usersRes.json();
                    console.log('Users fetched:', data);
                    setUsers(data);
                } else {
                    console.error('Failed to fetch users:', usersRes.status);
                }
                
                // Fetch purchase orders
                const posRes = await fetch(`${API_BASE}/api/purchase-orders/`);
                if (posRes.ok) {
                    const data = await posRes.json();
                    setPurchaseOrders(data);
                }
                
                // Fetch staging uploads
                const stagingRes = await fetch(`${API_BASE}/csv-upload/staging`);
                if (stagingRes.ok) {
                    const data = await stagingRes.json();
                    setStagingUploads(data);
                }
            } catch (error) {
                console.error('Failed to fetch admin data:', error);
            }
        };

        if (!loading) {
            fetchAdminData();
        }
    }, [loading]);

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

    const openUserModal = (user?: User) => {
        if (user) {
            setEditingUser(user);
            setUserForm({ name: user.name || '', email: user.email, password: '', role: user.role });
        } else {
            setEditingUser(null);
            setUserForm({ name: '', email: '', password: '', role: 'analyst' });
        }
        setShowUserModal(true);
    };

    const closeUserModal = () => {
        setShowUserModal(false);
        setEditingUser(null);
        setUserForm({ name: '', email: '', password: '', role: 'analyst' });
    };

    const handleSaveUser = async () => {
        try {
            if (editingUser) {
                // Update existing user
                const updateData: any = { name: userForm.name, email: userForm.email, role: userForm.role };
                if (userForm.password) {
                    updateData.password = userForm.password;
                }
                const res = await fetch(`${API_BASE}/api/users/${editingUser.id}`, {
                    method: 'PUT',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(updateData),
                });
                if (res.ok) {
                    const updated = await res.json();
                    setUsers(users.map(u => u.id === updated.id ? updated : u));
                    closeUserModal();
                }
            } else {
                // Create new user
                const res = await fetch(`${API_BASE}/api/users/`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(userForm),
                });
                if (res.ok) {
                    const newUser = await res.json();
                    setUsers([...users, newUser]);
                    closeUserModal();
                } else {
                    const error = await res.json();
                    alert(error.detail || 'Failed to create user');
                }
            }
        } catch (error) {
            console.error('Failed to save user:', error);
            alert('Failed to save user');
        }
    };

    const handleDeleteUser = async (userId: number) => {
        if (!confirm('Are you sure you want to delete this user?')) return;
        
        try {
            const res = await fetch(`${API_BASE}/api/users/${userId}`, { method: 'DELETE' });
            if (res.ok) {
                setUsers(users.filter(u => u.id !== userId));
            } else {
                const error = await res.json();
                alert(error.detail || 'Failed to delete user');
            }
        } catch (error) {
            console.error('Failed to delete user:', error);
            alert('Failed to delete user');
        }
    };

    const handleApprovePO = async (poId: number) => {
        try {
            const res = await fetch(`${API_BASE}/api/purchase-orders/${poId}/status`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ status: 'approved' }),
            });
            if (res.ok) {
                const updated = await res.json();
                setPurchaseOrders(purchaseOrders.map(po => po.id === updated.id ? updated : po));
            }
        } catch (error) {
            console.error('Failed to approve PO:', error);
        }
    };

    const handleDeliverPO = async () => {
        if (!selectedPO) return;
        
        try {
            const items = selectedPO.items?.map(item => ({
                item_id: item.id,
                quantity_delivered: item.quantity_requested
            })) || [];
            
            const res = await fetch(`${API_BASE}/api/purchase-orders/${selectedPO.id}/deliver`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    actual_delivery_date: deliveryDate,
                    items: items
                }),
            });
            
            if (res.ok) {
                const updated = await res.json();
                setPurchaseOrders(purchaseOrders.map(po => po.id === updated.id ? updated : po));
                setShowDeliverModal(false);
                setSelectedPO(null);
            }
        } catch (error) {
            console.error('Failed to deliver PO:', error);
        }
    };

    const getStatusColor = (status: string) => {
        switch (status) {
            case 'pending': return 'warning';
            case 'approved': return 'info';
            case 'delivered': return 'success';
            case 'cancelled': return 'error';
            default: return 'default';
        }
    };

    const handleRunAdversarialTest = async () => {
        setRunningTest(true);
        try {
            // Call backend endpoint to trigger adversarial testing
            const res = await fetch(`${API_BASE}/adversarial/run-test`, {
                method: 'POST',
            });
            
            if (res.ok) {
                const result = await res.json();
                alert(`✅ ${result.message}\n\nAdversarial testing completed successfully!`);
                
                // Refresh high-risk SKUs
                const riskRes = await fetch(`${API_BASE}/adversarial/?high_risk_only=true`);
                if (riskRes.ok) {
                    const data = await riskRes.json();
                    setHighRiskSKUs(data.slice(0, 5));
                }
            } else {
                const error = await res.json();
                alert(`❌ Test failed: ${error.detail}`);
            }
        } catch (error) {
            console.error('Failed to run adversarial test:', error);
            alert('Failed to trigger adversarial test. Check console for details.');
        } finally {
            setRunningTest(false);
        }
    };

    const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setSelectedFile(file);
        }
    };

    const handleUploadCSV = async () => {
        if (!selectedFile) {
            setToast({ message: 'Please select a CSV file first', type: 'warning' });
            return;
        }

        setUploadingCSV(true);
        try {
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('uploaded_by', user?.email || 'admin@stocksense.com');

            const res = await fetch(`${API_BASE}/csv-upload/upload`, {
                method: 'POST',
                body: formData,
            });

            if (res.ok) {
                const result = await res.json();
                setToast({ 
                    message: `Upload successful!\n\nFile: ${result.filename}\nRows: ${result.row_count}\nValid: ${result.valid_rows}\nInvalid: ${result.invalid_rows}`, 
                    type: 'success' 
                });
                
                // Refresh staging queue
                const stagingRes = await fetch(`${API_BASE}/csv-upload/staging`);
                if (stagingRes.ok) {
                    const data = await stagingRes.json();
                    setStagingUploads(data);
                }
                
                setSelectedFile(null);
                // Reset file input
                const fileInput = document.getElementById('csv-file-input') as HTMLInputElement;
                if (fileInput) fileInput.value = '';
            } else {
                const error = await res.json();
                setToast({ message: `Upload failed: ${error.detail}`, type: 'error' });
            }
        } catch (error) {
            console.error('Failed to upload CSV:', error);
            setToast({ message: 'Failed to upload CSV. Please try again.', type: 'error' });
        } finally {
            setUploadingCSV(false);
        }
    };

    const handleApproveUpload = async (uploadId: number) => {
        setConfirmDialog({
            title: 'Approve Upload',
            message: 'Are you sure you want to approve this upload and import to database?',
            type: 'info',
            onConfirm: async () => {
                setConfirmDialog(null);
                try {
                    const res = await fetch(`${API_BASE}/csv-upload/staging/${uploadId}/approve`, {
                        method: 'POST',
                    });
                    
                    if (res.ok) {
                        const result = await res.json();
                        setToast({ 
                            message: `Import successful!\n\nRows imported: ${result.rows_imported}\nDaily demand updated: ${result.daily_demand_updated}`, 
                            type: 'success' 
                        });
                        
                        // Refresh staging queue
                        const stagingRes = await fetch(`${API_BASE}/csv-upload/staging`);
                        if (stagingRes.ok) {
                            const data = await stagingRes.json();
                            setStagingUploads(data);
                        }
                    } else {
                        const error = await res.json();
                        setToast({ message: `Approval failed: ${error.detail}`, type: 'error' });
                    }
                } catch (error) {
                    console.error('Failed to approve upload:', error);
                    setToast({ message: 'Failed to approve upload. Please try again.', type: 'error' });
                }
            }
        });
    };

    const handleRejectUpload = async (uploadId: number) => {
        setConfirmDialog({
            title: 'Reject Upload',
            message: 'Are you sure you want to reject and delete this upload? This action cannot be undone.',
            type: 'danger',
            onConfirm: async () => {
                setConfirmDialog(null);
                try {
                    const res = await fetch(`${API_BASE}/csv-upload/staging/${uploadId}/reject`, {
                        method: 'POST',
                    });
                    
                    if (res.ok) {
                        setToast({ message: 'Upload rejected and deleted', type: 'success' });
                        
                        // Refresh staging queue
                        const stagingRes = await fetch(`${API_BASE}/csv-upload/staging`);
                        if (stagingRes.ok) {
                            const data = await stagingRes.json();
                            setStagingUploads(data);
                        }
                    } else {
                        const error = await res.json();
                        setToast({ message: `Rejection failed: ${error.detail}`, type: 'error' });
                    }
                } catch (error) {
                    console.error('Failed to reject upload:', error);
                    setToast({ message: 'Failed to reject upload. Please try again.', type: 'error' });
                }
            }
        });
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
                                {['overview', 'purchase-orders', 'ai-scenarios', 'scenario-chat', 'data', 'ml', 'testing', 'users', 'logs'].map((tab) => (
                                    <button
                                        key={tab}
                                        onClick={() => {
                                            if (tab === 'ai-scenarios') {
                                                router.push('/admin/ai-scenarios');
                                            } else if (tab === 'scenario-chat') {
                                                router.push('/admin/scenario-chat');
                                            } else {
                                                setActiveTab(tab);
                                            }
                                        }}
                                        className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${activeTab === tab
                                            ? 'bg-primary/20 text-primary'
                                            : 'text-muted hover:text-foreground hover:bg-white/5'
                                            }`}
                                    >
                                        {tab === 'purchase-orders' ? 'Purchase Orders' : 
                                         tab === 'ai-scenarios' ? '🤖 AI Scenarios' :
                                         tab === 'scenario-chat' ? '💬 AI Chat' :
                                         tab.charAt(0).toUpperCase() + tab.slice(1)}
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
                                <h3 className="text-2xl font-bold mt-1">{totalSKUs || 240}</h3>
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
                                <h3 className="text-2xl font-bold mt-1 text-error">{highRiskSKUs.length}</h3>
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

                {/* Purchase Orders Section */}
                {activeTab === 'purchase-orders' && (
                    <div className="space-y-6">
                        <Card glass>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <CheckIcon size={18} className="text-primary" />
                                    Purchase Order Management
                                </CardTitle>
                                <CardDescription>Approve and deliver pending purchase orders</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead>PO Number</TableHead>
                                            <TableHead>Store</TableHead>
                                            <TableHead>Items</TableHead>
                                            <TableHead>Quantity</TableHead>
                                            <TableHead>Amount</TableHead>
                                            <TableHead>Status</TableHead>
                                            <TableHead>Created</TableHead>
                                            <TableHead className="text-right">Actions</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {purchaseOrders.length > 0 ? (
                                            purchaseOrders.map((po) => (
                                                <TableRow key={po.id}>
                                                    <TableCell className="font-mono text-sm font-medium">{po.po_number}</TableCell>
                                                    <TableCell>
                                                        <Badge variant="default">{po.store_id}</Badge>
                                                    </TableCell>
                                                    <TableCell>{po.total_items}</TableCell>
                                                    <TableCell>{po.total_quantity}</TableCell>
                                                    <TableCell>
                                                        {po.total_amount ? `$${Number(po.total_amount).toFixed(2)}` : '-'}
                                                    </TableCell>
                                                    <TableCell>
                                                        <Badge variant={getStatusColor(po.status)}>{po.status}</Badge>
                                                    </TableCell>
                                                    <TableCell className="text-sm text-muted">
                                                        {new Date(po.created_at).toLocaleDateString()}
                                                    </TableCell>
                                                    <TableCell className="text-right">
                                                        <div className="flex gap-2 justify-end">
                                                            {po.status === 'pending' && (
                                                                <Button 
                                                                    variant="ghost" 
                                                                    size="sm" 
                                                                    onClick={() => handleApprovePO(po.id)}
                                                                >
                                                                    Approve
                                                                </Button>
                                                            )}
                                                            {po.status === 'approved' && (
                                                                <Button 
                                                                    variant="primary" 
                                                                    size="sm"
                                                                    onClick={async () => {
                                                                        // Fetch full PO details with items
                                                                        const res = await fetch(`${API_BASE}/api/purchase-orders/${po.id}`);
                                                                        if (res.ok) {
                                                                            const fullPO = await res.json();
                                                                            setSelectedPO(fullPO);
                                                                            setShowDeliverModal(true);
                                                                        }
                                                                    }}
                                                                >
                                                                    Deliver
                                                                </Button>
                                                            )}
                                                            {po.status === 'delivered' && (
                                                                <span className="text-xs text-success">✓ Delivered</span>
                                                            )}
                                                        </div>
                                                    </TableCell>
                                                </TableRow>
                                            ))
                                        ) : (
                                            <TableRow>
                                                <TableCell colSpan={8} className="text-center py-8 text-muted">
                                                    No purchase orders found
                                                </TableCell>
                                            </TableRow>
                                        )}
                                    </TableBody>
                                </Table>
                            </CardContent>
                        </Card>
                    </div>
                )}

                {/* Main Content Grid */}
                {activeTab === 'overview' && (
                    <>
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
                                <div className="flex gap-2">
                                    <input
                                        id="csv-file-input"
                                        type="file"
                                        accept=".csv"
                                        onChange={handleFileSelect}
                                        className="hidden"
                                    />
                                    <Button 
                                        variant="secondary" 
                                        size="sm"
                                        onClick={() => document.getElementById('csv-file-input')?.click()}
                                    >
                                        <DatabaseIcon size={14} />
                                        Choose File
                                    </Button>
                                    {selectedFile && (
                                        <Button 
                                            variant="primary" 
                                            size="sm"
                                            onClick={handleUploadCSV}
                                            disabled={uploadingCSV}
                                        >
                                            <UploadIcon size={14} />
                                            {uploadingCSV ? 'Uploading...' : `Upload ${selectedFile.name}`}
                                        </Button>
                                    )}
                                </div>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-3">
                                <p className="text-xs text-muted uppercase tracking-wider mb-2">Staging Queue</p>
                                {stagingUploads.length > 0 ? (
                                    stagingUploads.map((item) => (
                                        <div key={item.id} className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/5">
                                            <div className="flex items-center gap-3">
                                                <div className="w-8 h-8 bg-surface-elevated rounded flex items-center justify-center">
                                                    <DatabaseIcon size={14} className="text-muted" />
                                                </div>
                                                <div>
                                                    <div className="text-sm font-medium">{item.filename}</div>
                                                    <div className="text-xs text-muted">
                                                        {item.row_count.toLocaleString()} rows ({item.valid_rows} valid, {item.invalid_rows} invalid)
                                                    </div>
                                                    {item.date_range.min && (
                                                        <div className="text-xs text-muted mt-1">
                                                            Range: {item.date_range.min} to {item.date_range.max}
                                                        </div>
                                                    )}
                                                    {item.error_message && (
                                                        <div className="text-xs text-error mt-1">{item.error_message.split('\n')[0]}</div>
                                                    )}
                                                </div>
                                            </div>
                                            <div className="flex items-center gap-2">
                                                {getStatusBadge(item.status)}
                                                {item.status === 'pending' && item.invalid_rows === 0 && (
                                                    <>
                                                        <Button 
                                                            variant="primary" 
                                                            size="sm"
                                                            onClick={() => handleApproveUpload(item.id)}
                                                        >
                                                            Approve
                                                        </Button>
                                                        <Button 
                                                            variant="ghost" 
                                                            size="sm"
                                                            onClick={() => handleRejectUpload(item.id)}
                                                        >
                                                            Reject
                                                        </Button>
                                                    </>
                                                )}
                                                {item.status === 'error' && (
                                                    <Button 
                                                        variant="ghost" 
                                                        size="sm"
                                                        onClick={() => handleRejectUpload(item.id)}
                                                    >
                                                        Delete
                                                    </Button>
                                                )}
                                            </div>
                                        </div>
                                    ))
                                ) : (
                                    <div className="text-center py-8 text-muted text-sm">
                                        No uploads in staging queue
                                    </div>
                                )}
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
                                    <p className="text-lg font-bold text-primary">TFT+GNN v2.1</p>
                                    {modelMetrics && modelMetrics.mae != null && modelMetrics.mape != null && (
                                        <p className="text-xs text-muted mt-1">
                                            MAE: {modelMetrics.mae.toFixed(2)} | MAPE: {modelMetrics.mape.toFixed(1)}%
                                        </p>
                                    )}
                                </div>
                                <div className="p-4 bg-white/5 rounded-lg border border-white/5">
                                    <p className="text-xs text-muted uppercase tracking-wider mb-2">GNN Graph</p>
                                    <p className="text-lg font-bold">{graphStats?.num_nodes || 240} nodes</p>
                                    <p className="text-xs text-muted mt-1">{graphStats?.num_edges?.toLocaleString() || '14,578'} edges</p>
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
                                <Button 
                                    variant="secondary" 
                                    size="sm"
                                    onClick={handleRunAdversarialTest}
                                    disabled={runningTest}
                                >
                                    {runningTest ? 'Refreshing...' : 'Run Test'}
                                </Button>
                            </div>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-3 gap-3 mb-4">
                                <div className="text-center p-3 bg-error/10 rounded-lg">
                                    <p className="text-2xl font-bold text-error">{highRiskSKUs.length}</p>
                                    <p className="text-xs text-muted">High Risk</p>
                                </div>
                                <div className="text-center p-3 bg-warning/10 rounded-lg">
                                    <p className="text-2xl font-bold text-warning">-</p>
                                    <p className="text-xs text-muted">Medium</p>
                                </div>
                                <div className="text-center p-3 bg-success/10 rounded-lg">
                                    <p className="text-2xl font-bold text-success">-</p>
                                    <p className="text-xs text-muted">Low Risk</p>
                                </div>
                            </div>
                            <p className="text-xs text-muted uppercase tracking-wider mb-2">High Risk SKUs</p>
                            <div className="space-y-2">
                                {highRiskSKUs.length > 0 ? (
                                    highRiskSKUs.map((item, idx) => (
                                        <div key={idx} className="flex items-center justify-between p-2 bg-white/5 rounded-lg">
                                            <div className="flex items-center gap-2">
                                                <span className="font-mono text-sm">{item.sku}</span>
                                                <span className="text-xs text-muted">{item.store_id}</span>
                                            </div>
                                            <div className="flex items-center gap-3">
                                                <span className={`text-sm font-bold ${getRiskColor(item.risk_score)}`}>
                                                    {(item.risk_score * 100).toFixed(0)}%
                                                </span>
                                                <span className="text-xs text-muted">{item.days_of_cover.toFixed(1)}d cover</span>
                                            </div>
                                        </div>
                                    ))
                                ) : (
                                    <p className="text-sm text-muted text-center py-4">Loading high-risk SKUs...</p>
                                )}
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
                                <Button variant="primary" size="sm" onClick={() => openUserModal()}>
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
                                    {users.length > 0 ? (
                                        users.map((u) => (
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
                                                    <div className="flex gap-2 justify-end">
                                                        <Button variant="ghost" size="sm" onClick={() => openUserModal(u)}>Edit</Button>
                                                        <Button variant="ghost" size="sm" onClick={() => handleDeleteUser(u.id!)}>Delete</Button>
                                                    </div>
                                                </TableCell>
                                            </TableRow>
                                        ))
                                    ) : (
                                        <TableRow>
                                            <TableCell colSpan={3} className="text-center py-8 text-muted">
                                                {loading ? 'Loading users...' : 'No users found. Click "Add User" to create one.'}
                                            </TableCell>
                                        </TableRow>
                                    )}
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
                    </>
            )}
            </div>

            {/* User Modal */}
            {showUserModal && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={closeUserModal}>
                    <div className="bg-surface border border-white/10 rounded-xl p-6 w-full max-w-md" onClick={e => e.stopPropagation()}>
                        <h3 className="text-xl font-bold mb-4">{editingUser ? 'Edit User' : 'Create New User'}</h3>
                        
                        <div className="space-y-4">
                            <div>
                                <label className="text-sm text-muted mb-1 block">Name</label>
                                <Input
                                    value={userForm.name}
                                    onChange={(e) => setUserForm({ ...userForm, name: e.target.value })}
                                    placeholder="John Doe"
                                />
                            </div>
                            
                            <div>
                                <label className="text-sm text-muted mb-1 block">Email</label>
                                <Input
                                    type="email"
                                    value={userForm.email}
                                    onChange={(e) => setUserForm({ ...userForm, email: e.target.value })}
                                    placeholder="john@company.com"
                                />
                            </div>
                            
                            <div>
                                <label className="text-sm text-muted mb-1 block">
                                    Password {editingUser && '(leave blank to keep current)'}
                                </label>
                                <Input
                                    type="password"
                                    value={userForm.password}
                                    onChange={(e) => setUserForm({ ...userForm, password: e.target.value })}
                                    placeholder={editingUser ? 'Enter new password' : 'Password'}
                                />
                            </div>
                            
                            <div>
                                <label className="text-sm text-muted mb-1 block">Role</label>
                                <select
                                    value={userForm.role}
                                    onChange={(e) => setUserForm({ ...userForm, role: e.target.value })}
                                    className="w-full px-3 py-2 bg-white/5 border border-white/10 rounded-lg text-foreground"
                                >
                                    <option value="analyst">Analyst</option>
                                    <option value="manager">Manager</option>
                                    <option value="admin">Admin</option>
                                </select>
                            </div>
                        </div>
                        
                        <div className="flex gap-3 mt-6">
                            <Button variant="secondary" onClick={closeUserModal} className="flex-1">
                                Cancel
                            </Button>
                            <Button variant="primary" onClick={handleSaveUser} className="flex-1">
                                {editingUser ? 'Update' : 'Create'}
                            </Button>
                        </div>
                    </div>
                </div>
            )}
            
            {/* Deliver PO Modal */}
            {showDeliverModal && selectedPO && (
                <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50" onClick={() => setShowDeliverModal(false)}>
                    <div className="bg-surface border border-white/10 rounded-xl p-6 w-full max-w-2xl max-h-[80vh] overflow-y-auto" onClick={e => e.stopPropagation()}>
                        <h3 className="text-xl font-bold mb-4">Deliver Purchase Order</h3>
                        
                        <div className="bg-white/5 rounded-lg p-4 mb-4">
                            <div className="grid grid-cols-2 gap-4 text-sm">
                                <div>
                                    <span className="text-muted">PO Number:</span>
                                    <span className="ml-2 font-mono font-bold">{selectedPO.po_number}</span>
                                </div>
                                <div>
                                    <span className="text-muted">Store:</span>
                                    <Badge variant="default" className="ml-2">{selectedPO.store_id}</Badge>
                                </div>
                                <div>
                                    <span className="text-muted">Total Items:</span>
                                    <span className="ml-2 font-medium">{selectedPO.total_items}</span>
                                </div>
                                <div>
                                    <span className="text-muted">Total Quantity:</span>
                                    <span className="ml-2 font-medium">{selectedPO.total_quantity}</span>
                                </div>
                            </div>
                        </div>
                        
                        <div className="mb-4">
                            <label className="text-sm text-muted mb-2 block">Delivery Date</label>
                            <Input
                                type="date"
                                value={deliveryDate}
                                onChange={(e) => setDeliveryDate(e.target.value)}
                            />
                        </div>
                        
                        <div className="mb-4">
                            <p className="text-sm text-muted mb-2">Items to Deliver:</p>
                            <div className="space-y-2 max-h-60 overflow-y-auto">
                                {selectedPO.items?.map((item, idx) => (
                                    <div key={idx} className="flex items-center justify-between p-3 bg-white/5 rounded-lg">
                                        <div>
                                            <span className="font-mono text-sm font-medium">{item.sku}</span>
                                            {item.product_category && (
                                                <span className="ml-2 text-xs text-muted">({item.product_category})</span>
                                            )}
                                        </div>
                                        <div className="flex items-center gap-4">
                                            <span className="text-sm">
                                                Qty: <span className="font-bold">{item.quantity_requested}</span>
                                            </span>
                                            {item.unit_price && (
                                                <span className="text-sm text-muted">
                                                    @ ${Number(item.unit_price).toFixed(2)}
                                                </span>
                                            )}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                        
                        <div className="bg-warning/10 border border-warning/20 rounded-lg p-3 mb-4">
                            <p className="text-sm text-warning">
                                ⚠️ This will update inventory levels and create transaction records for all items.
                            </p>
                        </div>
                        
                        <div className="flex gap-3">
                            <Button variant="secondary" onClick={() => setShowDeliverModal(false)} className="flex-1">
                                Cancel
                            </Button>
                            <Button variant="primary" onClick={handleDeliverPO} className="flex-1">
                                Confirm Delivery
                            </Button>
                        </div>
                    </div>
                </div>
            )}
            
            {/* Toast Notification */}
            {toast && (
                <Toast
                    message={toast.message}
                    type={toast.type}
                    onClose={() => setToast(null)}
                />
            )}
            
            {/* Confirm Dialog */}
            {confirmDialog && (
                <ConfirmDialog
                    title={confirmDialog.title}
                    message={confirmDialog.message}
                    type={confirmDialog.type}
                    onConfirm={confirmDialog.onConfirm}
                    onCancel={() => setConfirmDialog(null)}
                />
            )}
        </div>
    );
}
