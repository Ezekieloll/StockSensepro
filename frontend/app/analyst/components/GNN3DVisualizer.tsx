'use client';

import { useEffect, useState, useRef } from 'react';
import dynamic from 'next/dynamic';
import Card, { CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import Button from '@/components/ui/Button';
import Badge from '@/components/ui/Badge';
import { RefreshIcon, DatabaseIcon, SearchIcon } from '@/components/ui/Icons';

// Dynamically import ForceGraph3D to avoid SSR issues
const ForceGraph3D = dynamic(() => import('react-force-graph-3d'), { ssr: false });

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface GraphNode {
    id: string;
    index: number;
    category: string;
    strength: number;
    label: string;
}

interface GraphEdge {
    source: string;
    target: string;
    weight: number;
}

interface GraphData {
    nodes: GraphNode[];
    edges: GraphEdge[];
    total_nodes: number;
    total_edges: number;
}

interface GraphStats {
    nodes: number;
    edges: number;
    density: number;
    degree_stats: {
        average: number;
        min: number;
        max: number;
    };
    edge_weight_stats: {
        average: number;
        min: number;
        max: number;
    };
}

interface ProductInfluence {
    sku: string;
    category: string;
    influence_weight: number;
    influence_strength: number;
}

interface ProductCatalog {
    [sku: string]: {
        category: string;
        name: string;
    };
}

// Category color mapping
const CATEGORY_COLORS: Record<string, string> = {
    'GROC': '#4ade80', 'FRPR': '#22c55e', 'BEVG': '#60a5fa', 'BKDY': '#f59e0b',
    'FRZN': '#06b6d4', 'SNCK': '#fbbf24', 'MEAT': '#ef4444', 'PRSN': '#ec4899',
    'BABC': '#a78bfa', 'CLOT': '#8b5cf6', 'FTRW': '#14b8a6', 'JWCH': '#fde047',
    'BAGL': '#fb923c', 'ELEC': '#3b82f6', 'STOF': '#6366f1', 'FURH': '#a855f7',
    'BEDM': '#d946ef', 'CLNS': '#10b981', 'KICH': '#f97316', 'PETC': '#84cc16',
    'SPRT': '#0ea5e9', 'TOYG': '#f43f5e', 'AUTO': '#64748b', 'BOOK': '#94a3b8',
};

// Category full names
const CATEGORY_NAMES: Record<string, string> = {
    'AUTO': 'Automotive & Accessories',
    'BABC': 'Baby Care & Products',
    'BAGL': 'Bags & Luggage',
    'BEDM': 'Bedding & Mattresses',
    'BEVG': 'Beverages',
    'BKDY': 'Bakery & Pastries',
    'BOOK': 'Books & Media',
    'CLNS': 'Cleaning Supplies',
    'CLOT': 'Clothing & Apparel',
    'ELEC': 'Electronics',
    'FRPR': 'Fresh Produce',
    'FRZN': 'Frozen Foods',
    'FURH': 'Furniture & Home',
    'FTRW': 'Footwear',
    'GROC': 'Groceries & Staples',
    'JWCH': 'Jewelry & Watches',
    'KICH': 'Kitchen & Dining',
    'MEAT': 'Meat & Seafood',
    'PETC': 'Pet Care & Supplies',
    'PRSN': 'Personal Care',
    'SNCK': 'Snacks & Confectionery',
    'SPRT': 'Sports & Outdoors',
    'STOF': 'Stationery & Office',
    'TOYG': 'Toys & Games',
};

export default function GNN3DVisualizer() {
    const fgRef = useRef<unknown>(null);
    const [view, setView] = useState<'3d' | 'explorer' | 'reference'>('explorer');
    const [stats, setStats] = useState<GraphStats | null>(null);
    const [loading, setLoading] = useState(false);
    const [selectedSKU, setSelectedSKU] = useState('');
    const [influences, setInfluences] = useState<ProductInfluence[]>([]);
    const [graphData, setGraphData] = useState<GraphData | null>(null);
    const [searchFilter, setSearchFilter] = useState('');
    const [categoryFilter, setCategoryFilter] = useState<string>('all');
    const [productCatalog, setProductCatalog] = useState<ProductCatalog>({});
    const [showEdges, setShowEdges] = useState(true);

    useEffect(() => {
        fetchStats();
        fetchGraphPreview();
        fetchProductCatalog();
    }, []);

    useEffect(() => {
        // Center camera after graph loads
        if (fgRef.current && graphData && view === '3d') {
            setTimeout(() => {
                // Center the graph and set good viewing distance
                const distance = 400;
                fgRef.current.cameraPosition(
                    { x: 0, y: 0, z: distance },
                    { x: 0, y: 0, z: 0 },
                    1000
                );
            }, 500);
        }
    }, [graphData, view]);

    useEffect(() => {
        if (selectedSKU) {
            fetchInfluences(selectedSKU);
        }
    }, [selectedSKU]);

    const fetchStats = async () => {
        try {
            const response = await fetch(`${API_URL}/gnn/graph-statistics`);
            if (response.ok) {
                const data = await response.json();
                setStats(data);
            }
        } catch (error) {
            console.error('Error fetching stats:', error);
        }
    };

    const fetchGraphPreview = async () => {
        setLoading(true);
        try {
            const response = await fetch(`${API_URL}/gnn/graph-structure?limit=240&min_edge_weight=0.0`);
            if (response.ok) {
                const data = await response.json();
                setGraphData(data);
                if (data.nodes.length > 0) {
                    setSelectedSKU(data.nodes[0].id);
                }
            }
        } catch (error) {
            console.error('Error fetching graph:', error);
        } finally {
            setLoading(false);
        }
    };

    const fetchInfluences = async (sku: string) => {
        try {
            const response = await fetch(`${API_URL}/gnn/product-influences/${sku}?top_k=10`);
            if (response.ok) {
                const data = await response.json();
                setInfluences(data.influences);
            }
        } catch (error) {
            console.error('Error fetching influences:', error);
        }
    };

    const fetchProductCatalog = async () => {
        try {
            const response = await fetch(`${API_URL}/products/catalog`);
            if (response.ok) {
                const data = await response.json();
                setProductCatalog(data.sku_lookup);
            }
        } catch (error) {
            console.error('Error fetching product catalog:', error);
        }
    };

    const getNodeColor = (node: GraphNode) => {
        return CATEGORY_COLORS[node.category] || '#64748b';
    };

    const getNodeSize = (node: GraphNode) => {
        return Math.max(3, Math.min(15, node.strength / 2));
    };

    const getEdgeColor = (weight: number) => {
        // Color gradient from blue (weak) to red (strong)
        if (weight < 0.5) return 'rgba(96, 165, 250, 0.4)'; // blue - weak
        if (weight < 1.0) return 'rgba(34, 197, 94, 0.5)'; // green - moderate
        if (weight < 2.0) return 'rgba(251, 191, 36, 0.6)'; // yellow - strong
        return 'rgba(239, 68, 68, 0.7)'; // red - very strong
    };

    const getSKUDescription = (sku: string) => {
        // Return real product name from catalog if available
        if (productCatalog[sku]?.name) {
            return productCatalog[sku].name;
        }
        // Fallback to generic description
        const match = sku.match(/SKU_([A-Z]{4})(\d{3})/);
        if (match) {
            const category = match[1];
            const number = match[2];
            return `${CATEGORY_NAMES[category] || category} - Item ${number}`;
        }
        return sku;
    };

    const selectedNode = graphData?.nodes.find(n => n.id === selectedSKU);

    const filteredSKUs = graphData?.nodes.filter(node => {
        const matchesSearch = searchFilter ? 
            node.id.toLowerCase().includes(searchFilter.toLowerCase()) ||
            getSKUDescription(node.id).toLowerCase().includes(searchFilter.toLowerCase())
            : true;
        const matchesCategory = categoryFilter === 'all' || node.category === categoryFilter;
        return matchesSearch && matchesCategory;
    }) || [];

    return (
        <div className="space-y-6">
            {/* Header with Tabs */}
            <Card glass>
                <CardHeader>
                    <div className="flex items-center justify-between flex-wrap gap-4">
                        <div>
                            <CardTitle className="flex items-center gap-2 text-lg">
                                <DatabaseIcon size={18} className="text-primary" />
                                GNN Product Influence Graph
                            </CardTitle>
                            <CardDescription>
                                Graph Neural Network-based product relationship analysis • {stats?.nodes || 0} products • {stats?.edges || 0} connections
                            </CardDescription>
                        </div>
                        <div className="flex gap-2">
                            <Button
                                variant={view === '3d' ? 'primary' : 'secondary'}
                                size="sm"
                                onClick={() => setView('3d')}
                            >
                                🌐 3D View
                            </Button>
                            <Button
                                variant={view === 'explorer' ? 'primary' : 'secondary'}
                                size="sm"
                                onClick={() => setView('explorer')}
                            >
                                🔍 Explorer
                            </Button>
                            <Button
                                variant={view === 'reference' ? 'primary' : 'secondary'}
                                size="sm"
                                onClick={() => setView('reference')}
                            >
                                📦 SKU Reference
                            </Button>
                        </div>
                    </div>
                </CardHeader>
            </Card>

            {/* Stats Grid */}
            {stats && (
                <div className="grid grid-cols-2 md:grid-cols-6 gap-4">
                    <Card glass>
                        <div className="p-4">
                            <p className="text-3xl font-bold text-primary">{stats.nodes}</p>
                            <p className="text-xs text-muted mt-1 uppercase">Total Nodes</p>
                        </div>
                    </Card>
                    <Card glass>
                        <div className="p-4">
                            <p className="text-3xl font-bold text-secondary">{stats.edges}</p>
                            <p className="text-xs text-muted mt-1 uppercase">Total Edges</p>
                        </div>
                    </Card>
                    <Card glass>
                        <div className="p-4">
                            <p className="text-3xl font-bold text-info">{stats.degree_stats.average.toFixed(1)}</p>
                            <p className="text-xs text-muted mt-1 uppercase">Avg Connections</p>
                        </div>
                    </Card>
                    <Card glass>
                        <div className="p-4">
                            <p className="text-3xl font-bold text-accent">{(stats.density * 100).toFixed(2)}%</p>
                            <p className="text-xs text-muted mt-1 uppercase">Graph Density</p>
                        </div>
                    </Card>
                    <Card glass>
                        <div className="p-4">
                            <p className="text-3xl font-bold text-warning">{stats.edge_weight_stats.average.toFixed(2)}</p>
                            <p className="text-xs text-muted mt-1 uppercase">Avg Edge Weight</p>
                        </div>
                    </Card>
                    <Card glass>
                        <div className="p-4">
                            <p className="text-2xl font-bold text-success">{stats.edge_weight_stats.min.toFixed(2)} - {stats.edge_weight_stats.max.toFixed(2)}</p>
                            <p className="text-xs text-muted mt-1 uppercase">Edge Weight Range</p>
                        </div>
                    </Card>
                </div>
            )}

            {/* 3D Visualization View */}
            {view === '3d' && (
                <>
                {/* Controls for 3D View */}
                <Card glass>
                    <CardContent className="p-4">
                        <div className="flex items-center justify-between mb-4">
                            <div className="flex items-center gap-4">
                                <label className="flex items-center gap-2 text-sm">
                                    <input
                                        type="checkbox"
                                        checked={showEdges}
                                        onChange={(e) => setShowEdges(e.target.checked)}
                                        className="rounded"
                                    />
                                    <span>Show Edges</span>
                                </label>
                            </div>
                            <div className="flex gap-4 text-xs">
                                <div className="flex items-center gap-2">
                                    <div className="w-8 h-1 bg-blue-400" />
                                    <span className="text-muted">Weak (0-0.5)</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-8 h-1 bg-green-500" />
                                    <span className="text-muted">Moderate (0.5-1.0)</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-8 h-1 bg-yellow-400" />
                                    <span className="text-muted">Strong (1.0-2.0)</span>
                                </div>
                                <div className="flex items-center gap-2">
                                    <div className="w-8 h-1 bg-red-500" />
                                    <span className="text-muted">Very Strong (2.0+)</span>
                                </div>
                            </div>
                        </div>
                        <div className="bg-info/10 border border-info/20 rounded-lg p-3">
                            <p className="text-xs text-info">
                                <strong>ℹ️ Real Data:</strong> Edge weights represent actual product relationships based on:
                                <span className="ml-2">• Co-purchase patterns from transaction data</span>
                                <span className="ml-2">• Category-based connections (substitutes)</span>
                                <span className="ml-2">• Temporal demand correlations</span>
                                <br />
                                <strong className="mt-1 block">Note:</strong> Products cluster naturally by category and purchase behavior. Disconnected groups indicate products rarely bought together.
                            </p>
                        </div>
                    </CardContent>
                </Card>

                <Card glass>
                    <CardContent className="p-0">
                        <div className="relative overflow-hidden rounded-lg" style={{ height: '700px', width: '100%' }}>
                            {loading ? (
                                <div className="absolute inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm">
                                    <div className="text-center">
                                        <RefreshIcon size={32} className="mx-auto mb-2 animate-spin text-primary" />
                                        <p className="text-sm text-muted">Loading graph...</p>
                                    </div>
                                </div>
                            ) : graphData ? (
                                <ForceGraph3D
                                    ref={fgRef}
                                    graphData={{
                                        nodes: graphData.nodes.map(node => ({
                                            ...node,
                                            color: getNodeColor(node),
                                            val: getNodeSize(node)
                                        })),
                                        links: showEdges ? graphData.edges.map(edge => ({
                                            source: edge.source,
                                            target: edge.target,
                                            value: edge.weight,
                                            color: getEdgeColor(edge.weight),
                                            distance: 50
                                        })) : []
                                    }}
                                    nodeLabel={(node: unknown) => {
                                        const n = node as { id: string };
                                        return `${n.id}\n${getSKUDescription(n.id)}`;
                                    }}
                                    linkColor={(link: unknown) => (link as { color: string }).color}
                                    linkWidth={(link: unknown) => {
                                        const l = link as { value: number };
                                        return Math.max(0.2, Math.min(3, l.value / 2));
                                    }}
                                    linkOpacity={0.6}
                                    backgroundColor="rgba(0,0,0,0)"
                                    showNavInfo={false}
                                    onNodeClick={(node: unknown) => setSelectedSKU((node as { id: string }).id)}
                                />
                            ) : (
                                <div className="flex items-center justify-center h-full text-muted">No graph data available</div>
                            )}
                        </div>
                    </CardContent>
                </Card>
                </>
            )}

            {/* Explorer View */}
            {view === 'explorer' && (
                <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                    <div className="lg:col-span-1">
                        <Card glass>
                            <CardHeader>
                                <CardTitle className="text-lg">Filters</CardTitle>
                                <CardDescription>Search and narrow down SKUs</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="space-y-4">
                                    <div>
                                        <label className="text-xs text-muted block mb-1">Search</label>
                                        <div className="relative">
                                            <SearchIcon size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
                                            <input
                                                type="text"
                                                value={searchFilter}
                                                onChange={(e) => setSearchFilter(e.target.value)}
                                                placeholder="SKU or name..."
                                                className="w-full bg-surface-elevated border border-white/10 rounded-lg py-2 pl-9 pr-3 text-sm focus:ring-1 focus:ring-info outline-none"
                                            />
                                        </div>
                                    </div>
                                    <div>
                                        <label className="text-xs text-muted block mb-1">Category</label>
                                        <select
                                            value={categoryFilter}
                                            onChange={(e) => setCategoryFilter(e.target.value)}
                                            className="w-full bg-surface-elevated border border-white/10 rounded-lg py-2 px-3 text-sm focus:ring-1 focus:ring-info outline-none"
                                        >
                                            <option value="all">All Categories</option>
                                            {Object.keys(CATEGORY_NAMES).map(code => (
                                                <option key={code} value={code}>{CATEGORY_NAMES[code]}</option>
                                            ))}
                                        </select>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </div>

                    <div className="lg:col-span-3">
                        <Card glass>
                            <CardHeader>
                                <CardTitle className="text-lg">SKU Explorer</CardTitle>
                                <CardDescription>Browse products and see their influence patterns</CardDescription>
                            </CardHeader>
                            <CardContent>
                                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
                                    {filteredSKUs.map(node => (
                                        <button
                                            key={node.id}
                                            onClick={() => setSelectedSKU(node.id)}
                                            className={`p-3 rounded-lg border text-left transition-colors ${selectedSKU === node.id ? 'border-primary/50 bg-primary/10' : 'border-white/10 hover:bg-white/5'}`}
                                        >
                                            <div className="flex items-center gap-2">
                                                <div className="w-6 h-6 rounded bg-surface-elevated flex items-center justify-center text-[10px] font-bold" style={{ color: getNodeColor(node) }}>
                                                    {node.category}
                                                </div>
                                                <div>
                                                    <div className="text-sm font-medium">{node.id}</div>
                                                    <div className="text-[10px] text-muted">{getSKUDescription(node.id)}</div>
                                                </div>
                                            </div>
                                        </button>
                                    ))}
                                </div>
                            </CardContent>
                        </Card>
                    </div>
                </div>
            )}

            {/* Reference View */}
            {view === 'reference' && selectedNode && (
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    {/* Selected SKU Summary */}
                    <Card glass>
                        <CardHeader>
                            <CardTitle className="text-lg">Selected SKU</CardTitle>
                            <CardDescription>{getSKUDescription(selectedNode.id)}</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="grid grid-cols-2 gap-4">
                                <div className="p-3 bg-white/5 rounded border border-white/10">
                                    <p className="text-[10px] text-muted uppercase">SKU</p>
                                    <p className="text-sm font-bold">{selectedNode.id}</p>
                                </div>
                                <div className="p-3 bg-white/5 rounded border border-white/10">
                                    <p className="text-[10px] text-muted uppercase">Category</p>
                                    <p className="text-sm font-bold">{CATEGORY_NAMES[selectedNode.category] || selectedNode.category}</p>
                                </div>
                                <div className="p-3 bg-white/5 rounded border border-white/10">
                                    <p className="text-[10px] text-muted uppercase">Strength</p>
                                    <p className="text-sm font-bold">{selectedNode.strength.toFixed(2)}</p>
                                </div>
                                <div className="p-3 bg-white/5 rounded border border-white/10">
                                    <p className="text-[10px] text-muted uppercase">Label</p>
                                    <p className="text-sm font-bold">{selectedNode.label}</p>
                                </div>
                            </div>
                        </CardContent>
                    </Card>

                    {/* Influences Table */}
                    <Card glass className="lg:col-span-2">
                        <CardHeader>
                            <CardTitle className="text-lg">Top Influences</CardTitle>
                            <CardDescription>Products most strongly connected to selected SKU</CardDescription>
                        </CardHeader>
                        <CardContent>
                            <div className="space-y-2">
                                {influences.length === 0 ? (
                                    <div className="text-center py-8 text-muted">No influences available</div>
                                ) : (
                                    influences.map((inf, idx) => (
                                        <div key={idx} className="flex items-center justify-between p-3 bg-white/5 rounded border border-white/10">
                                            <div>
                                                <div className="text-sm font-bold">{inf.sku}</div>
                                                <div className="text-[10px] text-muted">{CATEGORY_NAMES[inf.category] || inf.category}</div>
                                            </div>
                                            <div className="flex items-center gap-4">
                                                <Badge variant="info">Weight {inf.influence_weight.toFixed(2)}</Badge>
                                                <Badge variant="success">Strength {inf.influence_strength.toFixed(2)}</Badge>
                                            </div>
                                        </div>
                                    ))
                                )}
                            </div>
                        </CardContent>
                    </Card>
                </div>
            )}
        </div>
    );
}