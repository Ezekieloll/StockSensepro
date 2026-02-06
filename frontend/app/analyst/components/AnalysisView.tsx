'use client';

import { useState, useEffect } from 'react';
import Card, { CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import { ChartIcon, SearchIcon, CheckIcon, CalendarIcon, DatabaseIcon } from '@/components/ui/Icons';

// --- Types ---
interface Product {
    sku: string;
    name: string;
    category: string;
}

interface DataPoint {
    date: string;
    [key: string]: number | string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// --- Mock Data Generators ---
const generateMockHistory = (products: Product[], startDate: string, endDate: string, storeId: string) => {
    const data: DataPoint[] = [];
    const start = new Date(startDate);
    const end = new Date(endDate);

    // Safety check for date loop
    if (start > end) return [];

    const current = new Date(start);
    while (current <= end) {
        const dateStr = current.toISOString().split('T')[0];
        const point: DataPoint = { date: dateStr };

        products.forEach(p => {
            // Seed based on SKU + Store + Date to be deterministic but varied
            const seed = p.sku.charCodeAt(p.sku.length - 1) + storeId.charCodeAt(0) + current.getDate();
            const base = 50 + (seed % 100);
            const noise = (Math.sin(current.getTime()) * 20);

            point[p.sku] = Math.max(0, Math.round(base + noise));
        });

        data.push(point);
        current.setDate(current.getDate() + 1);
    }
    return data;
};

// --- Aggregate Data ---
const aggregateData = (data: DataPoint[], periodicity: 'daily' | 'weekly' | 'monthly') => {
    if (periodicity === 'daily' || data.length === 0) return data;

    const aggregated: DataPoint[] = [];
    let currentPeriod: DataPoint | null = null;
    let periodKey = '';

    data.forEach(d => {
        let key = '';
        if (periodicity === 'weekly') {
            // ISO Week (simplified: just grouping by year-week string if available, or just buckets of 7)
            // Using buckets of 7 for simplicity and speed (approx weekly)
            // Better: get actual week number
            const date = new Date(d.date);
            const onejan = new Date(date.getFullYear(), 0, 1);
            const week = Math.ceil((((date.getTime() - onejan.getTime()) / 86400000) + onejan.getDay() + 1) / 7);
            key = `${date.getFullYear()}-W${week}`;

        } else if (periodicity === 'monthly') {
            key = d.date.substring(0, 7); // YYYY-MM
        }

        if (key !== periodKey) {
            if (currentPeriod) {
                // finalize previous
                // Average the values? Or sum? For Demand/Sales, SUM is usually better. 
                // However, for viewing "trends" of avg daily sales, average is better.
                // Let's do Average for now to keep scale similar to daily.
                Object.keys(currentPeriod).forEach(k => {
                    if (k !== 'date' && k !== 'count') {
                        currentPeriod![k] = Number(currentPeriod![k]) / (currentPeriod!['count'] as number);
                    }
                });
                aggregated.push(currentPeriod);
            }
            periodKey = key;
            currentPeriod = { date: key, count: 0 };
            // Initialize sums
            Object.keys(d).forEach(k => {
                if (k !== 'date') currentPeriod![k] = 0;
            });
        }

        // Add to sum
        if (currentPeriod) {
            currentPeriod['count'] = (currentPeriod['count'] as number) + 1;
            Object.keys(d).forEach(k => {
                if (k !== 'date') {
                    if (!currentPeriod![k]) currentPeriod![k] = 0;
                    currentPeriod![k] = Number(currentPeriod![k]) + Number(d[k]);
                }
            });
        }
    });

    // Push last
    if (currentPeriod) {
        Object.keys(currentPeriod).forEach(k => {
            if (k !== 'date' && k !== 'count') {
                currentPeriod![k] = Number(currentPeriod![k]) / (currentPeriod!['count'] as number);
            }
        });
        aggregated.push(currentPeriod);
    }

    return aggregated;
};

export default function AnalysisView() {
    // --- Filters State ---
    const [selectedStore, setSelectedStore] = useState('S1');
    const [startDate, setStartDate] = useState('2023-01-01');
    const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
    const [searchQuery, setSearchQuery] = useState('');

    // --- Data State ---
    const [products, setProducts] = useState<Product[]>([]);
    const [selectedProducts, setSelectedProducts] = useState<string[]>([]);
    const [chartData, setChartData] = useState<DataPoint[]>([]);
    const [loading, setLoading] = useState(false);

    // --- Visualization State ---
    const [chartType, setChartType] = useState<'line' | 'bar' | 'area'>('line');

    // Fetch Products
    useEffect(() => {
        const fetchProducts = async () => {
            try {
                const res = await fetch(`${API_URL}/forecast/products`);
                if (res.ok) {
                    const data = await res.json();
                    // map to simple product
                    const mapped = data.map((p: any) => ({
                        sku: p.sku,
                        name: p.name,
                        category: p.category_name || p.category // fallback
                    }));
                    setProducts(mapped);
                    if (mapped.length > 0) {
                        setSelectedProducts([mapped[0].sku]);
                    }
                } else {
                    // Fallback mock
                    setProducts([
                        { sku: 'SKU_001', name: 'Organic Milk 1L', category: 'Dairy' },
                        { sku: 'SKU_015', name: 'Whole Wheat Bread', category: 'Bakery' },
                    ]);
                }
            } catch (e) {
                console.error("Failed to fetch products", e);
            }
        };
        fetchProducts();
    }, []);

    // Generate/Fetch Data
    useEffect(() => {
        const fetchHistoryData = async () => {
            if (selectedProducts.length === 0) {
                setChartData([]);
                return;
            }

            setLoading(true);
            try {
                // Fetch data for all selected products in parallel
                const fetchPromises = selectedProducts.map(sku =>
                    fetch(`${API_URL}/forecast/history/${sku}?store_id=${selectedStore}&start_date=${startDate}&end_date=${endDate}`)
                        .then(res => res.ok ? res.json() : { history: [] })
                );

                const results = await Promise.all(fetchPromises);

                // Merge data from multiple products by date
                const dateMap: { [date: string]: DataPoint } = {};

                results.forEach((result, idx) => {
                    const sku = selectedProducts[idx];
                    result.history.forEach((dp: any) => {
                        if (!dateMap[dp.date]) {
                            dateMap[dp.date] = { date: dp.date };
                        }
                        dateMap[dp.date][sku] = dp.actual;
                    });
                });

                // Convert map back to sorted array
                let combinedData = Object.values(dateMap).sort((a, b) => a.date.localeCompare(b.date));

                // Auto-Aggregation logic based on count
                if (combinedData.length > 300) {
                    combinedData = aggregateData(combinedData, 'monthly');
                } else if (combinedData.length > 60) {
                    combinedData = aggregateData(combinedData, 'weekly');
                }

                setChartData(combinedData);
            } catch (error) {
                console.error("Error fetching historical data:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchHistoryData();
    }, [selectedProducts, startDate, endDate, selectedStore]);

    const toggleProduct = (sku: string) => {
        if (selectedProducts.includes(sku)) {
            setSelectedProducts(selectedProducts.filter(s => s !== sku));
        } else {
            if (selectedProducts.length >= 5) {
                alert("You can compare up to 5 products at a time.");
                return;
            }
            setSelectedProducts([...selectedProducts, sku]);
        }
    };

    const colors = ['#06b6d4', '#8b5cf6', '#ec4899', '#10b981', '#f59e0b']; // Cyan, Violet, Pink, Emerald, Amber

    // --- Chart Renderer ---
    const renderChart = () => {
        if (chartData.length === 0) {
            return (
                <div className="h-full flex flex-col items-center justify-center text-muted gap-2">
                    <ChartIcon size={32} className="opacity-20" />
                    <p>Select products and a date range to generate analysis</p>
                </div>
            );
        }

        if (loading) {
            return (
                <div className="h-full flex items-center justify-center text-muted animate-pulse">
                    Generating analysis...
                </div>
            );
        }

        const width = 1000;
        const height = 400;
        const padding = { top: 40, right: 30, bottom: 60, left: 60 };
        const innerWidth = width - padding.left - padding.right;
        const innerHeight = height - padding.top - padding.bottom;

        // Calculate Scale
        let maxVal = 0;
        chartData.forEach(d => {
            selectedProducts.forEach(sku => {
                const val = Number(d[sku]);
                if (val > maxVal) maxVal = val;
            });
        });
        maxVal = Math.max(maxVal * 1.1, 10); // Minimum scale of 10

        const xScale = (index: number) => (index / (chartData.length - 1)) * innerWidth; // Note: -1 can be 0 if length is 1
        const safeXScale = (index: number) => {
            if (chartData.length <= 1) return innerWidth / 2;
            return xScale(index);
        }
        const yScale = (value: number) => innerHeight - ((value / maxVal) * innerHeight);

        // Grid Lines
        const yTicks = 5;

        return (
            <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
                <g transform={`translate(${padding.left}, ${padding.top})`}>

                    {/* Y-Axis Label */}
                    <text transform="rotate(-90)" x={-innerHeight / 2} y={-45} textAnchor="middle" fill="white" opacity="0.5" fontSize="12" fontWeight="bold" letterSpacing="1px">
                        UNITS (AVG)
                    </text>

                    {/* Y-Axis Grid & Labels */}
                    {Array.from({ length: yTicks + 1 }).map((_, i) => {
                        const val = (maxVal / yTicks) * i;
                        const y = yScale(val);
                        return (
                            <g key={`y-${i}`}>
                                <line x1={0} y1={y} x2={innerWidth} y2={y} stroke="rgba(255,255,255,0.1)" strokeDasharray="4" />
                                <text x={-10} y={y + 4} textAnchor="end" fill="rgba(255,255,255,0.5)" fontSize="11" className="font-mono">
                                    {Math.round(val)}
                                </text>
                            </g>
                        );
                    })}

                    {/* X-Axis Labels */}
                    {chartData.map((d, i) => {
                        // Show max 8 labels
                        const step = Math.ceil(chartData.length / 8);
                        if (i % step === 0) {
                            return (
                                <g key={`x-${i}`}>
                                    <text x={safeXScale(i)} y={innerHeight + 25} textAnchor="middle" fill="rgba(255,255,255,0.5)" fontSize="11">
                                        {d.date}
                                    </text>
                                    <line x1={safeXScale(i)} y1={innerHeight} x2={safeXScale(i)} y2={innerHeight + 5} stroke="rgba(255,255,255,0.2)" />
                                </g>
                            );
                        }
                        return null;
                    })}
                    <text x={innerWidth / 2} y={innerHeight + 50} textAnchor="middle" fill="white" opacity="0.5" fontSize="12" fontWeight="bold" letterSpacing="1px">
                        DATE
                    </text>

                    {/* Data Visualization */}
                    {selectedProducts.map((sku, idx) => {
                        const color = colors[idx % colors.length];

                        // Line Chart
                        if (chartType === 'line') {
                            const pathD = chartData.map((d, i) =>
                                `${i === 0 ? 'M' : 'L'} ${safeXScale(i)} ${yScale(Number(d[sku]))}`
                            ).join(' ');

                            return (
                                <g key={sku}>
                                    <path d={pathD} fill="none" stroke={color} strokeWidth="2" className="drop-shadow-md" />
                                </g>
                            );
                        }

                        // Area Chart
                        if (chartType === 'area') {
                            const pathD = chartData.map((d, i) =>
                                `${i === 0 ? 'M' : 'L'} ${safeXScale(i)} ${yScale(Number(d[sku]))}`
                            ).join(' ');
                            // Close the path
                            const areaPath = `${pathD} L ${chartData.length > 0 ? safeXScale(chartData.length - 1) : 0} ${innerHeight} L 0 ${innerHeight} Z`;

                            return (
                                <g key={sku}>
                                    <path d={areaPath} fill={color} fillOpacity="0.2" stroke="none" />
                                    <path d={pathD} fill="none" stroke={color} strokeWidth="2" />
                                </g>
                            );
                        }

                        // Bar Chart
                        if (chartType === 'bar') {
                            // Only render bars if data is not too dense, otherwise user should see line/aggregated
                            // Dynamic bar width
                            const groupWidth = innerWidth / chartData.length;
                            const barWidth = Math.max((groupWidth / selectedProducts.length) * 0.8, 4); // min 4px width

                            // If too dense, only render lines? Or render simpler bars
                            return chartData.map((d, i) => {
                                const val = Number(d[sku]);
                                const h = innerHeight - yScale(val);
                                // Center the group
                                const groupStart = safeXScale(i) - (groupWidth / 2);
                                const x = groupStart + (groupWidth / 2) - ((selectedProducts.length * barWidth) / 2) + (idx * barWidth);

                                return (
                                    <rect
                                        key={i}
                                        x={x}
                                        y={yScale(val)}
                                        width={barWidth}
                                        height={h}
                                        fill={color}
                                        opacity="0.8"
                                        rx="1"
                                    />
                                );
                            });
                        }
                    })}
                </g>
            </svg>
        );
    };

    return (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 min-h-[600px]">
            {/* Sidebar Controls */}
            <div className="md:col-span-1 space-y-6">

                {/* Configuration Panel */}
                <Card glass className="h-full">
                    <CardHeader>
                        <CardTitle className="text-lg flex items-center gap-2">
                            <DatabaseIcon size={18} className="text-primary" />
                            Configuration
                        </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-6">

                        {/* Store Selection */}
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-muted uppercase tracking-wider">Store Location</label>
                            <select
                                value={selectedStore}
                                onChange={(e) => setSelectedStore(e.target.value)}
                                className="w-full bg-slate-900 border border-white/20 rounded-lg p-2 text-sm text-white focus:ring-1 focus:ring-primary outline-none"
                            >
                                <option value="S1">Store S1 (Downtown)</option>
                                <option value="S2">Store S2 (Westside)</option>
                                <option value="S3">Store S3 (Airport)</option>
                            </select>
                        </div>

                        {/* Date Selection */}
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-muted uppercase tracking-wider">Analysis Period</label>
                            <div className="grid grid-cols-1 gap-2">
                                <div>
                                    <span className="text-[10px] text-muted block mb-1">Start Date</span>
                                    <div className="relative">
                                        <input
                                            type="date"
                                            value={startDate}
                                            onChange={(e) => setStartDate(e.target.value)}
                                            className="w-full bg-slate-900 border border-white/20 rounded-lg p-2 text-sm text-white focus:ring-1 focus:ring-primary outline-none"
                                        />
                                        <CalendarIcon size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted" />
                                    </div>
                                </div>
                                <div>
                                    <span className="text-[10px] text-muted block mb-1">End Date</span>
                                    <div className="relative">
                                        <input
                                            type="date"
                                            value={endDate}
                                            onChange={(e) => setEndDate(e.target.value)}
                                            className="w-full bg-slate-900 border border-white/20 rounded-lg p-2 text-sm text-white focus:ring-1 focus:ring-primary outline-none"
                                        />
                                        <CalendarIcon size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted" />
                                    </div>
                                </div>
                            </div>
                        </div>

                        {/* Product Search */}
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-muted uppercase tracking-wider">Product Search</label>
                            <div className="relative">
                                <SearchIcon size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-muted" />
                                <input
                                    type="text"
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    placeholder="Search SKU or name..."
                                    className="w-full bg-slate-900 border border-white/20 rounded-lg p-2 pl-9 text-sm text-white focus:ring-1 focus:ring-primary outline-none"
                                />
                            </div>
                        </div>

                        {/* Product Selection */}
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-muted uppercase tracking-wider">Products</label>
                            <div className="space-y-2">
                                {products
                                    .filter(p => p.name.toLowerCase().includes(searchQuery.toLowerCase()) || p.sku.toLowerCase().includes(searchQuery.toLowerCase()))
                                    .slice(0, 10)
                                    .map((p) => (
                                        <div key={p.sku} className="flex items-center justify-between p-2 bg-white/5 rounded-lg border border-white/10 hover:bg-white/10 transition-colors">
                                            <div className="flex items-center gap-2">
                                                <div className="w-6 h-6 rounded bg-surface-elevated flex items-center justify-center text-[10px] font-bold">
                                                    {p.sku.slice(-2)}
                                                </div>
                                                <div>
                                                    <div className="text-sm font-medium">{p.name}</div>
                                                    <div className="text-[10px] text-muted">{p.category}</div>
                                                </div>
                                            </div>
                                            <button
                                                onClick={() => toggleProduct(p.sku)}
                                                className={`px-2 py-1 rounded text-xs ${selectedProducts.includes(p.sku) ? 'bg-primary/20 text-primary' : 'bg-white/10 text-muted'}`}
                                            >
                                                {selectedProducts.includes(p.sku) ? 'Selected' : 'Compare'}
                                            </button>
                                        </div>
                                    ))}
                            </div>
                            {selectedProducts.length > 0 && (
                                <div className="flex flex-wrap gap-2 mt-2">
                                    {selectedProducts.map(sku => (
                                        <Badge key={sku} variant="info" className="text-[10px]">
                                            <CheckIcon size={10} className="mr-1" /> {sku}
                                        </Badge>
                                    ))}
                                </div>
                            )}
                        </div>

                        {/* Chart Type */}
                        <div className="space-y-2">
                            <label className="text-xs font-bold text-muted uppercase tracking-wider">Chart Type</label>
                            <div className="grid grid-cols-3 gap-2">
                                {['line', 'area', 'bar'].map(type => (
                                    <button
                                        key={type}
                                        onClick={() => setChartType(type as any)}
                                        className={`px-2 py-1 rounded text-xs ${chartType === type ? 'bg-primary/20 text-primary' : 'bg-white/10 text-muted'}`}
                                    >
                                        {type.toUpperCase()}
                                    </button>
                                ))}
                            </div>
                        </div>

                    </CardContent>
                </Card>
            </div>

            {/* Chart Panel */}
            <div className="md:col-span-3">
                <Card glass className="h-full">
                    <CardHeader>
                        <CardTitle className="text-lg flex items-center gap-2">
                            <ChartIcon size={18} className="text-info" />
                            Comparative Demand Analysis
                        </CardTitle>
                        <CardDescription>Visualize historical demand trends across selected products</CardDescription>
                    </CardHeader>
                    <CardContent>
                        <div className="w-full overflow-x-auto">
                            {renderChart()}
                        </div>
                        {selectedProducts.length > 0 && (
                            <div className="mt-4 flex flex-wrap gap-4 px-2">
                                {selectedProducts.map((sku, idx) => {
                                    const product = products.find(p => p.sku === sku);
                                    const color = colors[idx % colors.length];
                                    return (
                                        <div key={sku} className="flex items-center gap-2">
                                            <div className="w-3 h-3 rounded-full shadow-sm" style={{ backgroundColor: color }}></div>
                                            <span className="text-xs font-medium text-foreground/80">{product?.name || sku}</span>
                                            <span className="text-[10px] text-muted font-mono">{sku}</span>
                                        </div>
                                    );
                                })}
                            </div>
                        )}
                        <div className="mt-6 flex items-center gap-2 text-[10px] text-muted border-t border-white/5 pt-4">
                            <span className="w-2 h-2 rounded-full bg-primary/40"></span>
                            <span>Lines represent average demand values over time. For dense datasets, aggregation automatically switches to weekly/monthly.</span>
                        </div>
                    </CardContent>
                </Card>
            </div>

        </div>
    );
}