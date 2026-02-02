'use client';

import { useState, useEffect } from 'react';
import Card, { CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import Badge from '@/components/ui/Badge';
import { Table, TableHeader, TableRow, TableHead, TableBody, TableCell } from '@/components/ui/Table';
import { CheckIcon, RefreshIcon } from '@/components/ui/Icons';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface ProductAccuracy {
    product_id: string;
    avg_demand: number;
    mae: number;
    mape: number;
    wape: number;
}

interface ProductAccuracyResponse {
    products: ProductAccuracy[];
    total_count: number;
    showing: number;
}

import { useCallback } from 'react';

export default function ProductAccuracyTable() {
    const [products, setProducts] = useState<ProductAccuracy[]>([]);
    const [loading, setLoading] = useState(false);
    const [sortBy, setSortBy] = useState<'mae' | 'mape' | 'wape'>('mae');
    const [limit, setLimit] = useState(10);

    const fetchProductAccuracy = useCallback(async () => {
        setLoading(true);
        try {
            const response = await fetch(`${API_URL}/analytics/product-accuracy?limit=${limit}&sort_by=${sortBy}`);
            if (response.ok) {
                const data: ProductAccuracyResponse = await response.json();
                setProducts(data.products);
            }
        } catch (error) {
            console.error('Error fetching product accuracy:', error);
        } finally {
            setLoading(false);
        }
    }, [limit, sortBy]);

    useEffect(() => {
        fetchProductAccuracy();
    }, [fetchProductAccuracy]);

    const getAccuracyColor = (mape: number) => {
        if (mape < 10) return 'text-success';
        if (mape < 20) return 'text-warning';
        return 'text-error';
    };

    return (
        <Card glass>
            <CardHeader>
                <div className="flex items-center justify-between">
                    <div>
                        <CardTitle className="flex items-center gap-2 text-lg">
                            <CheckIcon size={18} className="text-success" />
                            Product-Level Accuracy
                        </CardTitle>
                        <CardDescription>Per-product forecast performance metrics</CardDescription>
                    </div>
                    <div className="flex items-center gap-2">
                        <select
                            value={sortBy}
                            onChange={(e) => setSortBy(e.target.value as 'mae' | 'mape' | 'wape')}
                            className="bg-surface-elevated border border-white/10 rounded px-2 py-1 text-xs"
                        >
                            <option value="mae">Sort by MAE</option>
                            <option value="mape">Sort by MAPE</option>
                            <option value="wape">Sort by WAPE</option>
                        </select>
                        <select
                            value={limit}
                            onChange={(e) => setLimit(Number(e.target.value))}
                            className="bg-surface-elevated border border-white/10 rounded px-2 py-1 text-xs"
                        >
                            <option value="5">Top 5</option>
                            <option value="10">Top 10</option>
                            <option value="20">Top 20</option>
                            <option value="50">Top 50</option>
                        </select>
                        <button
                            onClick={fetchProductAccuracy}
                            className="p-1.5 hover:bg-white/5 rounded transition-colors"
                            disabled={loading}
                        >
                            <RefreshIcon size={14} className={loading ? 'animate-spin' : ''} />
                        </button>
                    </div>
                </div>
            </CardHeader>
            <CardContent>
                {loading ? (
                    <div className="text-center py-8 text-muted">
                        <RefreshIcon size={24} className="mx-auto mb-2 animate-spin" />
                        Loading product accuracy data...
                    </div>
                ) : products.length === 0 ? (
                    <div className="text-center py-8 text-muted">No data available</div>
                ) : (
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Product ID</TableHead>
                                <TableHead className="text-right">Avg Demand</TableHead>
                                <TableHead className="text-right">MAE</TableHead>
                                <TableHead className="text-right">MAPE %</TableHead>
                                <TableHead className="text-right">WAPE %</TableHead>
                                <TableHead className="text-right">Performance</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {products.map((product) => (
                                <TableRow key={product.product_id} className="hover:bg-white/5">
                                    <TableCell>
                                        <span className="font-mono text-sm">{product.product_id}</span>
                                    </TableCell>
                                    <TableCell className="text-right font-mono text-sm">
                                        {product.avg_demand.toFixed(2)}
                                    </TableCell>
                                    <TableCell className="text-right font-mono">
                                        {product.mae.toFixed(2)}
                                    </TableCell>
                                    <TableCell className={`text-right font-mono ${getAccuracyColor(product.mape)}`}>
                                        {product.mape.toFixed(1)}%
                                    </TableCell>
                                    <TableCell className="text-right font-mono">
                                        {product.wape.toFixed(1)}%
                                    </TableCell>
                                    <TableCell className="text-right">
                                        {product.mape < 10 ? (
                                            <Badge variant="success">Excellent</Badge>
                                        ) : product.mape < 20 ? (
                                            <Badge variant="warning">Good</Badge>
                                        ) : (
                                            <Badge variant="error">Needs Review</Badge>
                                        )}
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                )}
            </CardContent>
        </Card>
    );
}