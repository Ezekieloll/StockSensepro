'use client';

import { useState, useEffect } from 'react';
import Card, { CardHeader, CardTitle, CardDescription, CardContent } from '@/components/ui/Card';
import { ChartIcon, DatabaseIcon, TrendingUpIcon } from '@/components/ui/Icons';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface AccuracySummary {
    overall: {
        mae: number;
        mape: number;
        wape: number;
        product_count: number;
    };
    mae_distribution: {
        min: number;
        max: number;
        median: number;
        p25: number;
        p75: number;
    };
    mape_distribution: {
        min: number;
        max: number;
        median: number;
        p25: number;
        p75: number;
    };
    analysis_date: string;
}

export default function AccuracySummaryCard() {
    const [summary, setSummary] = useState<AccuracySummary | null>(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        fetchSummary();
    }, []);

    const fetchSummary = async () => {
        setLoading(true);
        try {
            const response = await fetch(`${API_URL}/analytics/accuracy-summary`);
            if (response.ok) {
                const data = await response.json();
                setSummary(data);
            }
        } catch (error) {
            console.error('Error fetching accuracy summary:', error);
        } finally {
            setLoading(false);
        }
    };

    if (loading) {
        return (
            <Card glass>
                <CardContent className="py-8 text-center text-muted">
                    Loading summary...
                </CardContent>
            </Card>
        );
    }

    if (!summary) {
        return null;
    }

    return (
        <Card glass>
            <CardHeader>
                <CardTitle className="flex items-center gap-2 text-lg">
                    <DatabaseIcon size={18} className="text-info" />
                    Accuracy Distribution
                </CardTitle>
                <CardDescription>Statistical summary across {summary.overall.product_count} products</CardDescription>
            </CardHeader>
            <CardContent>
                <div className="grid grid-cols-2 gap-4">
                    {/* Overall Metrics */}
                    <div className="col-span-2 p-4 bg-white/5 rounded-lg border border-white/5">
                        <p className="text-xs text-muted uppercase tracking-wider mb-3">Overall Performance</p>
                        <div className="grid grid-cols-3 gap-4">
                            <div>
                                <p className="text-2xl font-bold text-info">{summary.overall.mae.toFixed(2)}</p>
                                <p className="text-xs text-muted mt-1">Avg MAE</p>
                            </div>
                            <div>
                                <p className="text-2xl font-bold text-warning">{summary.overall.mape.toFixed(1)}%</p>
                                <p className="text-xs text-muted mt-1">Avg MAPE</p>
                            </div>
                            <div>
                                <p className="text-2xl font-bold">{summary.overall.wape.toFixed(1)}%</p>
                                <p className="text-xs text-muted mt-1">Avg WAPE</p>
                            </div>
                        </div>
                    </div>

                    {/* MAE Distribution */}
                    <div className="p-4 bg-white/5 rounded-lg border border-white/5">
                        <p className="text-xs text-muted uppercase tracking-wider mb-3 flex items-center gap-1">
                            <ChartIcon size={12} /> MAE Distribution
                        </p>
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                                <span className="text-muted">Min</span>
                                <span className="font-mono text-success">{summary.mae_distribution.min.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted">25th %</span>
                                <span className="font-mono">{summary.mae_distribution.p25.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted">Median</span>
                                <span className="font-mono font-bold">{summary.mae_distribution.median.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted">75th %</span>
                                <span className="font-mono">{summary.mae_distribution.p75.toFixed(2)}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted">Max</span>
                                <span className="font-mono text-error">{summary.mae_distribution.max.toFixed(2)}</span>
                            </div>
                        </div>
                    </div>

                    {/* MAPE Distribution */}
                    <div className="p-4 bg-white/5 rounded-lg border border-white/5">
                        <p className="text-xs text-muted uppercase tracking-wider mb-3 flex items-center gap-1">
                            <TrendingUpIcon size={12} /> MAPE Distribution
                        </p>
                        <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                                <span className="text-muted">Min</span>
                                <span className="font-mono text-success">{summary.mape_distribution.min.toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted">25th %</span>
                                <span className="font-mono">{summary.mape_distribution.p25.toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted">Median</span>
                                <span className="font-mono font-bold">{summary.mape_distribution.median.toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted">75th %</span>
                                <span className="font-mono">{summary.mape_distribution.p75.toFixed(1)}%</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-muted">Max</span>
                                <span className="font-mono text-error">{summary.mape_distribution.max.toFixed(1)}%</span>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="mt-4 pt-3 border-t border-white/10 text-xs text-muted text-center">
                    Last analyzed: {new Date(summary.analysis_date).toLocaleString()}
                </div>
            </CardContent>
        </Card>
    );
}