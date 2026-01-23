'use client';

import { useState, useEffect, useRef } from 'react';
import Card, { CardHeader, CardTitle, CardContent } from './Card';
import Button from './Button';
import { RefreshIcon, CheckIcon, AlertIcon, TrendingUpIcon } from './Icons';

interface InsightAssistantProps {
    forecasts?: any[];
    alerts?: any[];
    summary?: any;
    productDetail?: any;
}

interface Insight {
    type: 'analysis' | 'suggestion' | 'warning';
    content: string;
}

export default function InsightAssistant({ forecasts, alerts, summary, productDetail }: InsightAssistantProps) {
    const [isOpen, setIsOpen] = useState(false);
    const [loading, setLoading] = useState(false);
    const [insights, setInsights] = useState<Insight[]>([]);
    const [hasGenerated, setHasGenerated] = useState(false);
    const chatEndRef = useRef<HTMLDivElement>(null);

    const generateInsights = async () => {
        setLoading(true);
        // Simulate LLM processing delay
        await new Promise(resolve => setTimeout(resolve, 1500));

        const newInsights: Insight[] = [];

        // --- SCOPE: SINGLE PRODUCT DETAIL (Forecasts Page) ---
        if (productDetail) {
            // 1. Core Status & Forecast
            newInsights.push({
                type: 'analysis',
                content: `Analyzing ${productDetail.product_name} (${productDetail.sku}): Current stock is ${productDetail.current_stock}. The forecast suggests a demand of ${productDetail.seven_day_forecast.toFixed(0)} units over the next 7 days.`
            });

            // 2. Trend Analysis (Is demand going up or down?)
            if (productDetail.demand_data && productDetail.demand_data.length > 0) {
                const forecastsPoints = productDetail.demand_data.filter((d: any) => d.is_forecast && d.forecast !== null);
                if (forecastsPoints.length > 1) {
                    const first = forecastsPoints[0].forecast;
                    const last = forecastsPoints[forecastsPoints.length - 1].forecast;
                    const diff = last - first;
                    const percentChange = (diff / first) * 100;

                    if (percentChange > 10) {
                        newInsights.push({
                            type: 'suggestion',
                            content: `📈 Trend Alert: Demand is rapidly increasing! It's expected to grow by ${percentChange.toFixed(0)}% over the forecast period. Consider increasing safety stock.`
                        });
                    } else if (percentChange < -10) {
                        newInsights.push({
                            type: 'analysis',
                            content: `📉 Trend Alert: Demand is tapering off. Expect a ${Math.abs(percentChange).toFixed(0)}% drop by the end of the period. You can safely reduce replenishment orders.`
                        });
                    } else {
                        newInsights.push({
                            type: 'analysis',
                            content: `➡️ Demand is relatively stable. Standard replenishment schedules should suffice.`
                        });
                    }
                }
            }

            // 3. Critical Stock & Restock Recommendation
            if (productDetail.stock_status === 'critical') {
                const deficit = productDetail.seven_day_forecast - productDetail.current_stock;
                const recommendedOrder = Math.ceil(deficit * 1.2); // +20% buffer
                newInsights.push({
                    type: 'warning',
                    content: `CRITICAL: You are projected to stock out in ${productDetail.stock_days_remaining} days. I recommend placing an immediate order for at least ${recommendedOrder} units to cover the coming week plus safety stock.`
                });
            } else if (productDetail.stock_days_remaining < 14) {
                newInsights.push({
                    type: 'suggestion',
                    content: `Stock is getting low (${productDetail.stock_days_remaining} days left). Add this item to your next scheduled purchase order.`
                });
            }

            // 4. Confidence Context
            if (productDetail.confidence < 0.7) {
                newInsights.push({
                    type: 'analysis',
                    content: `⚠️ Low Confidence (${(productDetail.confidence * 100).toFixed(0)}%): The model is uncertain, possibly due to erratic historical patterns. Verify with recent sales data manually before making large bulk orders.`
                });
            } else {
                newInsights.push({
                    type: 'analysis',
                    content: `✅ High Confidence (${(productDetail.confidence * 100).toFixed(0)}%): The prediction allows for aggressive optimization. You can lower safety stock buffers to free up capital.`
                });
            }

        }
        // --- SCOPE: DASHBOARD OVERVIEW (Manager Page) ---
        else {

            // 1. Overall Health Check
            if (summary) {
                if (summary.critical_stock_count > 0) {
                    newInsights.push({
                        type: 'warning',
                        content: `🚨 Action Required: ${summary.critical_stock_count} products are at critical levels. This poses a high risk of lost revenue.`
                    });
                }
                newInsights.push({
                    type: 'analysis',
                    content: `Model Health: The system is running with ${(summary.avg_confidence * 100).toFixed(1)}% average confidence. ${summary.avg_confidence > 0.85 ? "Predictions are highly reliable." : "Monitor low-confidence items closely."}`
                });
            }

            // 2. Category Intelligence (Aggregated Analysis)
            if (forecasts && forecasts.length > 0) {
                // Find most affected category (Mock logic as forecasts might not cover all)
                const categoryCounts: Record<string, number> = {};
                forecasts.forEach(f => {
                    if (f.stock_status === 'critical' || f.stock_status === 'low') {
                        categoryCounts[f.category] = (categoryCounts[f.category] || 0) + 1;
                    }
                });
                const riskyCategory = Object.keys(categoryCounts).reduce((a, b) => categoryCounts[a] > categoryCounts[b] ? a : b, '');

                if (riskyCategory) {
                    newInsights.push({
                        type: 'analysis',
                        content: `📦 Category Insight: The '${riskyCategory}' category has the highest number of inventory alerts. Check for supplier delays or seasonal demand spikes in this segment.`
                    });
                }

                // Demand Surges
                const trendingUp = forecasts.filter(f => f.seven_day_forecast > f.current_stock * 1.5);
                if (trendingUp.length > 0) {
                    newInsights.push({
                        type: 'suggestion',
                        content: `🚀 Opportunity: ${trendingUp.length} products (e.g., ${trendingUp[0].product_name}) are predicted to see massive demand (>150% of stock). Consider running a promotion or ensuring prime shelf placement.`
                    });
                }
            }

            // 3. Smart Alert Prioritization
            if (alerts && alerts.length > 0) {
                const highSeverity = alerts.filter(a => a.severity === 'high');
                if (highSeverity.length > 0) {
                    // Store clustering
                    const problematicStore = highSeverity[0].store_id;
                    newInsights.push({
                        type: 'warning',
                        content: `📍 Store Focus: Store ${problematicStore} has the most critical alerts. I suggest doing a full inventory audit for this location.`
                    });
                }
            } else {
                newInsights.push({
                    type: 'analysis',
                    content: "Everything looks optimal. Use this time to review slow-moving inventory candidates for potential clearance sales."
                });
            }
        }

        if (newInsights.length === 0) {
            newInsights.push({
                type: 'analysis',
                content: "System is stable. I'm monitoring for any new patterns."
            });
        }

        setInsights(newInsights);
        setLoading(false);
        setHasGenerated(true);
    };

    useEffect(() => {
        if (isOpen && !hasGenerated) {
            generateInsights();
        }
    }, [isOpen]);

    useEffect(() => {
        chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [insights]);

    return (
        <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end pointer-events-none">
            {/* Chat Window */}
            {isOpen && (
                <div className="mb-4 w-80 md:w-96 pointer-events-auto transition-all animate-in slide-in-from-bottom-5 fade-in duration-300">
                    <Card glass className="border-secondary/20 shadow-2xl shadow-secondary/10 flex flex-col max-h-[500px]">
                        <CardHeader className="bg-white/5 border-b border-white/5 py-3">
                            <div className="flex items-center justify-between">
                                <div className="flex items-center gap-2">
                                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center">
                                        <div className="text-white text-xs font-bold">AI</div>
                                    </div>
                                    <div>
                                        <CardTitle className="text-sm">Insight Assistant</CardTitle>
                                        <p className="text-[10px] text-muted">Powered by LLM</p>
                                    </div>
                                </div>
                                <Button variant="ghost" size="sm" onClick={() => setIsOpen(false)} className="h-8 w-8 p-0">
                                    ✕
                                </Button>
                            </div>
                        </CardHeader>
                        <CardContent className="p-0 flex-1 overflow-y-auto min-h-[300px] max-h-[400px]">
                            <div className="p-4 space-y-4">
                                <div className="flex gap-3">
                                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex-shrink-0 flex items-center justify-center mt-1">
                                        <span className="text-white text-xs">AI</span>
                                    </div>
                                    <div className="bg-white/5 rounded-2xl rounded-tl-none p-3 text-sm border border-white/10">
                                        <p>Hello! I've analyzed your latest forecast data. Here are my key findings and suggestions.</p>
                                    </div>
                                </div>

                                {loading && (
                                    <div className="flex gap-3">
                                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex-shrink-0 flex items-center justify-center mt-1">
                                            <span className="text-white text-xs">AI</span>
                                        </div>
                                        <div className="bg-white/5 rounded-2xl rounded-tl-none p-3 text-sm border border-white/10">
                                            <div className="flex gap-1">
                                                <span className="w-2 h-2 bg-secondary rounded-full animate-bounce"></span>
                                                <span className="w-2 h-2 bg-secondary rounded-full animate-bounce delay-100"></span>
                                                <span className="w-2 h-2 bg-secondary rounded-full animate-bounce delay-200"></span>
                                            </div>
                                        </div>
                                    </div>
                                )}

                                {insights.map((insight, idx) => (
                                    <div key={idx} className="flex gap-3 animate-in fade-in slide-in-from-bottom-2 duration-500">
                                        <div className="w-8 h-8 rounded-full bg-gradient-to-br from-indigo-500 to-purple-600 flex-shrink-0 flex items-center justify-center mt-1">
                                            <span className="text-white text-xs">AI</span>
                                        </div>
                                        <div className={`rounded-2xl rounded-tl-none p-3 text-sm border shadow-sm ${insight.type === 'warning' ? 'bg-error/20 border-error/30 text-white' :
                                                insight.type === 'suggestion' ? 'bg-secondary/20 border-secondary/30 text-white' :
                                                    'bg-white/10 border-white/20 text-white'
                                            }`}>
                                            <div className="flex items-start gap-2">
                                                {insight.type === 'warning' && <AlertIcon size={16} className="text-error mt-0.5" />}
                                                {insight.type === 'suggestion' && <TrendingUpIcon size={16} className="text-secondary mt-0.5" />}
                                                {insight.type === 'analysis' && <CheckIcon size={16} className="text-success mt-0.5" />}
                                                <p>{insight.content}</p>
                                            </div>
                                        </div>
                                    </div>
                                ))}
                                <div ref={chatEndRef} />
                            </div>
                        </CardContent>
                        <div className="p-3 border-t border-white/10 bg-surface-elevated/50 backdrop-blur-sm">
                            <Button
                                variant="secondary"
                                className="w-full text-xs"
                                onClick={generateInsights}
                                disabled={loading}
                            >
                                <RefreshIcon size={14} className={loading ? "animate-spin" : ""} />
                                {loading ? 'Analyzing...' : 'Regenerate Insights'}
                            </Button>
                        </div>
                    </Card>
                </div>
            )}

            {/* Floating Trigger Button */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-14 h-14 bg-gradient-to-r from-indigo-600 to-purple-600 rounded-full shadow-lg shadow-purple-500/30 flex items-center justify-center hover:scale-110 active:scale-95 transition-all pointer-events-auto"
            >
                {isOpen ? (
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                ) : (
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                        <line x1="9" y1="10" x2="15" y2="10" />
                        <line x1="12" y1="7" x2="12" y2="13" />
                    </svg>
                )}
            </button>
        </div>
    );
}
