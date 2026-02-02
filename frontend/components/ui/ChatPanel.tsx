'use client';

import { useState, useRef, useEffect } from 'react';
import Button from './Button';
import { XIcon } from './Icons';

interface Message {
    role: 'user' | 'assistant';
    content: string;
    timestamp: Date;
}

interface ChatPanelProps {
    isOpen: boolean;
    onClose: () => void;
    onScenarioAnalyzed?: (result: any) => void;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function ChatPanel({ isOpen, onClose, onScenarioAnalyzed }: ChatPanelProps) {
    const [messages, setMessages] = useState<Message[]>([
        {
            role: 'assistant',
            content: "Hi! I'm your demand forecasting assistant. Tell me about a scenario you want to explore, and I'll help you understand its impact on demand. For example, you could say:\n\n• \"What if there's a major snowstorm next week?\"\n• \"A competitor nearby just closed down\"\n• \"Economic recession is predicted\"",
            timestamp: new Date()
        }
    ]);
    const [input, setInput] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const [baseline, setBaseline] = useState<number | null>(null);
    const [categories, setCategories] = useState<Record<string, string>>({});
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    // Fetch baseline on mount
    useEffect(() => {
        const fetchBaseline = async () => {
            try {
                const response = await fetch(`${API_URL}/simulations/baseline`);
                if (response.ok) {
                    const data = await response.json();
                    setBaseline(data.avg_demand);
                }
            } catch (error) {
                console.error('Error fetching baseline:', error);
            }
        };
        
        const fetchCategories = async () => {
            try {
                const response = await fetch(`${API_URL}/gnn/category-summary`);
                if (response.ok) {
                    const data = await response.json();
                    // Convert to simple name mapping
                    const nameMap: Record<string, string> = {};
                    Object.keys(data).forEach(key => {
                        nameMap[key] = data[key].name;
                    });
                    setCategories(nameMap);
                }
            } catch (error) {
                console.error('Error fetching categories:', error);
                // Fallback to basic names
                setCategories({
                    "FRPR": "Fresh Produce",
                    "BKDY": "Bakery",
                    "BEVR": "Beverages",
                    "SNCK": "Snacks",
                    "FTRW": "Furniture",
                    "SPRT": "Sports"
                });
            }
        };
        
        fetchBaseline();
        fetchCategories();
    }, []);

    const analyzeScenario = async (scenarioText: string) => {
        try {
            const response = await fetch(`${API_URL}/simulations/custom?scenario_text=${encodeURIComponent(scenarioText)}`, {
                method: 'POST',
            });

            if (response.ok) {
                const result = await response.json();
                return result;
            }
        } catch (error) {
            console.error('Error analyzing scenario:', error);
        }
        return null;
    };

    const handleSend = async () => {
        if (!input.trim()) return;

        const userMessage: Message = {
            role: 'user',
            content: input,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, userMessage]);
        setInput('');
        setIsTyping(true);

        // Analyze the scenario
        const result = await analyzeScenario(input);

        let responseText = '';
        
        if (result) {
            // Notify parent component
            if (onScenarioAnalyzed) {
                onScenarioAnalyzed(result);
            }

            // Create conversational response
            const baselineDemand = baseline || result.demand; // Fallback to result demand if baseline not loaded
            const demandChange = ((result.demand / baselineDemand) - 1) * 100;
            const direction = demandChange > 0 ? 'increase' : 'decrease';
            const changeAmount = Math.abs(demandChange).toFixed(0);

            responseText = `Based on your scenario, here's what I predict:\n\n`;
            responseText += `📊 **Projected Demand**: ${result.demand.toLocaleString()} units per day\n\n`;
            
            if (result.ai_reasoning) {
                responseText += `💡 **Why**: ${result.ai_reasoning}\n\n`;
            }
            
            // Show category-specific impacts if available
            if (result.affected_categories && result.affected_categories.length > 0) {
                responseText += `🎯 **Affected Categories**:\n`;
                
                result.affected_categories.forEach((cat: string) => {
                    const impact = result.category_impacts?.[cat];
                    const change = impact ? ((impact - 1) * 100).toFixed(0) : "0";
                    const direction = parseFloat(change) > 0 ? "+" : "";
                    const catName = categories[cat] || cat;
                    responseText += `   • ${catName}: ${direction}${change}%\n`;
                });
                responseText += `\n`;
            } else {
                responseText += `🌐 **Impact**: Affects all product categories\n\n`;
            }
            
            // Show top product-level propagation impacts from GNN
            if (result.affected_products && Object.keys(result.affected_products).length > 0) {
                const productList = Object.entries(result.affected_products)
                    .map(([sku, mult]: [string, any]) => ({
                        sku,
                        name: typeof mult === 'object' && mult.name ? mult.name : sku,
                        mult: typeof mult === 'object' && mult.multiplier ? mult.multiplier : (typeof mult === 'number' ? mult : 1.0),
                        change: (() => {
                            const m = typeof mult === 'object' && mult.multiplier ? mult.multiplier : (typeof mult === 'number' ? mult : 1.0);
                            return ((m - 1) * 100).toFixed(1);
                        })()
                    }))
                    .sort((a, b) => Math.abs(b.mult - 1) - Math.abs(a.mult - 1))
                    .slice(0, 8);  // Show top 8 most affected products
                
                if (productList.length > 0) {
                    responseText += `🔗 **GNN Propagated Impacts** (Product Level):\n`;
                    productList.forEach(({ name, change }) => {
                        const dir = parseFloat(change) >= 0 ? "+" : "";
                        responseText += `   • ${name}: ${dir}${change}%\n`;
                    });
                    responseText += `\n`;
                }
            }
            
            responseText += `📈 **Overall Change**: ${direction} by about ${changeAmount}%\n\n`;
            responseText += `⚠️ **Risk Level**: ${result.risk.charAt(0).toUpperCase() + result.risk.slice(1)}\n\n`;
            responseText += `🎯 **Confidence**: ${result.confidence}%\n\n`;
            
            if (result.risk === 'high') {
                responseText += `This is a high-risk scenario. You should prepare by increasing safety stock and ensuring supplier capacity.`;
            } else if (result.risk === 'medium') {
                responseText += `Moderate risk. Monitor the situation closely and be ready to adjust inventory levels.`;
            } else {
                responseText += `Low risk impact. Your current inventory strategy should handle this comfortably.`;
            }

            responseText += `\n\nWant to explore another scenario? Just ask!`;
        } else {
            responseText = "I'm having trouble analyzing that scenario right now. Could you rephrase it or try a different scenario?";
        }

        setIsTyping(false);
        
        const assistantMessage: Message = {
            role: 'assistant',
            content: responseText,
            timestamp: new Date()
        };

        setMessages(prev => [...prev, assistantMessage]);
    };

    const handleKeyPress = (e: React.KeyboardEvent) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    if (!isOpen) return null;

    return (
        <div className="fixed bottom-6 right-6 w-96 h-[600px] glass border border-white/10 rounded-2xl shadow-2xl flex flex-col z-50 animate-in slide-in-from-bottom-4 duration-300">
            {/* Header */}
            <div className="flex items-center justify-between p-4 border-b border-white/10">
                <div>
                    <h3 className="font-bold text-lg gradient-text">AI Assistant</h3>
                    <p className="text-xs text-muted">Demand Scenario Analysis</p>
                </div>
                <button
                    onClick={onClose}
                    className="p-2 hover:bg-white/10 rounded-lg transition-colors"
                >
                    <XIcon size={18} />
                </button>
            </div>

            {/* Messages */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map((msg, idx) => (
                    <div
                        key={idx}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                        <div
                            className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                                msg.role === 'user'
                                    ? 'bg-gradient-to-r from-primary to-info text-white'
                                    : 'bg-surface-elevated border border-white/10'
                            }`}
                        >
                            <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
                            <div className={`text-xs mt-1 ${msg.role === 'user' ? 'text-white/70' : 'text-muted'}`}>
                                {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                            </div>
                        </div>
                    </div>
                ))}
                
                {isTyping && (
                    <div className="flex justify-start">
                        <div className="bg-surface-elevated border border-white/10 rounded-2xl px-4 py-3">
                            <div className="flex gap-1">
                                <span className="w-2 h-2 bg-muted rounded-full animate-bounce"></span>
                                <span className="w-2 h-2 bg-muted rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></span>
                                <span className="w-2 h-2 bg-muted rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></span>
                            </div>
                        </div>
                    </div>
                )}
                
                <div ref={messagesEndRef} />
            </div>

            {/* Input */}
            <div className="p-4 border-t border-white/10">
                <div className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={handleKeyPress}
                        placeholder="Describe a scenario..."
                        className="flex-1 bg-surface-elevated border border-white/10 rounded-lg px-4 py-2 text-sm focus:ring-1 focus:ring-primary outline-none"
                        disabled={isTyping}
                    />
                    <Button
                        variant="primary"
                        onClick={handleSend}
                        disabled={!input.trim() || isTyping}
                    >
                        Send
                    </Button>
                </div>
                <p className="text-xs text-muted mt-2">
                    💬 Ask about weather, competitors, holidays, economy, etc.
                </p>
            </div>
        </div>
    );
}
