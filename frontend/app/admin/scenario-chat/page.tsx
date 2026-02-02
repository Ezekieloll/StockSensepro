'use client';

import { useState } from 'react';
import Card from '@/components/ui/Card';
import Button from '@/components/ui/Button';

const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function InteractiveScenarioAnalyst() {
  const [scenario, setScenario] = useState('');
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState('');

  const exampleScenarios = [
    "Tomorrow there will be a citywide lockdown due to health emergency",
    "Weather forecast shows heavy snowstorm for next 3 days",
    "News: 15% tax increase on all grocery items from next week",
    "Competitor store in S1 area announced closure next Monday",
    "Local festival next weekend expected to bring 50% more shoppers"
  ];

  const analyzeScenario = async () => {
    if (!scenario.trim()) {
      setError('Please enter a scenario');
      return;
    }

    console.log('Starting analysis...', scenario);
    setAnalyzing(true);
    setError('');
    setResult(null);

    try {
      console.log('Sending request to:', `${API_BASE}/llm/analyze-scenario`);
      const res = await fetch(`${API_BASE}/llm/analyze-scenario`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scenario: scenario.trim() })
      });

      console.log('Response status:', res.status);
      
      if (res.ok) {
        const data = await res.json();
        console.log('Analysis result:', data);
        setResult(data);
      } else {
        const errData = await res.json();
        console.error('Error response:', errData);
        setError(errData.detail || errData.message || 'Analysis failed');
      }
    } catch (err) {
      console.error('Fetch error:', err);
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(`Failed to connect to AI service: ${errorMessage}`);
    } finally {
      setAnalyzing(false);
    }
  };

  const getImpactColor = (level: string) => {
    switch (level?.toLowerCase()) {
      case 'critical': return 'text-red-600 bg-red-50';
      case 'high': return 'text-orange-600 bg-orange-50';
      case 'medium': return 'text-yellow-600 bg-yellow-50';
      default: return 'text-blue-600 bg-blue-50';
    }
  };

  const getActionColor = (action: string) => {
    switch (action?.toUpperCase()) {
      case 'URGENT': return 'bg-red-100 text-red-800 border-red-300';
      case 'MODERATE': return 'bg-orange-100 text-orange-800 border-orange-300';
      default: return 'bg-blue-100 text-blue-800 border-blue-300';
    }
  };

  return (
    <div className="space-y-6 max-w-6xl mx-auto p-6">
      <div>
        <h1 className="text-3xl font-bold">💬 Interactive Scenario Analyst</h1>
        <p className="text-gray-600 mt-2">
          Describe any business scenario in plain English. AI will analyze your database and calculate exact impact on your stores.
        </p>
      </div>

      {/* Input Section */}
      <Card>
        <h2 className="text-lg font-semibold mb-4">📝 Describe Your Scenario</h2>
        
        <textarea
          value={scenario}
          onChange={(e) => setScenario(e.target.value)}
          placeholder="Example: Tomorrow there will be a lockdown..."
          className="w-full p-4 border rounded-lg min-h-[120px] focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          disabled={analyzing}
        />

        <div className="mt-4 flex gap-3">
          <Button
            onClick={analyzeScenario}
            disabled={analyzing || !scenario.trim()}
            className="bg-blue-600 hover:bg-blue-700"
          >
            {analyzing ? '🔄 Analyzing with AI...' : '🚀 Analyze Impact'}
          </Button>
          <Button
            onClick={() => setScenario('')}
            disabled={analyzing}
            className="bg-gray-500 hover:bg-gray-600"
          >
            Clear
          </Button>
        </div>

        <div className="mt-4">
          <p className="text-sm font-semibold text-gray-700 mb-2">💡 Try these examples:</p>
          <div className="space-y-2">
            {exampleScenarios.map((ex, idx) => (
              <button
                key={idx}
                onClick={() => setScenario(ex)}
                className="block w-full text-left text-sm text-gray-600 hover:text-blue-600 hover:bg-blue-50 p-2 rounded transition-colors"
                disabled={analyzing}
              >
                "{ex}"
              </button>
            ))}
          </div>
        </div>
      </Card>

      {/* Error Display */}
      {error && (
        <Card className="bg-red-50 border-red-200">
          <p className="text-red-700">❌ {error}</p>
        </Card>
      )}

      {/* Results Display */}
      {result && result.status === 'success' && (
        <Card className="bg-white">
          <h2 className="text-2xl font-bold mb-6">🤖 AI Analysis</h2>
          
          <div className="prose max-w-none">
            <div className="text-gray-800 whitespace-pre-wrap leading-relaxed text-lg">
              {result.analysis?.scenario_summary || result.analysis?.raw_response || 'No analysis available'}
            </div>
          </div>
        </Card>
      )}
    </div>
  );
}
