# Local LLM Setup Guide

## 🚀 Quick Start (3 Steps)

### Step 1: Install Ollama
```powershell
# Download and install from:
# https://ollama.com/download/windows

# Or use winget:
winget install Ollama.Ollama
```

### Step 2: Download the AI Model
```powershell
# Recommended: Qwen 2.5 14B (best for database analysis)
ollama pull qwen2.5:14b

# Alternative options:
# ollama pull qwen2.5:7b      # Faster, uses less RAM
# ollama pull llama3.2:3b     # Very lightweight
# ollama pull qwen2.5:32b     # Most powerful (needs GPU)
```

**⚠️ Important:** Download is ~8.5GB and takes 10-20 minutes
- ✅ Screen can go black - download continues
- ❌ Computer sleep interrupts download
- ✅ If interrupted, re-run command to resume from where it stopped
- 💡 Our setup script automatically prevents sleep during download

### Step 3: Start Using It
```powershell
# Test in terminal:
ollama run qwen2.5:14b "Explain inventory turnover ratio"

# Or use our Python setup:
cd C:\StockSense\ml\llm
python local_llm_setup.py

# Test the integration:
python test_llm.py
```

---

## 📦 Model Comparison

| Model | Size | RAM | Speed (CPU) | Best For |
|-------|------|-----|-------------|----------|
| **qwen2.5:14b** ⭐ | 8.5GB | 16GB | 10-20 tok/s | Database analysis, scenarios, chatbot |
| qwen2.5:7b | 4.7GB | 8GB | 20-30 tok/s | Faster responses, good reasoning |
| llama3.2:3b | 2GB | 4GB | 40+ tok/s | Quick answers, basic chat |
| qwen2.5:32b | 19GB | 32GB | GPU only | Complex analysis (overkill for most) |

**⭐ Recommended: qwen2.5:14b** - Best balance for StockSense

---

## 🔌 API Usage

### 1. Chat with Database Context
```javascript
// Frontend example
const response = await fetch('http://localhost:8000/llm/chat', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: "Which SKUs are at highest risk?",
    store_id: "S1",
    context: "risks"  // Auto-pulls risk data
  })
});

const data = await response.json();
console.log(data.response);  // AI-generated answer
```

### 2. Generate Custom Scenarios
```javascript
const scenario = await fetch('http://localhost:8000/llm/generate-scenario', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    sku: "SKU001",
    store_id: "S1",
    custom_prompt: "Generate a scenario for a local festival event"
  })
});

const scenarioData = await scenario.json();
// Returns: demand_multiplier, strategies, probability, etc.
```

### 3. General Supply Chain Chat
```python
# Python example
from llm.ollama_client import OllamaClient

client = OllamaClient(model_name="qwen2.5:14b")

# Single question
answer = client.generate("What is safety stock?")

# Conversation
client.chat("Hello, I need help with forecasting", reset_history=True)
client.chat("What's the difference between MAE and MAPE?")
```

---

## 🎯 Use Cases in StockSense

### 1. **Intelligent Scenario Generator**
- Ask: "Generate a scenario for milk during summer heatwave"
- AI creates: demand multiplier, duration, strategies

### 2. **Database Query Assistant**
- Ask: "Show me SKUs with less than 3 days of inventory"
- AI analyzes data and explains findings

### 3. **Chatbot for InsightAssistant**
- Upgrade the existing chatbot to use LLM
- Natural language queries about forecasts/inventory
- Contextual recommendations

### 4. **Automated Report Generation**
- "Summarize this week's high-risk SKUs"
- "Explain why SKU042 has high stockout probability"

---

## 🛠️ Integration Files

1. **ml/llm/local_llm_setup.py** - Setup wizard
2. **ml/llm/ollama_client.py** - Python client for Ollama
3. **backend/app/api/llm_chat.py** - REST API endpoints
4. **ml/llm/test_llm.py** - Test suite

---

## ⚡ Performance Tips

### CPU Optimization
```powershell
# Set threads for better CPU performance
$env:OLLAMA_NUM_THREADS = "8"  # Adjust based on your CPU cores
ollama serve
```

### GPU Acceleration (if you have NVIDIA GPU)
```powershell
# Ollama automatically uses GPU if CUDA is available
# Check GPU usage:
nvidia-smi
```

### Memory Management
```powershell
# Reduce model size in memory (uses less RAM but slower)
ollama run qwen2.5:14b --num-gpu 0  # Force CPU only
```

---

## 🔍 Troubleshooting

### "Ollama is not available"
```powershell
# Check if Ollama is running:
ollama --version

# Start Ollama service:
ollama serve

# Test connection:
curl http://localhost:11434/api/tags
```

### "Model not found"
```powershell
# List installed models:
ollama list

# Download missing model (prevents sleep automatically via our script):
cd C:\StockSense\ml\llm
.\setup_local_ai.bat

# Or manually (might be interrupted by sleep):
ollama pull qwen2.5:14b
```

### "Download interrupted by sleep"
```powershell
# Ollama can resume! Just run again:
ollama pull qwen2.5:14b

# Or prevent sleep manually before downloading:
powercfg /change standby-timeout-ac 0
ollama pull qwen2.5:14b
powercfg /change standby-timeout-ac 30  # Restore after
```

### Slow Responses
- Use smaller model: `qwen2.5:7b` instead of `14b`
- Reduce max_tokens in API calls
- Enable GPU if available

---

## 📚 Next Steps

1. **Install Ollama and download model**
   ```powershell
   ollama pull qwen2.5:14b
   ```

2. **Test the integration**
   ```powershell
   cd C:\StockSense\ml\llm
   python test_llm.py
   ```

3. **Upgrade InsightAssistant**
   - Replace mock data with LLM-powered responses
   - Enable natural language database queries

4. **Enable custom scenarios**
   - Add "Generate Custom Scenario" button in admin panel
   - Let users describe scenarios in plain English

**The LLM will understand your database and provide intelligent insights! 🤖**
