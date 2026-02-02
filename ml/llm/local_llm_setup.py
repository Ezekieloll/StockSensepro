"""
Local LLM Setup and Integration for StockSense
Uses Ollama to run Qwen 2.5 14B locally for:
- Database analysis
- Scenario generation
- Chatbot functionality
"""

import subprocess
import sys
import json
from pathlib import Path


def check_ollama_installed():
    """Check if Ollama is installed"""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✅ Ollama is installed: {result.stdout.strip()}")
            return True
        return False
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"Error checking Ollama: {e}")
        return False


def install_ollama_instructions():
    """Provide instructions to install Ollama"""
    print("\n" + "="*80)
    print("📥 OLLAMA NOT FOUND - Installation Instructions")
    print("="*80)
    print("\n1. Download Ollama for Windows:")
    print("   https://ollama.com/download/windows")
    print("\n2. Run the installer (OllamaSetup.exe)")
    print("\n3. Restart your terminal")
    print("\n4. Run this script again")
    print("\n" + "="*80)


def pull_model(model_name="qwen2.5:14b"):
    """Download the recommended model"""
    print(f"\n📦 Downloading {model_name}...")
    print("This may take 10-20 minutes depending on your internet speed...")
    print("Model size: ~8.5GB\n")
    
    try:
        result = subprocess.run(
            ["ollama", "pull", model_name],
            text=True,
            timeout=1800  # 30 minute timeout
        )
        
        if result.returncode == 0:
            print(f"\n✅ Model {model_name} downloaded successfully!")
            return True
        else:
            print(f"\n❌ Failed to download model")
            return False
    except subprocess.TimeoutExpired:
        print("\n⚠️ Download timed out. Please try again with a better connection.")
        return False
    except Exception as e:
        print(f"\n❌ Error downloading model: {e}")
        return False


def list_available_models():
    """List all downloaded models"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True
        )
        print("\n📋 Available Models:")
        print(result.stdout)
        return result.stdout
    except Exception as e:
        print(f"Error listing models: {e}")
        return ""


def test_model(model_name="qwen2.5:14b"):
    """Test the model with a sample query"""
    print(f"\n🧪 Testing {model_name}...")
    
    test_prompt = """You are a supply chain analyst. Analyze this scenario:
    
Current inventory: 100 units of milk
Average daily demand: 15 units
Upcoming holiday: Valentine's Day (3 days away)

What risks do you see and what should we do?"""

    try:
        result = subprocess.run(
            ["ollama", "run", model_name, test_prompt],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0:
            print("\n✅ Model Response:")
            print("-" * 80)
            print(result.stdout)
            print("-" * 80)
            return True
        else:
            print(f"❌ Model test failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("⚠️ Model response timed out")
        return False
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        return False


def setup_recommendations():
    """Provide setup recommendations"""
    print("\n" + "="*80)
    print("🎯 RECOMMENDED MODEL OPTIONS FOR YOUR HARDWARE")
    print("="*80)
    
    print("\n1. 🏆 RECOMMENDED: qwen2.5:14b (Best for your use case)")
    print("   - Size: 8.5GB download, ~16GB RAM needed")
    print("   - Best at: Database analysis, structured reasoning, chatbot")
    print("   - Speed: ~10-20 tokens/sec on CPU, ~50+ on GPU")
    print("   - Download: ollama pull qwen2.5:14b")
    
    print("\n2. 💎 ALTERNATIVE: qwen2.5:7b (If 14B is too slow)")
    print("   - Size: 4.7GB download, ~8GB RAM needed")
    print("   - Best at: Fast responses, still good reasoning")
    print("   - Speed: ~20-30 tokens/sec on CPU")
    print("   - Download: ollama pull qwen2.5:7b")
    
    print("\n3. 🚀 LIGHTWEIGHT: llama3.2:3b (If you have <8GB RAM)")
    print("   - Size: 2GB download, ~4GB RAM needed")
    print("   - Best at: Quick answers, basic chatbot")
    print("   - Speed: ~40+ tokens/sec on CPU")
    print("   - Download: ollama pull llama3.2:3b")
    
    print("\n4. 🧠 ADVANCED: qwen2.5:32b (If you have 32GB+ RAM and GPU)")
    print("   - Size: 19GB download, ~32GB RAM needed")
    print("   - Best at: Complex analysis, advanced reasoning")
    print("   - Speed: Needs GPU for practical use")
    print("   - Download: ollama pull qwen2.5:32b")
    
    print("\n" + "="*80)
    print("💡 For StockSense, we recommend starting with qwen2.5:14b")
    print("   It's the best balance of capability and performance.")
    print("="*80 + "\n")


def main():
    """Main setup flow"""
    print("\n🤖 StockSense Local LLM Setup")
    print("="*80)
    
    # Check Ollama installation
    if not check_ollama_installed():
        install_ollama_instructions()
        return
    
    # Show available models
    list_available_models()
    
    # Show recommendations
    setup_recommendations()
    
    # Ask user what to do
    print("\nWhat would you like to do?")
    print("1. Download qwen2.5:14b (Recommended)")
    print("2. Download qwen2.5:7b (Faster alternative)")
    print("3. Download llama3.2:3b (Lightweight)")
    print("4. Test existing model")
    print("5. Exit")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    if choice == "1":
        if pull_model("qwen2.5:14b"):
            test_model("qwen2.5:14b")
    elif choice == "2":
        if pull_model("qwen2.5:7b"):
            test_model("qwen2.5:7b")
    elif choice == "3":
        if pull_model("llama3.2:3b"):
            test_model("llama3.2:3b")
    elif choice == "4":
        model = input("Enter model name (e.g., qwen2.5:14b): ").strip()
        test_model(model)
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
