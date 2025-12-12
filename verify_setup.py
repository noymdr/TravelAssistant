"""
Setup verification script for Travel RAG System
Checks if all dependencies and services are properly configured.
"""

import sys
import subprocess


def check_python_version():
    """Check Python version."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"❌ {package_name} is not installed. Run: pip install {package_name}")
        return False


def check_ollama():
    """Check if Ollama is running and has llama3 model."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "") for model in models]
            if any("llama3" in name for name in model_names):
                print("✓ Ollama is running and llama3 model is available")
                return True
            else:
                print("⚠ Ollama is running but llama3 model not found. Run: ollama pull llama3")
                return False
        else:
            print("❌ Ollama is not responding correctly")
            return False
    except ImportError:
        print("⚠ requests module not available, skipping Ollama check")
        return None
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("   Make sure Ollama is running: ollama serve")
        return False


def check_documents_dir():
    """Check if documents directory exists and has files."""
    from pathlib import Path
    docs_dir = Path("documents")
    if not docs_dir.exists():
        print("⚠ documents/ directory does not exist (will be created automatically)")
        return True
    
    pdf_files = list(docs_dir.glob("*.pdf"))
    txt_files = list(docs_dir.glob("*.txt"))
    total = len(pdf_files) + len(txt_files)
    
    if total > 0:
        print(f"✓ Found {total} document(s) in documents/ directory")
        return True
    else:
        print("⚠ No documents found in documents/ directory")
        print("   Add your travel documents (PDF or TXT files) to the documents/ folder")
        return True


def main():
    """Run all checks."""
    print("=" * 60)
    print("Travel RAG System - Setup Verification")
    print("=" * 60)
    print()
    
    checks = []
    
    # Python version
    checks.append(check_python_version())
    print()
    
    # Required packages
    print("Checking required packages...")
    checks.append(check_import("langchain", "langchain"))
    checks.append(check_import("langchain_community", "langchain-community"))
    checks.append(check_import("langchain_ollama", "langchain-ollama"))
    checks.append(check_import("faiss", "faiss-cpu"))
    checks.append(check_import("sentence_transformers", "sentence-transformers"))
    checks.append(check_import("pypdf", "pypdf"))
    print()
    
    # Ollama check
    print("Checking Ollama...")
    ollama_check = check_ollama()
    if ollama_check is not None:
        checks.append(ollama_check)
    print()
    
    # Documents directory
    print("Checking documents directory...")
    check_documents_dir()
    print()
    
    # Summary
    print("=" * 60)
    passed = sum(1 for c in checks if c is True)
    total = len([c for c in checks if c is not None])
    
    if passed == total:
        print("✓ All checks passed! You're ready to use the RAG system.")
        print("\nNext steps:")
        print("1. Add your travel documents to the documents/ folder")
        print("2. Run: python example_usage.py")
    else:
        print(f"⚠ {passed}/{total} checks passed. Please fix the issues above.")
    print("=" * 60)


if __name__ == "__main__":
    main()

