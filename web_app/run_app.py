#!/usr/bin/env python3
"""
Script untuk menjalankan aplikasi Sistem Prediksi Obesitas
- Backend FastAPI di port 8000
- Frontend Streamlit di port 8501
"""

import subprocess
import threading
import time
import sys
import os
import webbrowser
from datetime import datetime

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║        🏥 SISTEM PREDIKSI TINGKAT OBESITAS 🏥                ║
    ║                                                              ║
    ║              AI-Powered Obesity Risk Assessment              ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)
    print(f"🕒 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 66)

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'streamlit', 'pandas', 
        'numpy', 'plotly', 'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("✅ All dependencies are installed")
    return True

def run_backend():
    """Run FastAPI backend"""
    try:
        print("🚀 Starting Backend (FastAPI)...")
        print("📍 Backend URL: http://localhost:8000")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("🔄 Interactive API: http://localhost:8000/redoc")
        print("-" * 50)
        
        # Run FastAPI with uvicorn
        subprocess.run([
            sys.executable, "-m", "uvicorn", "app:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--log-level", "info"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Backend stopped by user")
    except Exception as e:
        print(f"❌ Error running backend: {e}")

def run_frontend():
    """Run Streamlit frontend"""
    try:
        # Wait for backend to start
        print("⏳ Waiting for backend to start...")
        time.sleep(8)
        
        print("🚀 Starting Frontend (Streamlit)...")
        print("📍 Frontend URL: http://localhost:8501")
        print("-" * 50)
        
        # Run Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "frontend.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Frontend stopped by user")
    except Exception as e:
        print(f"❌ Error running frontend: {e}")

def check_files():
    """Check if required files exist"""
    required_files = ['app.py', 'frontend.py']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found")
    return True

def open_browser():
    """Open browser tabs for the application"""
    try:
        # Wait a bit more for services to be ready
        time.sleep(15)
        
        print("🌐 Opening browser tabs...")
        
        # Open frontend
        webbrowser.open('http://localhost:8501')
        time.sleep(2)
        
        # Open API docs
        webbrowser.open('http://localhost:8000/docs')
        
    except Exception as e:
        print(f"⚠️  Could not open browser automatically: {e}")
        print("Please manually open:")
        print("   - Frontend: http://localhost:8501")
        print("   - API Docs: http://localhost:8000/docs")

def main():
    """Main function to run the application"""
    print_banner()
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        print("\n❌ Please install missing dependencies and try again")
        return
    
    # Check required files
    print("📁 Checking required files...")
    if not check_files():
        print("\n❌ Please ensure all required files are present")
        return
    
    print("\n🚀 Starting application services...")
    print("💡 Use Ctrl+C to stop all services")
    print("=" * 66)
    
    try:
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # Start browser opener in a separate thread
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        # Start frontend in main thread
        run_frontend()
        
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
        print("=" * 66)
        print("Thank you for using Obesity Prediction System!")
        print("=" * 66)
    except Exception as e:
        print(f"\n❌ Error running application: {e}")
    finally:
        print(f"\n🕒 Stopped at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

def show_help():
    """Show help information"""
    help_text = """
🏥 Obesity Prediction System - Help

📋 Commands:
  python run_app.py          - Start both backend and frontend
  python run_app.py --help   - Show this help message
  python run_app.py --check  - Check dependencies only
  
🔧 Manual Setup:
  1. Backend only:  python app.py
  2. Frontend only: streamlit run frontend.py
  
📍 URLs:
  - Frontend:    http://localhost:8501
  - Backend API: http://localhost:8000
  - API Docs:    http://localhost:8000/docs
  
🐛 Troubleshooting:
  - Port conflicts: Change ports in the scripts
  - Missing packages: pip install -r requirements.txt
  - Permission issues: Run as administrator (Windows) or with sudo (Linux/Mac)
  
📞 Support:
  - Check console output for error messages
  - Ensure ports 8000 and 8501 are available
  - Verify all dependencies are installed
"""
    print(help_text)

if __name__ == "__main__":
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ['--help', '-h']:
            show_help()
        elif sys.argv[1] == '--check':
            print_banner()
            print("🔍 Checking system requirements...")
            if check_dependencies() and check_files():
                print("✅ System is ready to run!")
            else:
                print("❌ System checks failed")
        else:
            print(f"Unknown argument: {sys.argv[1]}")
            print("Use --help for available commands")
    else:
        # Run the main application
        main() 