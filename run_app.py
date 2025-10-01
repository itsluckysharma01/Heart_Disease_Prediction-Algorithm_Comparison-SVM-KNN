#!/usr/bin/env python3
"""
Heart Disease Prediction Web App Runner
This script sets up and runs the Flask web application
"""

import os
import sys
import subprocess
import platform

def print_banner():
    print("\n" + "="*50)
    print("    Heart Disease Prediction Web App")
    print("="*50 + "\n")

def check_python():
    """Check if Python is properly installed"""
    try:
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("❌ Python 3.8 or higher is required")
            print(f"   Current version: {version.major}.{version.minor}")
            return False
        print("✅ Python version check passed")
        return True
    except Exception as e:
        print(f"❌ Error checking Python version: {e}")
        return False

def setup_virtual_environment():
    """Create and activate virtual environment"""
    if not os.path.exists("venv"):
        print("📦 Creating virtual environment...")
        try:
            subprocess.run([sys.executable, "-m", "venv", "venv"], check=True)
            print("✅ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create virtual environment: {e}")
            return False
    else:
        print("✅ Virtual environment already exists")
    
    return True

def get_pip_command():
    """Get the correct pip command for the platform"""
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "pip.exe")
    else:
        return os.path.join("venv", "bin", "pip")

def get_python_command():
    """Get the correct python command for the platform"""
    if platform.system() == "Windows":
        return os.path.join("venv", "Scripts", "python.exe")
    else:
        return os.path.join("venv", "bin", "python")

def install_requirements():
    """Install required packages"""
    print("📋 Installing dependencies...")
    try:
        pip_cmd = get_pip_command()
        subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], check=True)
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def check_model_files():
    """Check if model files exist"""
    model_files = [
        "heart_disease_svm_model.pkl",
        "heart_disease_knn_model.pkl"
    ]
    
    missing_files = []
    for file in model_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("⚠️  Warning: Missing model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("   You may need to train the models first using the Jupyter notebook")
        return False
    else:
        print("✅ Model files found")
        return True

def start_application():
    """Start the Flask application"""
    print("\n" + "="*50)
    print("    Starting Heart Disease Prediction App")
    print("="*50)
    print("\n🌐 Open your browser and go to: http://127.0.0.1:5000")
    print("🛑 Press Ctrl+C to stop the server\n")
    
    try:
        python_cmd = get_python_command()
        subprocess.run([python_cmd, "app.py"])
    except KeyboardInterrupt:
        print("\n\n🛑 Application stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")

def main():
    """Main function to run the setup and start the app"""
    print_banner()
    
    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Check Python version
    if not check_python():
        sys.exit(1)
    
    # Setup virtual environment
    if not setup_virtual_environment():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        sys.exit(1)
    
    # Check model files
    check_model_files()
    
    # Start the application
    start_application()

if __name__ == "__main__":
    main()