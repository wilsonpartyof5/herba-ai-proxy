#!/usr/bin/env python3

"""
Configuration Validator for Herba2 Backend Deployment
Run this script before deployment to ensure correct configuration
"""

import json
import sys
import os
from typing import List, Dict, Any

def validate_vercel_config() -> bool:
    """Validate vercel.json configuration"""
    print("🔧 Validating vercel.json...")
    
    try:
        with open('vercel.json', 'r') as f:
            config = json.load(f)
        
        # Check if using main_fixed.py
        if 'builds' not in config or not config['builds']:
            print("❌ No builds section in vercel.json")
            return False
        
        build_src = config['builds'][0].get('src', '')
        if build_src != 'main_fixed.py':
            print(f"❌ vercel.json pointing to '{build_src}' instead of 'main_fixed.py'")
            return False
        
        # Check routes configuration
        if 'routes' not in config or not config['routes']:
            print("❌ No routes section in vercel.json")
            return False
        
        route_dest = config['routes'][0].get('dest', '')
        if route_dest != 'main_fixed.py':
            print(f"❌ Route destination is '{route_dest}' instead of 'main_fixed.py'")
            return False
        
        print("✅ Vercel config valid - using main_fixed.py")
        return True
        
    except FileNotFoundError:
        print("❌ vercel.json not found")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in vercel.json: {e}")
        return False
    except Exception as e:
        print(f"❌ Error reading vercel.json: {e}")
        return False

def validate_requirements() -> bool:
    """Validate requirements.txt"""
    print("📦 Validating requirements.txt...")
    
    try:
        with open('requirements.txt', 'r') as f:
            requirements = f.read()
        
        required_packages = [
            'firebase-admin',
            'fastapi',
            'openai',
            'uvicorn',
            'pydantic'
        ]
        
        missing_packages = []
        for package in required_packages:
            if package not in requirements:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            return False
        
        print("✅ All required packages present in requirements.txt")
        return True
        
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False
    except Exception as e:
        print(f"❌ Error reading requirements.txt: {e}")
        return False

def validate_main_fixed_py() -> bool:
    """Validate main_fixed.py exists and has required endpoints"""
    print("🐍 Validating main_fixed.py...")
    
    if not os.path.exists('main_fixed.py'):
        print("❌ main_fixed.py not found")
        return False
    
    try:
        with open('main_fixed.py', 'r') as f:
            content = f.read()
        
        required_endpoints = [
            '/health',
            '/rate-limit-info',
            '/analyzeResponseForRemedy',
            '/getHerbalistResponse'
        ]
        
        missing_endpoints = []
        for endpoint in required_endpoints:
            if endpoint not in content:
                missing_endpoints.append(endpoint)
        
        if missing_endpoints:
            print(f"❌ Missing endpoints in main_fixed.py: {', '.join(missing_endpoints)}")
            return False
        
        # Check for security features
        security_features = [
            'verify_firebase_token',
            'check_rate_limit',
            'firebase_admin'
        ]
        
        missing_features = []
        for feature in security_features:
            if feature not in content:
                missing_features.append(feature)
        
        if missing_features:
            print(f"❌ Missing security features: {', '.join(missing_features)}")
            return False
        
        print("✅ main_fixed.py contains all required endpoints and security features")
        return True
        
    except Exception as e:
        print(f"❌ Error reading main_fixed.py: {e}")
        return False

def validate_syntax() -> bool:
    """Validate Python syntax"""
    print("🔍 Validating Python syntax...")
    
    try:
        import subprocess
        result = subprocess.run(['python3', '-m', 'py_compile', 'main_fixed.py'], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Syntax error in main_fixed.py: {result.stderr}")
            return False
        
        print("✅ Python syntax valid")
        return True
        
    except Exception as e:
        print(f"❌ Error checking syntax: {e}")
        return False

def create_deployment_registry() -> Dict[str, Any]:
    """Create deployment registry entry"""
    import datetime
    
    return {
        "current_production": "https://herba-ai-proxy-31dzwinpz-dustins-projects-2a4636fb.vercel.app",
        "deployment_date": datetime.datetime.now().isoformat(),
        "version": "main_fixed.py",
        "features": [
            "firebase_auth",
            "rate_limiting", 
            "ai_analysis",
            "input_validation"
        ],
        "ios_app_urls": [
            "Herba2/ViewModels/AIHerbalistChatViewModel.swift",
            "Herba2/Services/AIService.swift",
            "Herba2/Services/AIHerbalistService.swift"
        ],
        "validation_passed": True
    }

def main():
    """Main validation function"""
    print("🚀 Herba2 Backend Configuration Validator")
    print("=" * 50)
    
    validations = [
        ("Vercel Configuration", validate_vercel_config),
        ("Requirements", validate_requirements),
        ("Main File", validate_main_fixed_py),
        ("Python Syntax", validate_syntax)
    ]
    
    passed = 0
    failed = 0
    
    for name, validation_func in validations:
        print(f"\n📋 {name}...")
        if validation_func():
            passed += 1
        else:
            failed += 1
    
    print("\n" + "=" * 50)
    print("📊 Validation Summary:")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    
    if failed == 0:
        print("\n🎉 All validations passed! Ready for deployment.")
        
        # Create deployment registry
        registry = create_deployment_registry()
        with open('deployment_registry.json', 'w') as f:
            json.dump(registry, f, indent=2)
        print("📝 Deployment registry created: deployment_registry.json")
        
        return True
    else:
        print("\n🚨 Some validations failed! Fix issues before deployment.")
        print("\n🔧 Common fixes:")
        print("1. Update vercel.json to use main_fixed.py")
        print("2. Add missing packages to requirements.txt")
        print("3. Fix syntax errors in main_fixed.py")
        print("4. Ensure all endpoints are implemented")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
