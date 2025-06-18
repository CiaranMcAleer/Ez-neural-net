#!/usr/bin/env python3
"""
Build script for Ez Vision package
"""

import os
import subprocess
import sys
import shutil

def run_command(cmd, description):
    """Run a command and check for errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def clean_build():
    """Clean previous build artifacts"""
    print("🧹 Cleaning previous build artifacts...")
    
    dirs_to_clean = ['build', 'dist', 'ez_vision.egg-info']
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"  Removed {dir_name}/")
    
    print("✅ Clean completed")

def build_package():
    """Build the package"""
    print("\n📦 Building Ez Vision package...")
    
    # Check if build module is available
    try:
        import build
    except ImportError:
        print("Installing build module...")
        if not run_command("pip install build", "Installing build"):
            return False
    
    # Build the package
    return run_command("python -m build", "Building package")

def validate_build():
    """Validate the built package"""
    print("\n✅ Validating build...")
    
    # Check if dist directory exists and has files
    if not os.path.exists('dist'):
        print("❌ dist directory not found")
        return False
    
    dist_files = os.listdir('dist')
    wheel_files = [f for f in dist_files if f.endswith('.whl')]
    tar_files = [f for f in dist_files if f.endswith('.tar.gz')]
    
    if not wheel_files:
        print("❌ No wheel (.whl) file found")
        return False
    
    if not tar_files:
        print("❌ No source distribution (.tar.gz) file found")
        return False
    
    print(f"✅ Found wheel: {wheel_files[0]}")
    print(f"✅ Found source dist: {tar_files[0]}")
    
    return True

def main():
    """Main build process"""
    print("🚀 Ez Vision Package Build")
    print("=" * 40)
    
    # Step 1: Clean
    clean_build()
    
    # Step 2: Build
    if not build_package():
        print("\n❌ Build failed!")
        return False
    
    # Step 3: Validate
    if not validate_build():
        print("\n❌ Build validation failed!")
        return False
    
    print("\n🎉 Package built successfully!")
    print("\n📝 Next steps:")
    print("1. Test the package:")
    print("   pip install dist/ez_vision-*.whl")
    print("2. Upload to Test PyPI:")
    print("   twine upload --repository testpypi dist/*")
    print("3. Upload to PyPI:")
    print("   twine upload dist/*")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
