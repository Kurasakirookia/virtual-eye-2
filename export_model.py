"""
MINIMAL FIX: Only fix NumPy compatibility issue
This only touches NumPy and keeps everything else intact.
"""

import subprocess
import sys
import os

def run_command(cmd, desc):
    """Run a command safely"""
    print(f"🔧 {desc}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {desc} - Success")
        return True
    except Exception as e:
        print(f"❌ {desc} - Error: {e}")
        return False

def minimal_numpy_fix():
    """Only fix NumPy - nothing else"""
    print("🎯 MINIMAL FIX: Only fixing NumPy compatibility")
    print("📦 Current issue: NumPy 2.x incompatible with TensorFlow 2.13")
    
    # Only touch NumPy
    commands = [
        ("pip uninstall numpy -y", "Removing NumPy 2.x"),
        ("pip install numpy==1.24.3", "Installing compatible NumPy 1.24.3")
    ]
    
    for cmd, desc in commands:
        run_command(cmd, desc)

def test_imports_only():
    """Test just the essential imports"""
    print("\n🧪 Testing imports...")
    try:
        import numpy as np
        print(f"✅ NumPy {np.__version__}")
        
        import tensorflow as tf
        print(f"✅ TensorFlow {tf.__version__}")
        
        from ultralytics import YOLO
        print("✅ Ultralytics OK")
        
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def create_export_only_script():
    """Create a simple export script"""
    script = '''
"""
Simple TFLite export - just the essentials
"""
import os
from ultralytics import YOLO

def export_tflite():
    print("🚀 YOLOv8 TFLite Export")
    
    try:
        # Load model (downloads if needed)
        model = YOLO("yolov8n.pt")
        print("✅ Model loaded")
        
        # Export to TFLite
        model.export(
            format="tflite",
            imgsz=640,
            half=False,
            int8=False,
            dynamic=False,
            simplify=True
        )
        
        # Check results
        tflite_files = [f for f in os.listdir('.') if f.endswith('.tflite')]
        if tflite_files:
            for file in tflite_files:
                size_mb = os.path.getsize(file) / (1024*1024)
                print(f"🎉 Created: {file} ({size_mb:.1f} MB)")
            print("\\n📱 Ready for Flutter!")
        else:
            print("❌ No .tflite files created")
            
    except Exception as e:
        print(f"❌ Export error: {e}")

if __name__ == "__main__":
    export_tflite()
'''
    
    with open("export_only.py", "w") as f:
        f.write(script)
    print("✅ Created export_only.py")

def main():
    print("🎯 MINIMAL FIX - Only NumPy Compatibility")
    print("=" * 50)
    print("This will ONLY:")
    print("- Remove NumPy 2.x") 
    print("- Install NumPy 1.24.3")
    print("- Keep all other packages intact")
    print("=" * 50)
    
    proceed = input("Continue with minimal fix? (y/N): ").lower().strip()
    if proceed != 'y':
        print("❌ Aborted")
        return
    
    # Step 1: Fix only NumPy
    minimal_numpy_fix()
    
    # Step 2: Test imports
    if test_imports_only():
        print("\n🎉 Imports working!")
        
        # Step 3: Create export script
        create_export_only_script()
        
        # Step 4: Offer to run export
        run_export = input("\\nRun TFLite export now? (y/N): ").lower().strip()
        if run_export == 'y':
            print("\\n🚀 Running export...")
            os.system("python export_only.py")
        else:
            print("\\n✅ Ready! Run: python export_only.py")
            
    else:
        print("\\n❌ Still having import issues.")
        print("\\n🔧 You might need to also fix TensorFlow:")
        print("pip install tensorflow==2.13.1")

if __name__ == "__main__":
    main()