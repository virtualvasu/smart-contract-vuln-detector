#!/usr/bin/env python3
"""Test script to verify model loading works correctly"""

import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_model_loading():
    """Test loading different model types"""
    
    print("="*70)
    print("TESTING MODEL LOADING")
    print("="*70)
    
    # Import after adding to path
    from streamlit_app import get_available_models
    
    available_models = get_available_models()
    
    print(f"\n✅ Found {len(available_models)} model types:")
    for name, info in available_models.items():
        print(f"   - {name}: {len(info['files'])} checkpoint(s)")
    
    if not available_models:
        print("❌ No models found!")
        return False
    
    # Test each model type
    success_count = 0
    fail_count = 0
    
    for model_name, model_info in available_models.items():
        print(f"\n{'-'*70}")
        print(f"Testing: {model_name}")
        print(f"{'-'*70}")
        
        try:
            # Import load_model
            from streamlit_app import load_model
            
            # Clear cache
            load_model.clear()
            
            # Try to load the model
            model, tokenizer, device, task_type, loaded_name = load_model(model_name)
            
            if model is None:
                print(f"❌ Failed to load {model_name}")
                fail_count += 1
                continue
            
            print(f"✅ Successfully loaded: {loaded_name}")
            print(f"   Device: {device}")
            print(f"   Task type: {task_type}")
            
            if isinstance(model, dict):
                print(f"   Model type: Ensemble")
                print(f"   Components: {[k for k in model.keys() if k not in ['type', 'vocab_size', 'tokenizer']]}")
            else:
                print(f"   Model type: {type(model).__name__}")
            
            success_count += 1
            
        except Exception as e:
            print(f"❌ Error loading {model_name}:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            fail_count += 1
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"✅ Successful: {success_count}/{len(available_models)}")
    print(f"❌ Failed: {fail_count}/{len(available_models)}")
    
    if fail_count == 0:
        print("\n🎉 All models loaded successfully!")
        return True
    else:
        print(f"\n⚠️ {fail_count} model(s) failed to load")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
