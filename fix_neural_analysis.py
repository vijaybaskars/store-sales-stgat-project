"""
Fix for the KeyError: 'best_model' issue in neural results analysis
This script shows the correct way to access neural baseline results
"""

import json

def analyze_neural_results():
    """Correctly analyze neural baseline results"""
    
    print("🧠 Neural Baseline Results Analysis")
    print("=" * 50)
    
    # Load neural results
    with open('results/neural_baseline_results.json', 'r') as f:
        neural_results = json.load(f)
    
    neural_summary = neural_results['summary_statistics']
    
    print("📊 Individual Model Performance:")
    best_model_name = None
    best_rmsle = float('inf')
    
    for model_name, stats in neural_summary.items():
        mean_rmsle = stats['mean_rmsle']
        std_rmsle = stats['std_rmsle']
        count = stats['count']
        
        print(f"   {model_name:20s}: {mean_rmsle:.4f} ± {std_rmsle:.4f} (n={count})")
        
        # Track best model
        if mean_rmsle < best_rmsle:
            best_rmsle = mean_rmsle
            best_model_name = model_name
    
    print(f"\n🏆 Best Neural Model: {best_model_name}")
    print(f"   Best Neural RMSLE: {best_rmsle:.4f}")
    
    # Compare with traditional baseline
    print(f"\n📊 Performance Comparison:")
    traditional_rmsle = 0.4755  # ARIMA from phase 2
    improvement = ((traditional_rmsle - best_rmsle) / traditional_rmsle) * 100
    
    print(f"   Traditional (ARIMA): {traditional_rmsle:.4f}")
    print(f"   Neural (Best):       {best_rmsle:.4f}")
    
    if improvement > 0:
        print(f"   ✅ Neural improvement: {improvement:.1f}%")
    else:
        print(f"   ❌ Neural worse by: {-improvement:.1f}%")
    
    return {
        'best_model_name': best_model_name,
        'best_rmsle': best_rmsle,
        'neural_summary': neural_summary,
        'traditional_rmsle': traditional_rmsle,
        'improvement_percentage': improvement
    }

def get_corrected_code():
    """Generate the corrected code for the notebook"""
    
    code = '''
# CORRECTED CODE for accessing neural results:

import json

# Load neural results
with open('results/neural_baseline_results.json', 'r') as f:
    neural_results = json.load(f)

neural_summary = neural_results['summary_statistics']

# Find best model by comparing mean_rmsle
best_model_name = None
best_rmsle = float('inf')

for model_name, stats in neural_summary.items():
    if stats['mean_rmsle'] < best_rmsle:
        best_rmsle = stats['mean_rmsle']
        best_model_name = model_name

current_neural_rmsle = best_rmsle

print(f"✅ Phase 3 Neural Results Loaded:")
print(f"   Best Model: {best_model_name}")
print(f"   Current Neural RMSLE: {current_neural_rmsle:.4f}")
'''
    
    return code

if __name__ == "__main__":
    results = analyze_neural_results()
    
    print("\n" + "=" * 50)
    print("📝 CORRECTED CODE FOR NOTEBOOK:")
    print("=" * 50)
    print(get_corrected_code())