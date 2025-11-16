# Model Selection Guide for Streamlit App

## 📋 Overview

The Streamlit app now supports **multiple AI models** for vulnerability detection. Users can select their preferred model based on their needs for accuracy, speed, or balance.

---

## 🎯 Available Models

### 1. **Ensemble - Stacking (Best)** 🥇 **RECOMMENDED**
- **Type**: Neural Ensemble combining CodeBERT, LSTM, and CNN
- **F1-Score**: 53.7%
- **Accuracy**: 92.8%
- **Precision**: 45.3%
- **Recall**: 66.0%
- **AUC**: 0.903

**Best For:**
- ✅ Production deployments requiring highest accuracy
- ✅ Critical contract audits
- ✅ When false negatives must be minimized
- ✅ Comprehensive vulnerability detection

**Tradeoffs:**
- ⏱️ Slower inference (processes through 3 models + ensemble layer)
- 💾 Higher memory usage

---

### 2. **Ensemble - Attention** 🥈
- **Type**: Attention-weighted ensemble
- **F1-Score**: 49.9%
- **Accuracy**: 91.1%
- **Precision**: 38.7%
- **Recall**: 70.3%
- **AUC**: 0.904

**Best For:**
- ✅ High recall scenarios (catching most vulnerabilities)
- ✅ Exploratory analysis
- ✅ When you want to review more potential issues

**Tradeoffs:**
- ⚠️ Higher false positive rate
- ⏱️ Similar speed to stacking ensemble

---

### 3. **CodeBERT (Latest)**
- **Type**: Transformer-based model for code
- **F1-Score**: 61.8%
- **Accuracy**: 96.2%
- **Precision**: 86.4%
- **Recall**: 48.1%
- **AUC**: 0.858

**Best For:**
- ✅ Balanced performance
- ✅ When precision is important (fewer false alarms)
- ✅ Good general-purpose choice
- ✅ Understanding code context and semantics

**Tradeoffs:**
- ⚠️ May miss some vulnerabilities (lower recall)
- ⏱️ Moderate inference speed

---

### 4. **LSTM (Latest)**
- **Type**: Recurrent Neural Network
- **F1-Score**: 45.7%
- **Accuracy**: 94.9%
- **Precision**: 71.7%
- **Recall**: 33.5%
- **AUC**: 0.891

**Best For:**
- ✅ Fast inference
- ✅ Sequential pattern detection
- ✅ Resource-constrained environments
- ✅ High precision needs

**Tradeoffs:**
- ⚠️ Lower recall (misses more vulnerabilities)
- ⚡ Faster than transformer models

---

### 5. **CNN (Latest)**
- **Type**: Convolutional Neural Network
- **F1-Score**: 33.7%
- **Accuracy**: 78.7%
- **Precision**: 21.0%
- **Recall**: 85.4%
- **AUC**: 0.905

**Best For:**
- ✅ Detecting local code patterns
- ✅ Very high recall (catches almost everything)
- ✅ Quick scanning
- ✅ Baseline comparison

**Tradeoffs:**
- ⚠️ High false positive rate
- ⚠️ Lower overall accuracy

---

## 📊 Model Comparison

| Model | F1-Score | Accuracy | Precision | Recall | Speed | Memory |
|-------|----------|----------|-----------|--------|-------|--------|
| **Ensemble Stacking** 🥇 | 53.7% | 92.8% | 45.3% | 66.0% | Slow | High |
| Ensemble Attention | 49.9% | 91.1% | 38.7% | 70.3% | Slow | High |
| **CodeBERT** | 61.8% | 96.2% | 86.4% | 48.1% | Medium | Medium |
| LSTM | 45.7% | 94.9% | 71.7% | 33.5% | Fast | Low |
| CNN | 33.7% | 78.7% | 21.0% | 85.4% | Fast | Low |

---

## 💡 Selection Guide

### Choose **Ensemble - Stacking** if:
- You need the best overall performance
- Accuracy is critical
- You're auditing important contracts
- Resource usage is not a concern

### Choose **CodeBERT** if:
- You want balanced performance
- You prefer fewer false alarms (high precision)
- You need moderate inference speed
- You want good semantic understanding

### Choose **LSTM** if:
- Speed is important
- You have limited resources
- You prefer high precision over recall
- Sequential patterns matter

### Choose **CNN** if:
- You want to catch as many vulnerabilities as possible
- You can manually review many results
- You need very fast inference
- Local code patterns are important

---

## 🚀 Usage in Streamlit

1. **Launch the app**:
   ```bash
   ./run_streamlit.sh
   # or
   streamlit run streamlit_app.py
   ```

2. **Select Model**:
   - Look at the sidebar on the left
   - Find the "🤖 Model Selection" section
   - Choose your preferred model from the dropdown
   - The app will automatically load it

3. **View Model Info**:
   - Expand the "ℹ️ Model Information" section
   - See detailed explanations of each model type
   - Understand the metrics

4. **Analyze Contracts**:
   - Paste your Solidity contract code
   - Click "🔍 Analyze Vulnerabilities"
   - Review the results with the selected model

---

## 🔧 Technical Details

### Model Architectures

**CodeBERT**:
- Pre-trained transformer model from Microsoft
- Fine-tuned on vulnerability detection
- Uses attention mechanisms for code understanding

**LSTM**:
- Bidirectional LSTM with 2 layers
- Embedding dimension: 128
- Hidden dimension: 256
- Dropout: 0.3

**CNN**:
- Multiple convolutional filters (3, 4, 5-gram)
- 100 filters per size
- Max pooling for feature extraction
- Dropout: 0.5

**Ensemble**:
- Combines predictions from all three models
- Stacking: Neural network meta-learner
- Attention: Weighted combination based on learned attention

---

## 📈 Performance Metrics Explained

- **F1-Score**: Harmonic mean of precision and recall (0-100%, higher is better)
- **Accuracy**: Percentage of correct predictions (0-100%, higher is better)
- **Precision**: Of flagged vulnerabilities, how many are real (0-100%, higher is better)
- **Recall**: Of real vulnerabilities, how many are detected (0-100%, higher is better)
- **AUC**: Area Under ROC Curve (0-1, higher is better)

---

## 🎓 Best Practices

1. **For Production Audits**: Use Ensemble - Stacking
2. **For Development**: Use CodeBERT for quick checks
3. **For Research**: Try multiple models and compare results
4. **For Speed**: Use LSTM or CNN
5. **For Maximum Coverage**: Use CNN or Ensemble - Attention

---

## 🐛 Troubleshooting

**Model not loading?**
- Check that model files exist in `models/` or `results/checkpoints/`
- Ensure models were trained successfully
- Check console for error messages

**Slow performance?**
- Try LSTM or CNN for faster inference
- Check if running on GPU (CUDA available)
- Reduce contract size or analyze fewer functions

**Unexpected results?**
- Try different models and compare
- Check model metrics for that specific model
- Review the contract code for edge cases

---

## 📚 Related Files

- `streamlit_app.py`: Main application file
- `results/metrics/ensemble_comparison_*.json`: Ensemble performance metrics
- `results/metrics/training_summary_*.json`: Individual model metrics
- `results/benchmark/benchmark_report.md`: Comparison with other tools

---

**Last Updated**: November 16, 2025
**Version**: 2.0 (Multi-Model Support)
