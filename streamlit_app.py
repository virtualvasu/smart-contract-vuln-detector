import streamlit as st
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import warnings
import sys

# Suppress warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'notebooks'))

# Set page config
st.set_page_config(
    page_title="Smart Contract Vulnerability Analyzer",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model architectures
class CodeBERTForVulnerabilityDetection(nn.Module):
    def __init__(self, model_name, num_classes, dropout=0.1, freeze_base=False):
        super().__init__()
        self.codebert = AutoModel.from_pretrained(model_name)
        
        if freeze_base:
            for param in self.codebert.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.codebert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.codebert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, hidden_size=256, num_classes=1, 
                 num_layers=2, dropout=0.3, use_attention=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                           bidirectional=True)
        self.use_attention = use_attention
        
        if use_attention:
            # Attention mechanism
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1, bias=False)
            )
        
        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(hidden_size * 2)
        
        # Classifier with dropout
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        if self.use_attention:
            # Attention weights
            attn_weights = self.attention(lstm_out)
            attn_weights = torch.softmax(attn_weights, dim=1)
            context = torch.sum(attn_weights * lstm_out, dim=1)
        else:
            # Use last hidden state
            context = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        # Batch normalization
        context = self.batch_norm(context)
        
        # Classification
        logits = self.classifier(context)
        return logits

class CNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256, num_filters=128, 
                 filter_sizes=[3,4,5,6,7], num_classes=1, dropout=0.3, use_batch_norm=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.use_batch_norm = use_batch_norm
        
        # Convolutional layers
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Batch normalization for each conv layer
        if use_batch_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(num_filters) for _ in filter_sizes
            ])
        
        # Classifier with residual connection
        total_filters = len(filter_sizes) * num_filters
        self.classifier = nn.ModuleDict({
            'layer1': nn.Linear(total_filters, total_filters // 2),
            'layer2': nn.Linear(total_filters // 2, total_filters // 4),
            'output': nn.Linear(total_filters // 4, num_classes),
            'skip': nn.Linear(total_filters, total_filters // 4)
        })
        
        if use_batch_norm:
            self.classifier_batch_norm1 = nn.BatchNorm1d(total_filters // 2)
            self.classifier_batch_norm2 = nn.BatchNorm1d(total_filters // 4)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        embedded = embedded.permute(0, 2, 1)
        
        # Apply convolutions with batch norm
        conved = []
        for i, conv in enumerate(self.convs):
            x = torch.relu(conv(embedded))
            if self.use_batch_norm:
                x = self.batch_norms[i](x)
            pooled = torch.max_pool1d(x, x.shape[2]).squeeze(2)
            conved.append(pooled)
        
        # Concatenate all features
        cat = torch.cat(conved, dim=1)
        cat = self.dropout(cat)
        
        # Residual classifier
        x = torch.relu(self.classifier['layer1'](cat))
        if self.use_batch_norm:
            x = self.classifier_batch_norm1(x)
        x = self.dropout(x)
        
        x = torch.relu(self.classifier['layer2'](x))
        skip = self.classifier['skip'](cat)
        x = x + skip  # Residual connection
        if self.use_batch_norm:
            x = self.classifier_batch_norm2(x)
        x = self.dropout(x)
        
        logits = self.classifier['output'](x)
        return logits

class EnsembleModel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.fc = nn.Linear(3, num_classes)
        
    def forward(self, codebert_logits, lstm_logits, cnn_logits):
        combined = torch.cat([codebert_logits, lstm_logits, cnn_logits], dim=1)
        logits = self.fc(combined)
        return logits

class VulnerabilitySolutions:
    """Provides solutions and recommendations for detected vulnerabilities"""
    
    def __init__(self):
        self.solutions = {
            'reentrancy': {
                'title': '🔄 Reentrancy Attack',
                'description': 'External calls before state updates can allow malicious contracts to re-enter your function.',
                'solution': '''
**Solutions:**
1. **Checks-Effects-Interactions Pattern**: Update state before external calls
2. **Reentrancy Guard**: Use OpenZeppelin's ReentrancyGuard modifier
3. **Pull Payment Pattern**: Let users withdraw funds themselves

**Fixed Code Example:**
```solidity
bool private locked;
modifier nonReentrant() {
    require(!locked, "ReentrancyGuard: reentrant call");
    locked = true;
    _;
    locked = false;
}

function withdraw(uint256 amount) public nonReentrant {
    require(balances[msg.sender] >= amount, "Insufficient balance");
    
    // Effects: Update state BEFORE external call
    balances[msg.sender] -= amount;
    
    // Interactions: External call last
    (bool success, ) = msg.sender.call{value: amount}("");
    require(success, "Transfer failed");
}
```''',
                'severity': 'HIGH'
            },
            'unchecked_send': {
                'title': '❌ Unchecked External Call',
                'description': 'External calls (send, call) can fail silently if return values are not checked.',
                'solution': '''
**Solutions:**
1. **Check Return Values**: Always verify success of external calls
2. **Use transfer()**: For simple Ether transfers (throws on failure)
3. **Implement Fallback**: Handle failed transfers gracefully

**Fixed Code Example:**
```solidity
// Instead of: payable(owner).send(amount);
// Use:
require(payable(owner).send(amount), "Transfer failed");

// Or better:
(bool success, ) = payable(owner).call{value: amount}("");
require(success, "Transfer failed");

// Or safest:
payable(owner).transfer(amount); // Throws on failure
```''',
                'severity': 'HIGH'
            },
            'tx_origin': {
                'title': '🎭 tx.origin Vulnerability',
                'description': 'Using tx.origin for authorization can be exploited through phishing attacks.',
                'solution': '''
**Solutions:**
1. **Use msg.sender**: Always use msg.sender instead of tx.origin
2. **Multi-signature**: Implement multi-signature authorization
3. **Access Control**: Use OpenZeppelin's AccessControl

**Fixed Code Example:**
```solidity
// Instead of: require(tx.origin == owner, "Not authorized");
// Use:
require(msg.sender == owner, "Not authorized");

// Or better with OpenZeppelin:
import "@openzeppelin/contracts/access/Ownable.sol";
contract MyContract is Ownable {
    function restrictedFunction() public onlyOwner {
        // Only owner can call this
    }
}
```''',
                'severity': 'MEDIUM'
            },
            'overflow': {
                'title': '🔢 Integer Overflow/Underflow',
                'description': 'Arithmetic operations can overflow or underflow, leading to unexpected values.',
                'solution': '''
**Solutions:**
1. **Use Solidity ^0.8.0**: Built-in overflow/underflow protection
2. **SafeMath Library**: For older Solidity versions
3. **Explicit Checks**: Manual bounds checking

**Fixed Code Example:**
```solidity
// Solidity ^0.8.0 automatically reverts on overflow
function addToBalance(uint256 amount) public {
    // This will automatically revert on overflow
    balances[msg.sender] += amount;
}

// For older versions, use SafeMath:
using SafeMath for uint256;
function addToBalance(uint256 amount) public {
    balances[msg.sender] = balances[msg.sender].add(amount);
}
```''',
                'severity': 'HIGH'
            },
            'timestamp_dependency': {
                'title': '⏰ Timestamp Dependency',
                'description': 'Relying on block.timestamp can be manipulated by miners within a small range.',
                'solution': '''
**Solutions:**
1. **Use Block Numbers**: More reliable than timestamps
2. **Oracle Services**: External time sources for critical timing
3. **Tolerance Ranges**: Accept small time variations

**Fixed Code Example:**
```solidity
// Instead of exact timestamp checks:
// require(block.timestamp > deadline, "Too early");

// Use block numbers or ranges:
require(block.number > deadlineBlock, "Too early");

// Or allow tolerance:
require(block.timestamp > deadline - 300, "Too early"); // 5 min tolerance
```''',
                'severity': 'MEDIUM'
            },
            'unhandled_exception': {
                'title': '💥 Unhandled Exception',
                'description': 'Functions may fail silently without proper error handling.',
                'solution': '''
**Solutions:**
1. **Try-Catch Blocks**: Handle external call failures
2. **Return Booleans**: Check success of operations
3. **Event Logging**: Log failures for monitoring

**Fixed Code Example:**
```solidity
function safeExternalCall(address target, bytes memory data) public returns (bool success) {
    try target.call(data) returns (bytes memory) {
        return true;
    } catch {
        emit ExternalCallFailed(target);
        return false;
    }
}
```''',
                'severity': 'MEDIUM'
            }
        }
    
    def get_vulnerability_solution(self, function_code, prediction_type):
        """Analyze function code and provide specific solution"""
        code_lower = function_code.lower()
        
        # Detect specific vulnerability patterns
        if 'call{value:' in code_lower and 'balances[' in code_lower:
            if code_lower.find('balances[') > code_lower.find('call{value:'):
                return self.solutions['reentrancy']
        
        if '.send(' in code_lower and 'require(' not in code_lower:
            return self.solutions['unchecked_send']
        
        if 'tx.origin' in code_lower:
            return self.solutions['tx_origin']
        
        if '+=' in code_lower and 'balances[' in code_lower:
            return self.solutions['overflow']
        
        if 'block.timestamp' in code_lower or 'now' in code_lower:
            return self.solutions['timestamp_dependency']
        
        if 'call(' in code_lower and 'try' not in code_lower:
            return self.solutions['unhandled_exception']
        
        # Default solution for vulnerable functions
        if prediction_type == 'Vulnerable':
            return {
                'title': '⚠️ General Vulnerability Detected',
                'description': 'This function has been flagged as potentially vulnerable by the AI model.',
                'solution': '''
**General Security Recommendations:**
1. **Follow Security Best Practices**: Use established patterns and libraries
2. **External Audits**: Have your contract audited by security experts
3. **Testing**: Implement comprehensive unit and integration tests
4. **Formal Verification**: Use tools like Mythril, Slither, or Manticore
5. **Bug Bounty**: Consider running a bug bounty program

**Common Issues to Check:**
- Reentrancy attacks
- Integer overflow/underflow
- Unchecked external calls
- Access control issues
- Front-running vulnerabilities
''',
                'severity': 'MEDIUM'
            }
        
        return None

class SolidityFunctionParser:
    """Extract functions from Solidity contract code"""
    
    def __init__(self):
        self.function_pattern = re.compile(
            r'function\s+(\w+)\s*\([^)]*\)\s*(?:public|private|internal|external)?\s*(?:view|pure|payable)?\s*(?:returns\s*\([^)]*\))?\s*\{',
            re.MULTILINE | re.IGNORECASE
        )
        
    def extract_functions(self, contract_code):
        """Extract all functions from contract code"""
        functions = []
        matches = list(self.function_pattern.finditer(contract_code))
        
        for i, match in enumerate(matches):
            func_name = match.group(1)
            start_pos = match.start()
            
            # Find the end of the function by counting braces
            brace_count = 0
            func_start = match.end() - 1
            
            for j, char in enumerate(contract_code[func_start:]):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        func_end = func_start + j + 1
                        break
            else:
                continue
            
            func_code = contract_code[start_pos:func_end].strip()
            func_code = self._clean_function_code(func_code)
            
            functions.append({
                'name': func_name,
                'code': func_code,
                'start_line': contract_code[:start_pos].count('\n') + 1,
                'length': len(func_code)
            })
        
        return functions
    
    def _clean_function_code(self, code):
        """Clean and normalize function code"""
        code = re.sub(r'\n\s*\n', '\n', code)
        code = re.sub(r'\s+', ' ', code)
        code = re.sub(r'//.*?\n', '\n', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        return code.strip()

def load_model_with_checkpoint_compatibility(model, checkpoint):
    """Load model state dict with compatibility"""
    state_dict = checkpoint['model_state_dict']
    
    if any(key.startswith('bert.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('bert.'):
                new_key = key.replace('bert.', 'codebert.')
                new_state_dict[new_key] = value
            else:
                new_state_dict[key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict, strict=False)
    return model

def get_available_models():
    """Get all available trained models"""
    models_dir = Path('models')
    checkpoints_dir = Path('results/checkpoints')
    
    available_models = {
        'CodeBERT (Latest)': {'type': 'codebert', 'files': []},
        'LSTM (Latest)': {'type': 'lstm', 'files': []},
        'CNN (Latest)': {'type': 'cnn', 'files': []},
        'Ensemble - Stacking (Best)': {'type': 'ensemble_stacking', 'files': []},
        'Ensemble - Attention': {'type': 'ensemble_attention', 'files': []}
    }
    
    # Find CodeBERT models
    if models_dir.exists():
        available_models['CodeBERT (Latest)']['files'].extend(list(models_dir.glob('codebert*.pt')))
    if checkpoints_dir.exists():
        available_models['CodeBERT (Latest)']['files'].extend(list(checkpoints_dir.glob('best_model_codebert*.pt')))
    
    # Find LSTM models
    if models_dir.exists():
        available_models['LSTM (Latest)']['files'].extend(list(models_dir.glob('lstm*.pt')))
    if checkpoints_dir.exists():
        available_models['LSTM (Latest)']['files'].extend(list(checkpoints_dir.glob('best_model_lstm*.pt')))
    
    # Find CNN models
    if models_dir.exists():
        available_models['CNN (Latest)']['files'].extend(list(models_dir.glob('cnn*.pt')))
    if checkpoints_dir.exists():
        available_models['CNN (Latest)']['files'].extend(list(checkpoints_dir.glob('best_model_cnn*.pt')))
    
    # Find Ensemble models
    if checkpoints_dir.exists():
        available_models['Ensemble - Stacking (Best)']['files'].extend(
            list(checkpoints_dir.glob('best_ensemble_stacking*.pt'))
        )
        available_models['Ensemble - Attention']['files'].extend(
            list(checkpoints_dir.glob('best_ensemble_attention*.pt'))
        )
    
    # Filter out models with no files
    available_models = {name: info for name, info in available_models.items() if info['files']}
    
    return available_models

@st.cache_resource
def load_model(model_choice='auto'):
    """Load the selected model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    available_models = get_available_models()
    
    if not available_models:
        st.error("❌ No model checkpoints found! Please train a model first using the notebooks")
        return None, None, None, None, None
    
    # Select model based on choice
    if model_choice == 'auto':
        # Prefer ensemble stacking if available
        if 'Ensemble - Stacking (Best)' in available_models:
            model_choice = 'Ensemble - Stacking (Best)'
        elif 'CodeBERT (Latest)' in available_models:
            model_choice = 'CodeBERT (Latest)'
        else:
            model_choice = list(available_models.keys())[0]
    
    if model_choice not in available_models:
        st.error(f"❌ Selected model '{model_choice}' not found!")
        return None, None, None, None, None
    
    model_info = available_models[model_choice]
    model_type = model_info['type']
    model_file = max(model_info['files'], key=lambda x: x.stat().st_mtime)
    
    try:
        checkpoint = torch.load(model_file, map_location=device)
        
        # Load based on model type
        if model_type == 'codebert':
            model_name = checkpoint.get('model_name', 'microsoft/codebert-base')
            num_classes = checkpoint.get('num_classes', 1)
            task_type = checkpoint.get('task', 'binary')
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = CodeBERTForVulnerabilityDetection(model_name, num_classes)
            model = load_model_with_checkpoint_compatibility(model, checkpoint)
            
        elif model_type == 'lstm':
            vocab_size = checkpoint.get('vocab_size', 10000)
            num_classes = checkpoint.get('num_classes', 1)
            task_type = checkpoint.get('task', 'binary')
            
            # Get model config from checkpoint
            model_config = checkpoint.get('model_config', {})
            embedding_dim = model_config.get('embedding_dim', 256)
            hidden_size = model_config.get('hidden_size', 256)
            num_layers = model_config.get('num_layers', 2)
            dropout = model_config.get('dropout', 0.3)
            use_attention = model_config.get('use_attention', True)
            
            tokenizer = checkpoint.get('tokenizer', None)
            model = LSTMClassifier(
                vocab_size=vocab_size, 
                embedding_dim=embedding_dim,
                hidden_size=hidden_size,
                num_classes=num_classes,
                num_layers=num_layers,
                dropout=dropout,
                use_attention=use_attention
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            
        elif model_type == 'cnn':
            vocab_size = checkpoint.get('vocab_size', 10000)
            num_classes = checkpoint.get('num_classes', 1)
            task_type = checkpoint.get('task', 'binary')
            
            # Get model config from checkpoint
            model_config = checkpoint.get('model_config', {})
            embedding_dim = model_config.get('embedding_dim', 256)
            num_filters = model_config.get('num_filters', 128)
            filter_sizes = model_config.get('filter_sizes', [3, 4, 5, 6, 7])
            dropout = model_config.get('dropout', 0.3)
            use_batch_norm = model_config.get('use_batch_norm', True)
            
            tokenizer = checkpoint.get('tokenizer', None)
            model = CNNClassifier(
                vocab_size=vocab_size,
                embedding_dim=embedding_dim,
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                num_classes=num_classes,
                dropout=dropout,
                use_batch_norm=use_batch_norm
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            
        elif model_type in ['ensemble_stacking', 'ensemble_attention']:
            # Load ensemble components
            state_dict = checkpoint['model_state_dict']
            
            # Infer num_classes from meta-learner's last layer
            meta_keys = [k for k in state_dict.keys() if 'meta_learner' in k and 'weight' in k]
            if meta_keys:
                last_layer_key = [k for k in meta_keys if 'weight' in k][-1]
                num_classes = state_dict[last_layer_key].shape[0]
            else:
                num_classes = checkpoint.get('num_classes', 1)
            
            task_type = checkpoint.get('task', 'binary')
            
            # Load CodeBERT
            model_name = checkpoint.get('model_name', 'microsoft/codebert-base')
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            codebert_model = CodeBERTForVulnerabilityDetection(model_name, num_classes)
            
            # Get vocab size from checkpoint (should be in the state dict)
            state_dict = checkpoint['model_state_dict']
            vocab_size = None
            for key in state_dict.keys():
                if 'embedding.weight' in key and ('lstm_model' in key or 'cnn_model' in key):
                    vocab_size = state_dict[key].shape[0]
                    break
            
            if vocab_size is None:
                vocab_size = 50265  # Default CodeBERT vocab size
            
            # Load LSTM with correct architecture from checkpoint
            lstm_embedding_dim = 256
            lstm_hidden_size = 256
            for key in state_dict.keys():
                if 'lstm_model.embedding.weight' in key:
                    lstm_embedding_dim = state_dict[key].shape[1]
                    break
            
            lstm_model = LSTMClassifier(
                vocab_size=vocab_size,
                embedding_dim=lstm_embedding_dim,
                hidden_size=lstm_hidden_size,
                num_classes=num_classes,
                num_layers=2,
                dropout=0.3,
                use_attention=True
            )
            
            # Load CNN with correct architecture from checkpoint
            cnn_embedding_dim = 256
            cnn_num_filters = 128
            cnn_filter_sizes = [3, 4, 5, 6, 7]
            for key in state_dict.keys():
                if 'cnn_model.embedding.weight' in key:
                    cnn_embedding_dim = state_dict[key].shape[1]
                    break
            
            cnn_model = CNNClassifier(
                vocab_size=vocab_size,
                embedding_dim=cnn_embedding_dim,
                num_filters=cnn_num_filters,
                filter_sizes=cnn_filter_sizes,
                num_classes=num_classes,
                dropout=0.3,
                use_batch_norm=True
            )
            
            # Create meta-learner/attention layer based on what's in checkpoint
            if 'meta_learner.0.weight' in state_dict:
                # Stacking ensemble
                layer0_out = state_dict['meta_learner.0.weight'].shape[0]
                layer1_out = state_dict['meta_learner.3.weight'].shape[0]
                
                meta_learner = nn.Sequential(
                    nn.Linear(3, layer0_out),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(layer0_out, layer1_out),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(layer1_out, num_classes)
                )
                meta_key_prefix = 'meta_learner'
            elif 'attention_layer.0.weight' in state_dict:
                # Attention ensemble
                layer0_out = state_dict['attention_layer.0.weight'].shape[0]
                
                meta_learner = nn.Sequential(
                    nn.Linear(3, layer0_out),
                    nn.Tanh(),
                    nn.Linear(layer0_out, 3),  # Outputs attention weights for 3 models
                    nn.Softmax(dim=-1)
                )
                meta_key_prefix = 'attention_layer'
            else:
                # Default stacking architecture
                meta_learner = nn.Sequential(
                    nn.Linear(3, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(64, num_classes)
                )
                meta_key_prefix = 'meta_learner'
            
            # Load state dicts by filtering keys
            codebert_state = {k.replace('codebert_model.', ''): v 
                            for k, v in state_dict.items() if k.startswith('codebert_model.')}
            lstm_state = {k.replace('lstm_model.', ''): v 
                         for k, v in state_dict.items() if k.startswith('lstm_model.')}
            cnn_state = {k.replace('cnn_model.', ''): v 
                        for k, v in state_dict.items() if k.startswith('cnn_model.')}
            meta_state = {k.replace(f'{meta_key_prefix}.', ''): v 
                         for k, v in state_dict.items() if k.startswith(f'{meta_key_prefix}.')}
            
            # Load with compatibility for bert vs codebert naming
            if any(key.startswith('bert.') for key in codebert_state.keys()):
                codebert_state_new = {}
                for key, value in codebert_state.items():
                    if key.startswith('bert.'):
                        new_key = key.replace('bert.', 'codebert.')
                        codebert_state_new[new_key] = value
                    else:
                        codebert_state_new[key] = value
                codebert_state = codebert_state_new
            
            codebert_model.load_state_dict(codebert_state, strict=False)
            lstm_model.load_state_dict(lstm_state)
            cnn_model.load_state_dict(cnn_state)
            meta_learner.load_state_dict(meta_state)
            
            model = {
                'type': 'ensemble',
                'ensemble_type': model_type,  # stacking or attention
                'codebert': codebert_model,
                'lstm': lstm_model,
                'cnn': cnn_model,
                'meta_learner': meta_learner,
                'vocab_size': vocab_size,
                'tokenizer': tokenizer
            }
        
        else:
            st.error(f"❌ Unknown model type: {model_type}")
            return None, None, None, None, None
        
        # Move models to device
        if isinstance(model, dict):
            for key in ['codebert', 'lstm', 'cnn', 'meta_learner']:
                if key in model:
                    model[key].to(device)
                    model[key].eval()
        else:
            model.to(device)
            model.eval()
        
        return model, tokenizer, device, task_type, model_choice
        
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        import traceback
        st.error(traceback.format_exc())
        return None, None, None, None, None

def simple_tokenize(text, vocab_size=50265, max_length=512):
    """Simple tokenization for LSTM/CNN models"""
    # Use CodeBERT tokenizer for consistency with training
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
    
    # Tokenize and get input IDs
    encoding = tokenizer(
        text,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors='pt'
    )
    
    return encoding['input_ids']

def predict_function_vulnerability(function_code, model, tokenizer, device, task_type, max_length=512):
    """Predict vulnerability for a single function"""
    
    # Get model type name for checking (handles __main__ module issue in Streamlit)
    model_type_name = type(model).__name__ if not isinstance(model, dict) else None
    
    # Handle ensemble models
    if isinstance(model, dict) and model.get('type') == 'ensemble':
        # CodeBERT encoding
        encoding = tokenizer(
            function_code,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        input_ids_bert = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # LSTM/CNN encoding (use same tokenization as CodeBERT)
        input_ids_lstm_cnn = simple_tokenize(function_code, model.get('vocab_size', 50265), max_length).to(device)
        
        with torch.no_grad():
            # Get predictions from each model
            codebert_logits = model['codebert'](input_ids_bert, attention_mask)
            lstm_logits = model['lstm'](input_ids_lstm_cnn)
            cnn_logits = model['cnn'](input_ids_lstm_cnn)
            
            # Handle different ensemble types
            ensemble_type = model.get('ensemble_type', 'ensemble_stacking')
            
            if 'attention' in ensemble_type:
                # Attention-based ensemble
                if task_type == 'binary':
                    # For binary, get probabilities first
                    codebert_prob = torch.sigmoid(codebert_logits)
                    lstm_prob = torch.sigmoid(lstm_logits)
                    cnn_prob = torch.sigmoid(cnn_logits)
                    
                    # Stack predictions
                    stacked = torch.cat([codebert_prob, lstm_prob, cnn_prob], dim=1)
                    
                    # Get attention weights
                    attention_weights = model['meta_learner'](stacked)
                    
                    # Weighted average
                    weighted_prob = (attention_weights[:, 0:1] * codebert_prob + 
                                   attention_weights[:, 1:2] * lstm_prob + 
                                   attention_weights[:, 2:3] * cnn_prob)
                    
                    probabilities = weighted_prob.cpu().numpy()[0]
                    vulnerability_prob = probabilities[0]
                    safe_prob = 1 - vulnerability_prob
                    
                    prediction = 'Vulnerable' if vulnerability_prob > 0.5 else 'Safe'
                    confidence = max(vulnerability_prob, safe_prob)
                else:
                    # Multi-class attention ensemble
                    codebert_pred = torch.softmax(codebert_logits, dim=-1)
                    lstm_pred = torch.softmax(lstm_logits, dim=-1)
                    cnn_pred = torch.softmax(cnn_logits, dim=-1)
                    
                    # Simple average for multi-class
                    avg_pred = (codebert_pred + lstm_pred + cnn_pred) / 3
                    probabilities = avg_pred.cpu().numpy()[0]
                    predicted_class = np.argmax(probabilities)
                    confidence = probabilities[predicted_class]
                    
                    class_names = ['Overflow-Underflow', 'Re-entrancy', 'SAFE', 'TOD', 
                                  'Timestamp-Dependency', 'Unchecked-Send', 'Unhandled-Exceptions', 'tx.origin']
                    prediction = class_names[predicted_class] if predicted_class < len(class_names) else 'Unknown'
                    vulnerability_prob = 1 - probabilities[2] if predicted_class != 2 else 0
                    safe_prob = probabilities[2] if len(probabilities) > 2 else 0
                    
            else:
                # Stacking ensemble
                if task_type == 'binary':
                    # For binary, apply sigmoid first
                    codebert_pred = torch.sigmoid(codebert_logits)
                    lstm_pred = torch.sigmoid(lstm_logits)
                    cnn_pred = torch.sigmoid(cnn_logits)
                    
                    # Concatenate predictions
                    stacked = torch.cat([codebert_pred, lstm_pred, cnn_pred], dim=1)
                    
                    # Meta-learner prediction
                    logits = model['meta_learner'](stacked)
                    probabilities = torch.sigmoid(logits).cpu().numpy()[0]
                    vulnerability_prob = probabilities[0]
                    safe_prob = 1 - vulnerability_prob
                    
                    prediction = 'Vulnerable' if vulnerability_prob > 0.5 else 'Safe'
                    confidence = max(vulnerability_prob, safe_prob)
                else:
                    # For multi-class
                    codebert_pred = torch.softmax(codebert_logits, dim=-1)
                    lstm_pred = torch.softmax(lstm_logits, dim=-1)
                    cnn_pred = torch.softmax(cnn_logits, dim=-1)
                    
                    # Average predictions (simple ensemble for multi-class)
                    avg_pred = (codebert_pred + lstm_pred + cnn_pred) / 3
                    probabilities = avg_pred.cpu().numpy()[0]
                    predicted_class = np.argmax(probabilities)
                    confidence = probabilities[predicted_class]
                    
                    class_names = ['Overflow-Underflow', 'Re-entrancy', 'SAFE', 'TOD', 
                                  'Timestamp-Dependency', 'Unchecked-Send', 'Unhandled-Exceptions', 'tx.origin']
                    prediction = class_names[predicted_class] if predicted_class < len(class_names) else 'Unknown'
                    vulnerability_prob = 1 - probabilities[2] if predicted_class != 2 else 0
                    safe_prob = probabilities[2] if len(probabilities) > 2 else 0
    
    # Handle CodeBERT models
    elif model_type_name == 'CodeBERTForVulnerabilityDetection':
        encoding = tokenizer(
            function_code,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            
            if task_type == 'binary':
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
                vulnerability_prob = probabilities[0]
                safe_prob = 1 - vulnerability_prob
                
                prediction = 'Vulnerable' if vulnerability_prob > 0.5 else 'Safe'
                confidence = max(vulnerability_prob, safe_prob)
                
            else:
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]
                
                class_names = ['Overflow-Underflow', 'Re-entrancy', 'SAFE', 'TOD', 
                              'Timestamp-Dependency', 'Unchecked-Send', 'Unhandled-Exceptions', 'tx.origin']
                prediction = class_names[predicted_class] if predicted_class < len(class_names) else 'Unknown'
                vulnerability_prob = 1 - probabilities[2] if predicted_class != 2 else 0
                safe_prob = probabilities[2] if len(probabilities) > 2 else 0
    
    # Handle LSTM/CNN models
    elif model_type_name in ['LSTMClassifier', 'CNNClassifier']:
        # For LSTM/CNN, use CodeBERT tokenizer
        from transformers import AutoTokenizer
        codebert_tokenizer = AutoTokenizer.from_pretrained('microsoft/codebert-base')
        
        encoding = codebert_tokenizer(
            function_code,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].to(device)
        
        with torch.no_grad():
            logits = model(input_ids)
            
            if task_type == 'binary':
                probabilities = torch.sigmoid(logits).cpu().numpy()[0]
                vulnerability_prob = probabilities[0]
                safe_prob = 1 - vulnerability_prob
                
                prediction = 'Vulnerable' if vulnerability_prob > 0.5 else 'Safe'
                confidence = max(vulnerability_prob, safe_prob)
            else:
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()[0]
                predicted_class = np.argmax(probabilities)
                confidence = probabilities[predicted_class]
                
                class_names = ['Overflow-Underflow', 'Re-entrancy', 'SAFE', 'TOD', 
                              'Timestamp-Dependency', 'Unchecked-Send', 'Unhandled-Exceptions', 'tx.origin']
                prediction = class_names[predicted_class] if predicted_class < len(class_names) else 'Unknown'
                vulnerability_prob = 1 - probabilities[2] if predicted_class != 2 else 0
                safe_prob = probabilities[2] if len(probabilities) > 2 else 0
    
    # Unknown model type
    else:
        st.error(f"❌ Unknown model type: {type(model)}")
        return {
            'prediction': 'Error',
            'confidence': 0.0,
            'vulnerability_probability': 0.0,
            'safe_probability': 0.0
        }
    
    return {
        'prediction': prediction,
        'confidence': float(confidence),
        'vulnerability_probability': float(vulnerability_prob),
        'safe_probability': float(safe_prob)
    }

def main():
    st.title("🔐 Smart Contract Vulnerability Analyzer")
    st.markdown("**AI-powered security analysis using Deep Learning**")
    
    # Sidebar - Model Selection
    st.sidebar.title("🤖 Model Selection")
    
    # Get available models
    available_models = get_available_models()
    
    if not available_models:
        st.error("❌ No models found! Please train models using the notebooks first.")
        st.stop()
    
    model_names = list(available_models.keys())
    
    # Add model performance info
    model_performance = {
        'Ensemble - Stacking (Best)': '🥇 F1: 53.7% | Acc: 92.8% (Recommended)',
        'Ensemble - Attention': '🥈 F1: 49.9% | Acc: 91.1%',
        'CodeBERT (Latest)': 'F1: 61.8% | Acc: 96.2%',
        'LSTM (Latest)': 'F1: 45.7% | Acc: 94.9%',
        'CNN (Latest)': 'F1: 33.7% | Acc: 78.7%'
    }
    
    # Create selection options with performance info
    default_index = 0
    if 'Ensemble - Stacking (Best)' in model_names:
        default_index = model_names.index('Ensemble - Stacking (Best)')
    
    selected_model = st.sidebar.selectbox(
        "Choose AI Model:",
        model_names,
        index=default_index,
        format_func=lambda x: f"{x} - {model_performance.get(x, 'No metrics available')}"
    )
    
    st.sidebar.info(f"**Selected:** {selected_model}")
    
    # Model info
    with st.sidebar.expander("ℹ️ Model Information"):
        st.markdown("""
        **Model Types:**
        - 🥇 **Ensemble (Stacking)**: Combines CodeBERT, LSTM & CNN for best accuracy
        - **CodeBERT**: Transformer-based model specialized for code
        - **LSTM**: Recurrent neural network for sequential patterns
        - **CNN**: Convolutional network for local feature detection
        
        **Metrics:**
        - **F1-Score**: Balance between precision and recall
        - **Accuracy**: Overall correctness
        - **Precision**: How many detected vulnerabilities are real
        - **Recall**: How many real vulnerabilities are detected
        """)
    
    # Load model
    with st.spinner(f'Loading {selected_model}...'):
        model, tokenizer, device, task_type, loaded_model_name = load_model(selected_model)
    
    if model is None:
        st.stop()
    
    # Display model info
    col1, col2 = st.columns([2, 1])
    with col1:
        st.success(f"✅ Model loaded successfully: **{loaded_model_name}**")
    with col2:
        st.info(f"🖥️ Device: **{device}**")
    
    # Sidebar - Sample Contracts
    st.sidebar.divider()
    st.sidebar.title("📋 Sample Contracts")
    
    sample_vulnerable = """pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint256) public balances;
    address public owner;
    
    constructor() {
        owner = msg.sender;
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) public {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Vulnerable to reentrancy - external call before state update
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
        
        balances[msg.sender] -= amount;
    }
    
    function emergencyWithdraw() public {
        require(msg.sender == owner, "Only owner");
        
        // Unchecked send - vulnerable
        payable(owner).send(address(this).balance);
    }
    
    function addToBalance(uint256 amount) public {
        // Potential overflow vulnerability
        balances[msg.sender] += amount;
    }
    
    function transferOwnership(address newOwner) public {
        // tx.origin vulnerability
        require(tx.origin == owner, "Not authorized");
        owner = newOwner;
    }
    
    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
}"""

    sample_safe = """pragma solidity ^0.8.0;

contract SafeBank {
    mapping(address => uint256) public balances;
    address public owner;
    bool private locked;
    
    modifier nonReentrant() {
        require(!locked, "ReentrancyGuard: reentrant call");
        locked = true;
        _;
        locked = false;
    }
    
    constructor() {
        owner = msg.sender;
    }
    
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }
    
    function withdraw(uint256 amount) public nonReentrant {
        require(balances[msg.sender] >= amount, "Insufficient balance");
        
        // Safe: state update before external call
        balances[msg.sender] -= amount;
        
        (bool success, ) = msg.sender.call{value: amount}("");
        require(success, "Transfer failed");
    }
    
    function emergencyWithdraw() public {
        require(msg.sender == owner, "Only owner");
        
        // Safe: check return value
        require(payable(owner).send(address(this).balance), "Transfer failed");
    }
    
    function getBalance() public view returns (uint256) {
        return balances[msg.sender];
    }
}"""

    if st.sidebar.button("📊 Load Vulnerable Sample"):
        st.session_state.contract_code = sample_vulnerable
        st.rerun()
    
    if st.sidebar.button("✅ Load Safe Sample"):
        st.session_state.contract_code = sample_safe
        st.rerun()
    
    # Main input
    st.subheader("📝 Contract Code")
    
    contract_code = st.text_area(
        "Paste your Solidity contract code:",
        value=st.session_state.get('contract_code', ''),
        height=300,
        placeholder="pragma solidity ^0.8.0;\n\ncontract MyContract {\n    // Your contract code here...\n}"
    )
    
    contract_name = st.text_input("Contract Name (optional):", value="MyContract")
    
    if st.button("🔍 Analyze Vulnerabilities", type="primary"):
        if not contract_code.strip():
            st.error("Please enter contract code to analyze")
            return
        
        with st.spinner('Analyzing contract for vulnerabilities...'):
            # Parse functions
            parser = SolidityFunctionParser()
            functions = parser.extract_functions(contract_code)
            
            if not functions:
                st.error("No functions found in the contract. Please check your Solidity code.")
                return
            
            # Analyze each function
            results = []
            progress_bar = st.progress(0)
            solutions_provider = VulnerabilitySolutions()
            
            for i, func in enumerate(functions):
                prediction = predict_function_vulnerability(
                    func['code'], model, tokenizer, device, task_type
                )
                
                # Get vulnerability solution if vulnerable
                solution = solutions_provider.get_vulnerability_solution(
                    func['code'], prediction['prediction']
                )
                
                result = {
                    'contract_name': contract_name,
                    'function_name': func['name'],
                    'function_code': func['code'],
                    'code_length': func['length'],
                    'start_line': func['start_line'],
                    'solution': solution,
                    **prediction
                }
                results.append(result)
                progress_bar.progress((i + 1) / len(functions))
        
        # Display results
        st.success(f"✅ Analysis complete! Found {len(functions)} functions")
        
        # Summary metrics
        vulnerable_count = len([r for r in results if r['prediction'] == 'Vulnerable'])
        avg_vuln_prob = np.mean([r['vulnerability_probability'] for r in results])
        
        # Risk level
        if vulnerable_count > len(results) * 0.5:
            risk_level = "🔴 HIGH RISK"
            risk_color = "red"
        elif vulnerable_count > 0 or avg_vuln_prob > 0.3:
            risk_level = "🟡 MEDIUM RISK"
            risk_color = "orange"
        else:
            risk_level = "🟢 LOW RISK"
            risk_color = "green"
        
        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Functions", len(results))
        
        with col2:
            st.metric("Vulnerable Functions", vulnerable_count)
        
        with col3:
            st.metric("Avg Vulnerability", f"{avg_vuln_prob:.1%}")
        
        with col4:
            st.markdown(f"**Risk Level**")
            st.markdown(f"<h3 style='color: {risk_color}'>{risk_level}</h3>", unsafe_allow_html=True)
        
        # Vulnerability Summary
        vulnerable_results = [r for r in results if r['prediction'] == 'Vulnerable']
        if vulnerable_results:
            st.divider()
            st.subheader("⚠️ Security Issues Summary")
            
            # Group vulnerabilities by type
            vuln_types = {}
            for result in vulnerable_results:
                if result.get('solution'):
                    vuln_type = result['solution']['title']
                    if vuln_type not in vuln_types:
                        vuln_types[vuln_type] = {
                            'functions': [],
                            'solution': result['solution']
                        }
                    vuln_types[vuln_type]['functions'].append(result['function_name'])
            
            for vuln_type, data in vuln_types.items():
                with st.expander(f"🚨 {vuln_type} ({len(data['functions'])} functions affected)", expanded=False):
                    st.markdown(f"**Affected Functions:** {', '.join(data['functions'])}")
                    st.markdown(f"**Severity:** {data['solution']['severity']}")
                    st.markdown(f"**Description:** {data['solution']['description']}")
                    st.markdown("**How to Fix:**")
                    st.markdown(data['solution']['solution'])
        
        st.divider()
        
        # Function analysis results
        st.subheader("📋 Function Analysis Results")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            show_filter = st.selectbox(
                "Filter by:",
                ["All Functions", "Vulnerable Only", "Safe Only"]
            )
        
        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                ["Vulnerability Probability", "Function Name", "Confidence"]
            )
        
        # Filter and sort results
        filtered_results = results.copy()
        
        if show_filter == "Vulnerable Only":
            filtered_results = [r for r in results if r['prediction'] == 'Vulnerable']
        elif show_filter == "Safe Only":
            filtered_results = [r for r in results if r['prediction'] == 'Safe']
        
        if sort_by == "Vulnerability Probability":
            filtered_results.sort(key=lambda x: x['vulnerability_probability'], reverse=True)
        elif sort_by == "Function Name":
            filtered_results.sort(key=lambda x: x['function_name'])
        elif sort_by == "Confidence":
            filtered_results.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Display functions
        for i, result in enumerate(filtered_results):
            with st.container():
                # Function header
                col1, col2, col3 = st.columns([3, 1, 1])
                
                with col1:
                    st.markdown(f"### 📋 {result['function_name']}")
                
                with col2:
                    if result['prediction'] == 'Vulnerable':
                        st.markdown("🔴 **VULNERABLE**")
                    else:
                        st.markdown("🟢 **SAFE**")
                
                with col3:
                    vuln_prob = result['vulnerability_probability']
                    st.metric("Vuln. Prob.", f"{vuln_prob:.1%}")
                
                # Progress bar for vulnerability probability
                st.progress(vuln_prob, text=f"Vulnerability: {vuln_prob:.1%}")
                
                # Details
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                with col2:
                    st.metric("Code Length", f"{result['code_length']} chars")
                with col3:
                    st.metric("Start Line", result['start_line'])
                
                # Function code
                with st.expander("📄 View Function Code"):
                    st.code(result['function_code'], language='solidity')
                
                # Vulnerability solution (if applicable)
                if result.get('solution') and result['prediction'] == 'Vulnerable':
                    solution = result['solution']
                    with st.expander(f"🛠️ **{solution['title']} - How to Fix**", expanded=True):
                        # Severity badge
                        severity_color = {
                            'HIGH': '🔴',
                            'MEDIUM': '🟡',
                            'LOW': '🟢'
                        }
                        st.markdown(f"**Severity:** {severity_color.get(solution['severity'], '⚪')} {solution['severity']}")
                        
                        # Description
                        st.markdown(f"**Description:** {solution['description']}")
                        
                        # Solution
                        st.markdown(solution['solution'])
                
                st.divider()
        
        # Export options
        st.subheader("📊 Export Results")
        
        # Create DataFrame for export
        export_results = []
        for result in results:
            export_result = result.copy()
            # Add solution info to export
            if result.get('solution'):
                export_result['vulnerability_type'] = result['solution']['title']
                export_result['severity'] = result['solution']['severity']
                export_result['description'] = result['solution']['description']
            else:
                export_result['vulnerability_type'] = 'None'
                export_result['severity'] = 'None'
                export_result['description'] = 'No vulnerabilities detected'
            
            # Remove the solution object from export (too complex for CSV)
            export_result.pop('solution', None)
            export_results.append(export_result)
        
        df = pd.DataFrame(export_results)
        df['analysis_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv_data = df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV Report",
                data=csv_data,
                file_name=f"vulnerability_report_{contract_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON Report",
                data=json_data,
                file_name=f"vulnerability_report_{contract_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()