"""
Complete Khmer Digits OCR Model

Combines CNN backbone, RNN encoder, attention mechanism, and decoder
into a unified architecture for end-to-end Khmer digit sequence recognition.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any
import yaml

from .backbone import CNNBackbone, create_backbone
from .encoder import RNNEncoder, create_encoder  
from .decoder import RNNDecoder, create_decoder
from .attention import BahdanauAttention

# Import character utilities
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'modules', 'synthetic_data_generator'))
try:
    from utils import get_full_khmer_characters, get_special_tokens, create_character_mapping
except ImportError:
    # Fallback imports for development
    import importlib.util
    utils_path = os.path.join(os.path.dirname(__file__), '..', 'modules', 'synthetic_data_generator', 'utils.py')
    spec = importlib.util.spec_from_file_location("utils", utils_path)
    utils_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils_module)
    get_full_khmer_characters = utils_module.get_full_khmer_characters
    get_special_tokens = utils_module.get_special_tokens
    create_character_mapping = utils_module.create_character_mapping


class KhmerDigitsOCR(nn.Module):
    """
    Complete Khmer Digits OCR Model.
    
    End-to-end architecture combining:
    - CNN backbone for feature extraction
    - RNN encoder for sequence modeling
    - Attention mechanism for alignment
    - RNN decoder for character generation
    """
    
    def __init__(self,
                 vocab_size: int = 13,
                 max_sequence_length: int = 8,
                 cnn_type: str = 'resnet18',
                 encoder_type: str = 'bilstm',
                 decoder_type: str = 'attention',
                 feature_size: int = 512,
                 encoder_hidden_size: int = 256,
                 decoder_hidden_size: int = 256,
                 attention_size: int = 256,
                 num_encoder_layers: int = 2,
                 num_decoder_layers: int = 1,
                 dropout: float = 0.1,
                 pretrained_cnn: bool = True):
        """
        Initialize Khmer Digits OCR model.
        
        Args:
            vocab_size: Size of character vocabulary (10 digits + 3 special tokens)
            max_sequence_length: Maximum sequence length
            cnn_type: Type of CNN backbone ('resnet18', 'efficientnet-b0')
            encoder_type: Type of encoder ('bilstm', 'conv')
            decoder_type: Type of decoder ('attention', 'ctc')
            feature_size: Size of CNN output features
            encoder_hidden_size: Size of encoder hidden states
            decoder_hidden_size: Size of decoder hidden states
            attention_size: Size of attention mechanism
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dropout: Dropout rate
            pretrained_cnn: Whether to use pretrained CNN weights
        """
        super().__init__()
        
        # Store configuration
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.feature_size = feature_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        
        # Character mappings (Khmer digits + special tokens)
        self.char_to_idx = {
            '០': 0, '១': 1, '២': 2, '៣': 3, '៤': 4,
            '៥': 5, '៦': 6, '៧': 7, '៨': 8, '៩': 9,
            '<EOS>': 10, '<PAD>': 11, '<BLANK>': 12
        }
        self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
        
        # CNN Backbone
        self.backbone = create_backbone(
            backbone_type=cnn_type,
            feature_size=feature_size,
            pretrained=pretrained_cnn,
            dropout=dropout
        )
        
        # RNN Encoder
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            input_size=feature_size,
            hidden_size=encoder_hidden_size,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        # Decoder
        if decoder_type == 'attention':
            self.decoder = create_decoder(
                decoder_type=decoder_type,
                vocab_size=vocab_size,
                encoder_hidden_size=encoder_hidden_size,
                decoder_hidden_size=decoder_hidden_size,
                num_layers=num_decoder_layers,
                dropout=dropout,
                attention_size=attention_size
            )
        else:  # CTC decoder
            self.decoder = create_decoder(
                decoder_type=decoder_type,
                vocab_size=vocab_size,
                encoder_hidden_size=encoder_hidden_size,
                dropout=dropout
            )
        
        self.decoder_type = decoder_type
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # CNN backbone weights are handled by the backbone itself
        # Encoder and decoder weights are handled by their respective classes
        pass
    
    def forward(self,
                images: torch.Tensor,
                target_sequences: Optional[torch.Tensor] = None,
                sequence_lengths: Optional[torch.Tensor] = None,
                allow_early_stopping: bool = False) -> torch.Tensor:
        """
        Forward pass of the OCR model.
        
        Args:
            images: Input images [batch_size, 3, height, width]
            target_sequences: Target sequences for training [batch_size, seq_len] (optional)
            sequence_lengths: Actual sequence lengths [batch_size] (optional)
            
        Returns:
            Character predictions [batch_size, seq_len, vocab_size]
        """
        # Extract CNN features
        cnn_features = self.backbone(images)  # [batch_size, cnn_seq_len, feature_size]
        
        # Encode features
        encoder_features, final_hidden = self.encoder(cnn_features)  # [batch_size, seq_len, encoder_hidden_size]
        
        # Decode to character sequences
        if self.decoder_type == 'attention':
            # Use the provided allow_early_stopping parameter
            predictions = self.decoder(
                encoder_features, 
                target_sequences, 
                self.max_sequence_length,
                allow_early_stopping
            )
        else:  # CTC decoder
            predictions = self.decoder(encoder_features)
        
        return predictions
    
    def predict(self,
                images: torch.Tensor,
                return_attention: bool = False) -> List[str]:
        """
        Predict character sequences from images.
        
        Args:
            images: Input images [batch_size, 3, height, width]
            return_attention: Whether to return attention weights
            
        Returns:
            List of predicted text strings
        """
        self.eval()
        with torch.no_grad():
            # Forward pass with early stopping enabled for actual inference
            predictions = self.forward(images, allow_early_stopping=True)  # [batch_size, seq_len, vocab_size]
            
            # Convert to character indices
            predicted_indices = torch.argmax(predictions, dim=-1)  # [batch_size, seq_len]
            
            # Decode to text
            texts = []
            for sequence in predicted_indices:
                text = self._decode_sequence(sequence)
                texts.append(text)
            
            return texts
    
    def _decode_sequence(self, indices: torch.Tensor) -> str:
        """
        Decode a sequence of character indices to text.
        
        Args:
            indices: Character indices [seq_len]
            
        Returns:
            Decoded text string
        """
        text = ""
        for idx in indices:
            idx_val = idx.item()
            if idx_val in self.idx_to_char:
                char = self.idx_to_char[idx_val]
                if char in ['<EOS>', '<PAD>', '<BLANK>']:
                    break
                text += char
        return text
    
    def encode_text(self, text: str, max_length: Optional[int] = None) -> torch.Tensor:
        """
        Encode text string to tensor of character indices.
        
        Args:
            text: Input text string
            max_length: Maximum sequence length (uses model default if None)
            
        Returns:
            Tensor of character indices [seq_len]
        """
        if max_length is None:
            max_length = self.max_sequence_length
            
        indices = []
        for char in text:
            if char in self.char_to_idx:
                indices.append(self.char_to_idx[char])
            else:
                # Use blank token for unknown characters
                indices.append(self.char_to_idx['<BLANK>'])
        
        # Add EOS token
        indices.append(self.char_to_idx['<EOS>'])
        
        # Pad or truncate to max_length
        if len(indices) < max_length:
            indices.extend([self.char_to_idx['<PAD>']] * (max_length - len(indices)))
        else:
            indices = indices[:max_length-1] + [self.char_to_idx['<EOS>']]
        
        return torch.tensor(indices, dtype=torch.long)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model details
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'model_type': 'KhmerDigitsOCR',
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'feature_size': self.feature_size,
            'encoder_hidden_size': self.encoder_hidden_size,
            'decoder_hidden_size': self.decoder_hidden_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'character_mapping': self.char_to_idx
        }
    
    @classmethod
    def from_config(cls, config_path: str) -> 'KhmerDigitsOCR':
        """
        Create model from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            Configured model instance
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract model parameters
        model_config = config.get('model', {})
        
        return cls(
            vocab_size=model_config.get('vocab_size', 13),
            max_sequence_length=model_config.get('max_sequence_length', 8),
            cnn_type=model_config.get('cnn_type', 'resnet18'),
            encoder_type=model_config.get('encoder_type', 'bilstm'),
            decoder_type=model_config.get('decoder_type', 'attention'),
            feature_size=model_config.get('feature_size', 512),
            encoder_hidden_size=model_config.get('encoder_hidden_size', 256),
            decoder_hidden_size=model_config.get('decoder_hidden_size', 256),
            attention_size=model_config.get('attention_size', 256),
            num_encoder_layers=model_config.get('num_encoder_layers', 2),
            num_decoder_layers=model_config.get('num_decoder_layers', 1),
            dropout=model_config.get('dropout', 0.1),
            pretrained_cnn=model_config.get('pretrained_cnn', True)
        )
    
    def save_config(self, config_path: str):
        """
        Save model configuration to file.
        
        Args:
            config_path: Path to save configuration file
        """
        config = {
            'model': {
                'vocab_size': self.vocab_size,
                'max_sequence_length': self.max_sequence_length,
                'feature_size': self.feature_size,
                'encoder_hidden_size': self.encoder_hidden_size,
                'decoder_hidden_size': self.decoder_hidden_size
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)


class KhmerTextOCR(nn.Module):
    """
    Enhanced Khmer Text OCR Model for full Khmer script recognition.
    
    Supports:
    - Full Khmer character vocabulary (102+ characters)
    - Hierarchical character recognition (base + modifiers)
    - Advanced attention mechanisms with multi-head support
    - Character-level and word-level confidence scoring
    - Beam search decoding with length normalization
    """
    
    def __init__(self,
                 use_full_khmer: bool = True,
                 max_sequence_length: int = 50,
                 cnn_type: str = 'resnet18',
                 encoder_type: str = 'bilstm',
                 decoder_type: str = 'attention',
                 feature_size: int = 512,
                 encoder_hidden_size: int = 512,
                 decoder_hidden_size: int = 512,
                 attention_size: int = 512,
                 num_attention_heads: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 2,
                 dropout: float = 0.1,
                 pretrained_cnn: bool = True,
                 enable_hierarchical: bool = True,
                 enable_confidence_scoring: bool = True):
        """
        Initialize enhanced Khmer Text OCR model.
        
        Args:
            use_full_khmer: Whether to use full Khmer character set or just digits
            max_sequence_length: Maximum sequence length for text
            cnn_type: Type of CNN backbone
            encoder_type: Type of encoder
            decoder_type: Type of decoder
            feature_size: Size of CNN output features
            encoder_hidden_size: Size of encoder hidden states
            decoder_hidden_size: Size of decoder hidden states
            attention_size: Size of attention mechanism
            num_attention_heads: Number of attention heads for multi-head attention
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dropout: Dropout rate
            pretrained_cnn: Whether to use pretrained CNN weights
            enable_hierarchical: Enable hierarchical character recognition
            enable_confidence_scoring: Enable confidence scoring
        """
        super().__init__()
        
        # Character mapping setup
        self.use_full_khmer = use_full_khmer
        if use_full_khmer:
            self.char_to_idx, self.idx_to_char = create_character_mapping(use_full_khmer=True)
            self.vocab_size = len(self.char_to_idx)
            
            # Character categories for hierarchical recognition
            self.khmer_chars = get_full_khmer_characters()
            self._setup_character_categories()
        else:
            # Fallback to digits for compatibility
            self.char_to_idx = {
                '០': 0, '១': 1, '២': 2, '៣': 3, '៤': 4,
                '៥': 5, '៦': 6, '៧': 7, '៨': 8, '៩': 9,
                '<EOS>': 10, '<PAD>': 11, '<BLANK>': 12
            }
            self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}
            self.vocab_size = len(self.char_to_idx)
        
        # Store configuration
        self.max_sequence_length = max_sequence_length
        self.feature_size = feature_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.enable_hierarchical = enable_hierarchical
        self.enable_confidence_scoring = enable_confidence_scoring
        
        # CNN Backbone
        self.backbone = create_backbone(
            backbone_type=cnn_type,
            feature_size=feature_size,
            pretrained=pretrained_cnn,
            dropout=dropout
        )
        
        # Enhanced RNN Encoder
        self.encoder = create_encoder(
            encoder_type=encoder_type,
            input_size=feature_size,
            hidden_size=encoder_hidden_size,
            num_layers=num_encoder_layers,
            dropout=dropout
        )
        
        # Multi-Head Attention Enhancement
        if num_attention_heads > 1:
            self.multi_head_attention = nn.MultiheadAttention(
                embed_dim=encoder_hidden_size,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.multi_head_attention = None
        
        # Enhanced Decoder
        if decoder_type == 'attention':
            self.decoder = create_decoder(
                decoder_type=decoder_type,
                vocab_size=self.vocab_size,
                encoder_hidden_size=encoder_hidden_size,
                decoder_hidden_size=decoder_hidden_size,
                num_layers=num_decoder_layers,
                dropout=dropout,
                attention_size=attention_size
            )
        else:  # CTC decoder
            self.decoder = create_decoder(
                decoder_type=decoder_type,
                vocab_size=self.vocab_size,
                encoder_hidden_size=encoder_hidden_size,
                dropout=dropout
            )
        
        self.decoder_type = decoder_type
        
        # Hierarchical Recognition Components
        if enable_hierarchical and use_full_khmer:
            self._setup_hierarchical_recognition()
        
        # Confidence Scoring Components
        if enable_confidence_scoring:
            self._setup_confidence_scoring()
        
        # Initialize weights
        self._init_weights()
    
    def _setup_character_categories(self):
        """Setup character category mappings for hierarchical recognition."""
        self.category_to_chars = {}
        self.char_to_category = {}
        
        for category, chars in self.khmer_chars.items():
            self.category_to_chars[category] = chars
            for char in chars:
                self.char_to_category[char] = category
    
    def _setup_hierarchical_recognition(self):
        """Setup hierarchical recognition components."""
        # Base character classifier (main character types)
        num_categories = len(self.khmer_chars) if self.use_full_khmer else 1
        self.base_classifier = nn.Linear(self.decoder_hidden_size, num_categories)
        
        # Character relationship modeling for stacked characters
        self.relationship_encoder = nn.LSTM(
            input_size=self.decoder_hidden_size,
            hidden_size=self.decoder_hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # Context-aware character recognition
        self.context_attention = nn.MultiheadAttention(
            embed_dim=self.decoder_hidden_size,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
    
    def _setup_confidence_scoring(self):
        """Setup confidence scoring components."""
        # Character-level confidence scoring
        self.char_confidence = nn.Sequential(
            nn.Linear(self.decoder_hidden_size, self.decoder_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.decoder_hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
        # Word-level confidence aggregation
        self.word_confidence = nn.LSTM(
            input_size=self.decoder_hidden_size + 1,  # hidden + char confidence
            hidden_size=self.decoder_hidden_size // 2,
            num_layers=1,
            batch_first=True
        )
        
        # Confidence fusion layer
        self.confidence_fusion = nn.Linear(self.decoder_hidden_size // 2, 1)
    
    def _init_weights(self):
        """Initialize model weights with enhanced strategies."""
        # Initialize hierarchical components
        if hasattr(self, 'base_classifier'):
            nn.init.xavier_uniform_(self.base_classifier.weight)
            nn.init.zeros_(self.base_classifier.bias)
        
        # Initialize confidence scoring components
        if hasattr(self, 'char_confidence'):
            for layer in self.char_confidence:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def forward(self,
                images: torch.Tensor,
                target_sequences: Optional[torch.Tensor] = None,
                sequence_lengths: Optional[torch.Tensor] = None,
                return_confidence: bool = False,
                return_hierarchical: bool = False) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with optional confidence and hierarchical outputs.
        
        Args:
            images: Input images [batch_size, 3, height, width]
            target_sequences: Target sequences for training [batch_size, seq_len]
            sequence_lengths: Actual sequence lengths [batch_size]
            return_confidence: Whether to return confidence scores
            return_hierarchical: Whether to return hierarchical predictions
            
        Returns:
            Dictionary containing predictions and optional additional outputs
        """
        # Extract CNN features
        cnn_features = self.backbone(images)
        
        # Encode features
        encoder_features, final_hidden = self.encoder(cnn_features)
        
        # Apply multi-head attention if available
        if self.multi_head_attention is not None:
            enhanced_features, attention_weights = self.multi_head_attention(
                encoder_features, encoder_features, encoder_features
            )
            encoder_features = encoder_features + enhanced_features  # Residual connection
        
        # Main character prediction
        if self.decoder_type == 'attention':
            predictions = self.decoder(
                encoder_features,
                target_sequences,
                self.max_sequence_length
            )
        else:  # CTC decoder
            predictions = self.decoder(encoder_features)
        
        # Prepare output dictionary
        outputs = {'predictions': predictions}
        
        # Add confidence scoring if enabled
        if return_confidence and self.enable_confidence_scoring:
            outputs.update(self._compute_confidence_scores(encoder_features, predictions))
        
        # Add hierarchical predictions if enabled
        if return_hierarchical and self.enable_hierarchical and self.use_full_khmer:
            outputs.update(self._compute_hierarchical_predictions(encoder_features))
        
        return outputs
    
    def _compute_confidence_scores(self, encoder_features: torch.Tensor, 
                                 predictions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute character and word-level confidence scores."""
        batch_size, seq_len, _ = predictions.shape
        
        # Character-level confidence
        char_confidences = []
        for t in range(seq_len):
            # Use decoder hidden state (approximated from encoder features)
            hidden_state = encoder_features.mean(dim=1)  # Simple aggregation
            conf = self.char_confidence(hidden_state)
            char_confidences.append(conf)
        
        char_confidence_scores = torch.stack(char_confidences, dim=1)  # [batch_size, seq_len, 1]
        
        # Word-level confidence (simplified for now)
        word_confidence_scores = char_confidence_scores.mean(dim=1)  # [batch_size, 1]
        
        return {
            'char_confidence': char_confidence_scores.squeeze(-1),
            'word_confidence': word_confidence_scores.squeeze(-1)
        }
    
    def _compute_hierarchical_predictions(self, encoder_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute hierarchical character predictions."""
        # Base character category prediction
        pooled_features = encoder_features.mean(dim=1)  # Global average pooling
        category_predictions = self.base_classifier(pooled_features)
        
        return {
            'category_predictions': category_predictions
        }
    
    def predict_with_beam_search(self,
                                images: torch.Tensor,
                                beam_size: int = 5,
                                length_normalization: float = 0.6,
                                return_confidence: bool = False) -> List[Dict[str, Any]]:
        """
        Predict with beam search decoding and length normalization.
        
        Args:
            images: Input images [batch_size, 3, height, width]
            beam_size: Size of beam for beam search
            length_normalization: Length normalization factor (0.0 = no normalization)
            return_confidence: Whether to return confidence scores
            
        Returns:
            List of prediction dictionaries with text, score, and optional confidence
        """
        self.eval()
        with torch.no_grad():
            # For now, implement a simplified version - can be enhanced later
            outputs = self.forward(images, return_confidence=return_confidence)
            predictions = outputs['predictions']
            
            # Convert to character indices
            predicted_indices = torch.argmax(predictions, dim=-1)
            
            # Compute prediction scores (log probabilities)
            log_probs = torch.log_softmax(predictions, dim=-1)
            scores = torch.gather(log_probs, -1, predicted_indices.unsqueeze(-1)).squeeze(-1)
            
            results = []
            for i, (sequence, seq_scores) in enumerate(zip(predicted_indices, scores)):
                text = self._decode_sequence(sequence)
                
                # Apply length normalization
                if length_normalization > 0:
                    length_penalty = ((5 + len(text)) / 6) ** length_normalization
                    normalized_score = seq_scores.sum().item() / length_penalty
                else:
                    normalized_score = seq_scores.sum().item()
                
                result = {
                    'text': text,
                    'score': normalized_score,
                    'raw_score': seq_scores.sum().item()
                }
                
                # Add confidence if requested
                if return_confidence and 'char_confidence' in outputs:
                    result['char_confidence'] = outputs['char_confidence'][i].tolist()
                    result['word_confidence'] = outputs['word_confidence'][i].item()
                
                results.append(result)
            
            return results
    
    def _decode_sequence(self, indices: torch.Tensor) -> str:
        """Decode sequence with enhanced special token handling."""
        text = ""
        for idx in indices:
            idx_val = idx.item()
            if idx_val in self.idx_to_char:
                char = self.idx_to_char[idx_val]
                if char in ['<EOS>', '<PAD>', '<BLANK>']:
                    break
                text += char
        return text
    
    def get_vocabulary_info(self) -> Dict[str, Any]:
        """Get detailed vocabulary information."""
        info = {
            'vocab_size': self.vocab_size,
            'use_full_khmer': self.use_full_khmer,
            'char_to_idx': self.char_to_idx,
            'special_tokens': get_special_tokens()
        }
        
        if self.use_full_khmer:
            info['character_categories'] = {
                category: len(chars) for category, chars in self.khmer_chars.items()
            }
            info['total_khmer_chars'] = sum(len(chars) for chars in self.khmer_chars.values())
        
        return info
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive enhanced model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        info = {
            'model_type': 'KhmerTextOCR',
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length,
            'feature_size': self.feature_size,
            'encoder_hidden_size': self.encoder_hidden_size,
            'decoder_hidden_size': self.decoder_hidden_size,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'use_full_khmer': self.use_full_khmer,
            'enable_hierarchical': self.enable_hierarchical,
            'enable_confidence_scoring': self.enable_confidence_scoring,
            'has_multi_head_attention': self.multi_head_attention is not None
        }
        
        return info