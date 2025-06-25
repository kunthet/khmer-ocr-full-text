"""
Attention mechanisms for sequence-to-sequence modeling in OCR.

Implements Bahdanau (additive) attention for focusing on relevant
image regions during character sequence generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
import math


class BahdanauAttention(nn.Module):
    """
    Bahdanau (additive) attention mechanism.
    
    Computes attention weights between decoder hidden states and 
    encoder feature sequences to focus on relevant image regions.
    
    Reference: Bahdanau et al. "Neural Machine Translation by Jointly Learning to Align and Translate"
    """
    
    def __init__(self, 
                 encoder_hidden_size: int,
                 decoder_hidden_size: int, 
                 attention_size: int = 256):
        """
        Initialize Bahdanau attention.
        
        Args:
            encoder_hidden_size: Size of encoder hidden states
            decoder_hidden_size: Size of decoder hidden states  
            attention_size: Size of attention projection layer
        """
        super().__init__()
        
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.attention_size = attention_size
        
        # Linear projections for encoder and decoder states
        self.encoder_projection = nn.Linear(encoder_hidden_size, attention_size, bias=False)
        self.decoder_projection = nn.Linear(decoder_hidden_size, attention_size, bias=False)
        
        # Attention weight computation
        self.attention_weight = nn.Linear(attention_size, 1, bias=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        nn.init.xavier_uniform_(self.encoder_projection.weight)
        nn.init.xavier_uniform_(self.decoder_projection.weight)
        nn.init.xavier_uniform_(self.attention_weight.weight)
    
    def forward(self, 
                encoder_states: torch.Tensor,
                decoder_state: torch.Tensor,
                encoder_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute attention weights and context vector.
        
        Args:
            encoder_states: Encoder hidden states [batch_size, seq_len, encoder_hidden_size]
            decoder_state: Current decoder hidden state [batch_size, decoder_hidden_size]
            encoder_mask: Mask for encoder states [batch_size, seq_len] (optional)
            
        Returns:
            Tuple of (context_vector, attention_weights)
            - context_vector: [batch_size, encoder_hidden_size]
            - attention_weights: [batch_size, seq_len]
        """
        batch_size, seq_len, encoder_hidden_size = encoder_states.size()
        
        # Project encoder states
        # [batch_size, seq_len, attention_size]
        projected_encoder = self.encoder_projection(encoder_states)
        
        # Project decoder state and expand to match encoder sequence length
        # [batch_size, attention_size] -> [batch_size, 1, attention_size] -> [batch_size, seq_len, attention_size]
        projected_decoder = self.decoder_projection(decoder_state).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Compute attention scores
        # [batch_size, seq_len, attention_size]
        attention_input = torch.tanh(projected_encoder + projected_decoder)
        
        # [batch_size, seq_len, 1] -> [batch_size, seq_len]
        attention_scores = self.attention_weight(attention_input).squeeze(-1)
        
        # Apply mask if provided
        if encoder_mask is not None:
            attention_scores = attention_scores.masked_fill(encoder_mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute context vector
        # [batch_size, seq_len, 1] * [batch_size, seq_len, encoder_hidden_size] -> [batch_size, encoder_hidden_size]
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * encoder_states, dim=1)
        
        return context_vector, attention_weights


class EnhancedBahdanauAttention(BahdanauAttention):
    """
    Enhanced Bahdanau attention with additional features for complex character sequences.
    
    Includes:
    - Multi-layer attention computation
    - Gating mechanism for attention control
    - Coverage mechanism to prevent repetition
    """
    
    def __init__(self,
                 encoder_hidden_size: int,
                 decoder_hidden_size: int,
                 attention_size: int = 256,
                 num_layers: int = 2,
                 use_coverage: bool = True,
                 use_gating: bool = True):
        """
        Initialize enhanced Bahdanau attention.
        
        Args:
            encoder_hidden_size: Size of encoder hidden states
            decoder_hidden_size: Size of decoder hidden states
            attention_size: Size of attention projection layer
            num_layers: Number of attention layers
            use_coverage: Whether to use coverage mechanism
            use_gating: Whether to use gating mechanism
        """
        super().__init__(encoder_hidden_size, decoder_hidden_size, attention_size)
        
        self.num_layers = num_layers
        self.use_coverage = use_coverage
        self.use_gating = use_gating
        
        # Multi-layer attention computation
        if num_layers > 1:
            self.attention_layers = nn.ModuleList([
                nn.Linear(attention_size, attention_size) for _ in range(num_layers - 1)
            ])
        
        # Coverage mechanism
        if use_coverage:
            self.coverage_projection = nn.Linear(1, attention_size, bias=False)
        
        # Gating mechanism
        if use_gating:
            self.gate = nn.Sequential(
                nn.Linear(encoder_hidden_size + decoder_hidden_size, attention_size),
                nn.Sigmoid()
            )
    
    def forward(self,
                encoder_states: torch.Tensor,
                decoder_state: torch.Tensor,
                encoder_mask: Optional[torch.Tensor] = None,
                coverage: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Enhanced forward pass with coverage and gating.
        
        Args:
            encoder_states: Encoder hidden states [batch_size, seq_len, encoder_hidden_size]
            decoder_state: Current decoder hidden state [batch_size, decoder_hidden_size]
            encoder_mask: Mask for encoder states [batch_size, seq_len]
            coverage: Coverage vector from previous steps [batch_size, seq_len]
            
        Returns:
            Tuple of (context_vector, attention_weights, updated_coverage)
        """
        batch_size, seq_len, encoder_hidden_size = encoder_states.size()
        
        # Project encoder states
        projected_encoder = self.encoder_projection(encoder_states)
        
        # Project decoder state
        projected_decoder = self.decoder_projection(decoder_state).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Add coverage if enabled
        attention_input = projected_encoder + projected_decoder
        if self.use_coverage and coverage is not None:
            coverage_input = self.coverage_projection(coverage.unsqueeze(-1))
            attention_input = attention_input + coverage_input
        
        # Multi-layer attention computation
        attention_features = torch.tanh(attention_input)
        if hasattr(self, 'attention_layers'):
            for layer in self.attention_layers:
                attention_features = torch.tanh(layer(attention_features) + attention_features)
        
        # Compute attention scores
        attention_scores = self.attention_weight(attention_features).squeeze(-1)
        
        # Apply mask if provided
        if encoder_mask is not None:
            attention_scores = attention_scores.masked_fill(encoder_mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Compute context vector
        context_vector = torch.sum(attention_weights.unsqueeze(-1) * encoder_states, dim=1)
        
        # Apply gating if enabled
        if self.use_gating:
            gate_input = torch.cat([context_vector, decoder_state], dim=-1)
            gate_value = self.gate(gate_input)
            context_vector = gate_value * context_vector
        
        # Update coverage
        updated_coverage = coverage
        if self.use_coverage:
            if coverage is None:
                updated_coverage = attention_weights
            else:
                updated_coverage = coverage + attention_weights
        
        return context_vector, attention_weights, updated_coverage


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism for enhanced feature representation.
    
    Implements scaled dot-product attention with multiple heads for
    capturing different types of dependencies in character sequences.
    """
    
    def __init__(self,
                 d_model: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 temperature: float = 1.0):
        """
        Initialize multi-head attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
            temperature: Temperature for attention scaling
        """
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.temperature = temperature
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        # Output projection
        self.w_o = nn.Linear(d_model, d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize attention weights."""
        for module in [self.w_q, self.w_k, self.w_v, self.w_o]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-head attention forward pass.
        
        Args:
            query: Query tensor [batch_size, seq_len_q, d_model]
            key: Key tensor [batch_size, seq_len_k, d_model]
            value: Value tensor [batch_size, seq_len_v, d_model]
            mask: Attention mask [batch_size, seq_len_q, seq_len_k]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len_q, d_model = query.size()
        seq_len_k = key.size(1)
        
        # Store residual connection
        residual = query
        
        # Linear projections and reshape for multi-head
        Q = self.w_q(query).view(batch_size, seq_len_q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len_k, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        attention_output, attention_weights = self._scaled_dot_product_attention(
            Q, K, V, mask, self.temperature
        )
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len_q, d_model
        )
        
        # Final linear projection
        output = self.w_o(attention_output)
        output = self.dropout(output)
        
        # Residual connection and layer normalization
        output = self.layer_norm(output + residual)
        
        return output, attention_weights
    
    def _scaled_dot_product_attention(self,
                                    Q: torch.Tensor,
                                    K: torch.Tensor,
                                    V: torch.Tensor,
                                    mask: Optional[torch.Tensor] = None,
                                    temperature: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Scaled dot-product attention computation.
        
        Args:
            Q: Query tensor [batch_size, num_heads, seq_len_q, d_k]
            K: Key tensor [batch_size, num_heads, seq_len_k, d_k]
            V: Value tensor [batch_size, num_heads, seq_len_v, d_k]
            mask: Attention mask
            temperature: Attention temperature
            
        Returns:
            Tuple of (attention_output, attention_weights)
        """
        d_k = Q.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(d_k) * temperature)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:  # [batch_size, seq_len_q, seq_len_k]
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        return attention_output, attention_weights


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention mechanism for character-level and word-level features.
    
    Combines multiple attention mechanisms to capture both local character
    relationships and global word-level context.
    """
    
    def __init__(self,
                 feature_size: int,
                 char_attention_size: int = 256,
                 word_attention_size: int = 512,
                 num_heads: int = 8):
        """
        Initialize hierarchical attention.
        
        Args:
            feature_size: Size of input features
            char_attention_size: Size of character-level attention
            word_attention_size: Size of word-level attention
            num_heads: Number of attention heads
        """
        super().__init__()
        
        self.feature_size = feature_size
        self.char_attention_size = char_attention_size
        self.word_attention_size = word_attention_size
        
        # Character-level attention (local)
        self.char_attention = EnhancedBahdanauAttention(
            encoder_hidden_size=feature_size,
            decoder_hidden_size=feature_size,
            attention_size=char_attention_size,
            use_coverage=True,
            use_gating=True
        )
        
        # Word-level attention (global)
        self.word_attention = MultiHeadAttention(
            d_model=feature_size,
            num_heads=num_heads,
            dropout=0.1
        )
        
        # Feature fusion
        self.fusion = nn.Sequential(
            nn.Linear(feature_size * 2, feature_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_size, feature_size)
        )
        
        # Attention weight combination
        self.attention_combiner = nn.Linear(2, 1)
    
    def forward(self,
                encoder_features: torch.Tensor,
                decoder_state: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Hierarchical attention forward pass.
        
        Args:
            encoder_features: Encoder features [batch_size, seq_len, feature_size]
            decoder_state: Decoder state [batch_size, feature_size]
            mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Tuple of (context_vector, attention_info)
        """
        # Character-level attention
        char_context, char_weights, coverage = self.char_attention(
            encoder_features, decoder_state, mask
        )
        
        # Word-level attention (self-attention on encoder features)
        word_features, word_weights = self.word_attention(
            encoder_features, encoder_features, encoder_features, mask
        )
        word_context = word_features.mean(dim=1)  # Global pooling
        
        # Combine contexts
        combined_context = torch.cat([char_context, word_context], dim=-1)
        fused_context = self.fusion(combined_context)
        
        # Attention information
        attention_info = {
            'char_weights': char_weights,
            'word_weights': word_weights.mean(dim=1),  # Average over heads
            'coverage': coverage,
            'char_context': char_context,
            'word_context': word_context
        }
        
        return fused_context, attention_info


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer-like attention mechanisms.
    
    Adds positional information to feature representations to help
    attention mechanisms understand spatial relationships.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:x.size(0), :] 