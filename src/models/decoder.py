"""
RNN Decoder components for generating character sequences from encoded features.

Implements LSTM decoders with attention mechanisms for converting encoded
image features into Khmer digit sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List
from abc import ABC, abstractmethod

from .attention import BahdanauAttention


class RNNDecoder(nn.Module, ABC):
    """
    Abstract base class for RNN decoders.
    
    Defines interface for decoding encoded features into character sequences.
    """
    
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 num_layers: int = 1):
        """
        Initialize RNN decoder.
        
        Args:
            vocab_size: Size of character vocabulary
            hidden_size: Size of hidden states  
            num_layers: Number of RNN layers
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
    
    @abstractmethod
    def forward(self, 
                encoder_features: torch.Tensor,
                target_sequence: Optional[torch.Tensor] = None,
                max_length: int = 9,
                allow_early_stopping: bool = True) -> torch.Tensor:
        """
        Decode encoder features into character sequences.
        
        Args:
            encoder_features: Encoded features from encoder
            target_sequence: Target sequence for training (optional)
            max_length: Maximum sequence length for inference
            
        Returns:
            Character predictions [batch_size, seq_len, vocab_size]
        """
        pass


class AttentionDecoder(RNNDecoder):
    """
    LSTM decoder with Bahdanau attention mechanism.
    
    Generates character sequences by attending to relevant encoder features
    at each decoding step.
    """
    
    def __init__(self,
                 vocab_size: int,
                 encoder_hidden_size: int,
                 decoder_hidden_size: int = 256,
                 num_layers: int = 1,
                 dropout: float = 0.1,
                 attention_size: int = 256):
        """
        Initialize attention decoder.
        
        Args:
            vocab_size: Size of character vocabulary
            encoder_hidden_size: Size of encoder hidden states
            decoder_hidden_size: Size of decoder hidden states
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            attention_size: Size of attention mechanism
        """
        super().__init__(vocab_size, decoder_hidden_size, num_layers)
        
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        
        # Character embedding
        self.embedding = nn.Embedding(vocab_size, decoder_hidden_size)
        
        # Attention mechanism
        self.attention = BahdanauAttention(
            encoder_hidden_size, decoder_hidden_size, attention_size
        )
        
        # LSTM decoder
        self.lstm = nn.LSTM(
            input_size=decoder_hidden_size + encoder_hidden_size,  # embedding + context
            hidden_size=decoder_hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Linear(
            decoder_hidden_size + encoder_hidden_size,  # hidden + context
            vocab_size
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Special token indices
        self.eos_token = vocab_size - 3  # <EOS> token
        self.pad_token = vocab_size - 2  # <PAD> token
        self.blank_token = vocab_size - 1  # <BLANK> token
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize decoder weights."""
        # Initialize embedding
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        
        # Initialize LSTM weights
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Initialize forget gate bias to 1
                with torch.no_grad():
                    n = param.size(0)
                    param[n//4:n//2].fill_(1.0)
        
        # Initialize output projection
        nn.init.xavier_uniform_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)
    
    def forward(self,
                encoder_features: torch.Tensor,
                target_sequence: Optional[torch.Tensor] = None,
                max_length: int = 9,
                allow_early_stopping: bool = True) -> torch.Tensor:
        """
        Decode encoder features into character sequences.
        
        Args:
            encoder_features: [batch_size, encoder_seq_len, encoder_hidden_size]
            target_sequence: [batch_size, target_seq_len] (for training)
            max_length: Maximum sequence length for inference
            
        Returns:
            Character predictions [batch_size, seq_len, vocab_size]
        """
        batch_size = encoder_features.size(0)
        device = encoder_features.device
        
        # Initialize decoder state
        hidden = self._init_hidden_state(batch_size, device)
        
        if self.training and target_sequence is not None:
            # Training mode: teacher forcing
            return self._forward_train(encoder_features, target_sequence, hidden)
        else:
            # Inference mode: autoregressive generation
            return self._forward_inference(encoder_features, max_length, hidden, allow_early_stopping)
    
    def _init_hidden_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize LSTM hidden state."""
        h0 = torch.zeros(self.num_layers, batch_size, self.decoder_hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.decoder_hidden_size, device=device)
        return (h0, c0)
    
    def _forward_train(self,
                      encoder_features: torch.Tensor,
                      target_sequence: torch.Tensor,
                      hidden: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Forward pass during training with teacher forcing."""
        batch_size, target_len = target_sequence.size()
        outputs = []
        
        # Start with SOS token (use first character as SOS)
        input_token = torch.zeros(batch_size, dtype=torch.long, device=target_sequence.device)
        
        for t in range(target_len):
            # Get character embedding
            embedded = self.embedding(input_token)  # [batch_size, hidden_size]
            embedded = self.dropout(embedded)
            
            # Compute attention
            context, attention_weights = self.attention(
                encoder_features, 
                hidden[0][-1]  # Last layer hidden state
            )
            
            # Combine embedding and context
            lstm_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
            
            # LSTM forward pass
            lstm_output, hidden = self.lstm(lstm_input, hidden)
            lstm_output = lstm_output.squeeze(1)  # [batch_size, hidden_size]
            
            # Output projection
            output_input = torch.cat([lstm_output, context], dim=-1)
            output = self.output_projection(output_input)  # [batch_size, vocab_size]
            outputs.append(output)
            
            # Teacher forcing: use target token as next input
            input_token = target_sequence[:, t]
        
        return torch.stack(outputs, dim=1)  # [batch_size, target_len, vocab_size]
    
    def _forward_inference(self,
                          encoder_features: torch.Tensor,
                          max_length: int,
                          hidden: Tuple[torch.Tensor, torch.Tensor],
                          allow_early_stopping: bool = True) -> torch.Tensor:
        """Forward pass during inference with autoregressive generation."""
        batch_size = encoder_features.size(0)
        device = encoder_features.device
        outputs = []
        
        # Start with SOS token (use blank token as SOS)
        input_token = torch.full((batch_size,), self.blank_token, dtype=torch.long, device=device)
        
        for t in range(max_length):
            # Get character embedding
            embedded = self.embedding(input_token)
            embedded = self.dropout(embedded)
            
            # Compute attention
            context, attention_weights = self.attention(
                encoder_features,
                hidden[0][-1]  # Last layer hidden state
            )
            
            # Combine embedding and context
            lstm_input = torch.cat([embedded, context], dim=-1).unsqueeze(1)
            
            # LSTM forward pass
            lstm_output, hidden = self.lstm(lstm_input, hidden)
            lstm_output = lstm_output.squeeze(1)
            
            # Output projection
            output_input = torch.cat([lstm_output, context], dim=-1)
            output = self.output_projection(output_input)
            outputs.append(output)
            
            # Get next input token
            input_token = torch.argmax(output, dim=-1)
            
            # Check for EOS token (only if early stopping is allowed)
            if allow_early_stopping and torch.all(input_token == self.eos_token):
                break
        
        return torch.stack(outputs, dim=1)  # [batch_size, seq_len, vocab_size]


class CTCDecoder(nn.Module):
    """
    CTC (Connectionist Temporal Classification) decoder.
    
    Alternative decoder using CTC loss for alignment-free training.
    Simpler than attention but less flexible for complex sequences.
    """
    
    def __init__(self,
                 vocab_size: int,
                 encoder_hidden_size: int,
                 dropout: float = 0.1):
        """
        Initialize CTC decoder.
        
        Args:
            vocab_size: Size of character vocabulary
            encoder_hidden_size: Size of encoder hidden states
            dropout: Dropout rate
        """
        super().__init__()
        
        self.vocab_size = vocab_size
        self.encoder_hidden_size = encoder_hidden_size
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(encoder_hidden_size, encoder_hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(encoder_hidden_size, vocab_size)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize CTC decoder weights."""
        for module in self.output_projection:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CTC decoder.
        
        Args:
            encoder_features: [batch_size, seq_len, encoder_hidden_size]
            
        Returns:
            Character logits [batch_size, seq_len, vocab_size]
        """
        # Apply output projection to each time step
        logits = self.output_projection(encoder_features)
        
        # Apply log softmax for CTC loss (CTC expects log probabilities)
        log_probs = F.log_softmax(logits, dim=-1)
        
        return log_probs


def create_decoder(decoder_type: str,
                  vocab_size: int,
                  encoder_hidden_size: int,
                  **kwargs) -> nn.Module:
    """
    Factory function to create decoder.
    
    Args:
        decoder_type: Type of decoder ('attention', 'ctc')
        vocab_size: Size of character vocabulary
        encoder_hidden_size: Size of encoder hidden states
        **kwargs: Additional arguments for the decoder
        
    Returns:
        Decoder instance
        
    Raises:
        ValueError: If decoder_type is not supported
    """
    if decoder_type.lower() in ['attention', 'attention_decoder']:
        return AttentionDecoder(vocab_size, encoder_hidden_size, **kwargs)
    elif decoder_type.lower() in ['ctc', 'ctc_decoder']:
        return CTCDecoder(vocab_size, encoder_hidden_size, **kwargs)
    else:
        raise ValueError(
            f"Unsupported decoder type: {decoder_type}. "
            f"Supported types: ['attention', 'ctc']"
        ) 