"""
RNN Encoder components for processing CNN feature sequences.

Implements bidirectional LSTM encoders for converting spatial CNN features
into contextual sequence representations for OCR decoding.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
from abc import ABC, abstractmethod


class RNNEncoder(nn.Module, ABC):
    """
    Abstract base class for RNN encoders.
    
    Defines the interface for encoding CNN feature sequences
    into contextual representations.
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 1,
                 dropout: float = 0.0):
        """
        Initialize RNN encoder.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden states
            num_layers: Number of RNN layers
            dropout: Dropout rate between layers
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
    
    @abstractmethod
    def forward(self, 
                features: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input feature sequences.
        
        Args:
            features: Input features [batch_size, seq_len, input_size]
            lengths: Sequence lengths [batch_size] (optional)
            
        Returns:
            Tuple of (encoded_features, final_hidden_state)
        """
        pass
    
    @property
    @abstractmethod
    def output_size(self) -> int:
        """Return the output feature size."""
        pass


class BiLSTMEncoder(RNNEncoder):
    """
    Bidirectional LSTM encoder.
    
    Processes feature sequences in both forward and backward directions
    to capture complete contextual information for each position.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 2,
                 dropout: float = 0.1,
                 batch_first: bool = True):
        """
        Initialize bidirectional LSTM encoder.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden states (per direction)
            num_layers: Number of LSTM layers
            dropout: Dropout rate between layers
            batch_first: Whether batch dimension is first
        """
        super().__init__(input_size, hidden_size, num_layers, dropout)
        
        self.batch_first = batch_first
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=batch_first
        )
        
        # Output projection to combine bidirectional features
        self.output_projection = nn.Linear(hidden_size * 2, hidden_size)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize LSTM weights using Xavier initialization."""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
                # Initialize forget gate bias to 1 for better gradient flow
                with torch.no_grad():
                    n = param.size(0)
                    param[n//4:n//2].fill_(1.0)
    
    def forward(self, 
                features: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input feature sequences with bidirectional LSTM.
        
        Args:
            features: Input features [batch_size, seq_len, input_size]
            lengths: Sequence lengths [batch_size] (optional)
            
        Returns:
            Tuple of (encoded_features, final_hidden_state)
            - encoded_features: [batch_size, seq_len, hidden_size]
            - final_hidden_state: [batch_size, hidden_size]
        """
        batch_size, seq_len, _ = features.size()
        
        # Process without packing for simplicity in this implementation
        lstm_output, (hidden, cell) = self.lstm(features)
        
        # Project bidirectional output to target size
        # lstm_output: [batch_size, seq_len, hidden_size * 2]
        encoded_features = self.output_projection(lstm_output)  # [batch_size, seq_len, hidden_size]
        
        # Apply layer normalization and dropout
        encoded_features = self.layer_norm(encoded_features)
        encoded_features = self.dropout_layer(encoded_features)
        
        # Combine final hidden states from both directions
        # hidden: [num_layers * 2, batch_size, hidden_size]
        final_hidden_forward = hidden[-2]  # Last layer, forward direction
        final_hidden_backward = hidden[-1]  # Last layer, backward direction
        
        # Combine forward and backward final states
        final_hidden_combined = torch.cat([final_hidden_forward, final_hidden_backward], dim=-1)
        final_hidden_state = self.output_projection(final_hidden_combined)  # [batch_size, hidden_size]
        
        return encoded_features, final_hidden_state
    
    @property
    def output_size(self) -> int:
        """Return the output feature size."""
        return self.hidden_size


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for processing 2D feature maps.
    
    Alternative to RNN encoder using 1D convolutions along
    the sequence dimension for faster processing.
    """
    
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int = 3,
                 kernel_size: int = 3,
                 dropout: float = 0.1):
        """
        Initialize convolutional encoder.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of hidden features
            num_layers: Number of conv layers
            kernel_size: Convolution kernel size
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Convolutional layers
        layers = []
        in_channels = input_size
        
        for i in range(num_layers):
            layers.extend([
                nn.Conv1d(in_channels, hidden_size, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            in_channels = hidden_size
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Final projection
        self.output_projection = nn.Linear(hidden_size, hidden_size)
        
    def forward(self, 
                features: torch.Tensor,
                lengths: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode features using 1D convolutions.
        
        Args:
            features: Input features [batch_size, seq_len, input_size]
            lengths: Sequence lengths [batch_size] (optional)
            
        Returns:
            Tuple of (encoded_features, final_state)
        """
        # Transpose for conv1d: [batch_size, input_size, seq_len]
        features = features.transpose(1, 2)
        
        # Apply convolutions
        encoded = self.conv_layers(features)
        
        # Transpose back: [batch_size, seq_len, hidden_size]
        encoded = encoded.transpose(1, 2)
        
        # Apply final projection
        encoded_features = self.output_projection(encoded)
        
        # Compute final state as mean over sequence
        if lengths is not None:
            # Mask padded positions
            mask = torch.arange(encoded.size(1), device=encoded.device).unsqueeze(0) < lengths.unsqueeze(1)
            masked_encoded = encoded * mask.unsqueeze(-1).float()
            final_state = masked_encoded.sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            final_state = encoded.mean(dim=1)
        
        return encoded_features, final_state
    
    @property
    def output_size(self) -> int:
        """Return the output feature size."""
        return self.hidden_size


def create_encoder(encoder_type: str,
                  input_size: int,
                  hidden_size: int,
                  **kwargs) -> RNNEncoder:
    """
    Factory function to create encoder.
    
    Args:
        encoder_type: Type of encoder ('bilstm', 'conv')
        input_size: Size of input features
        hidden_size: Size of hidden features
        **kwargs: Additional arguments for the encoder
        
    Returns:
        Encoder instance
        
    Raises:
        ValueError: If encoder_type is not supported
    """
    if encoder_type.lower() in ['bilstm', 'bidirectional_lstm']:
        return BiLSTMEncoder(input_size, hidden_size, **kwargs)
    elif encoder_type.lower() == 'conv':
        return ConvEncoder(input_size, hidden_size, **kwargs)
    else:
        raise ValueError(
            f"Unsupported encoder type: {encoder_type}. "
            f"Supported types: ['bilstm', 'conv']"
        ) 