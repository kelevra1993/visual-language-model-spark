import torch

from rich.console import Console
from rich.panel import Panel
from torch import nn
from typing import Any, Dict, Optional

# Temporal modules summarising a sequence into one vector without keeping its time axis, many to one only
POOLING_TEMPORAL_TYPES = ["mean_pooling", "max_pooling", "attention_pooling"]

# Temporal modules modelling the order of the frames, able to summarise a sequence (many to one) or to keep one
# vector per frame (many to many)
ORDERED_TEMPORAL_TYPES = ["gru", "lstm", "transformer", "temporal_convolution"]


def print_sequence_module_summary(title: str, module_configuration: Dict[str, Any], output_features: int,
                                  border_style: str) -> None:
    """
    Prints the settings of a module of the sequence models in their architecture summary (the temporal module, the
    fusion of the members of a pair), with the number of features of the vectors it gives to the next module. The
    'pretrained' settings are left out, the source run of a transferred module being printed with the experiment
    information.

    Args:
        title (str): The title of the panel, e.g. "Temporal Module".
        module_configuration (Dict[str, Any]): The configuration section of the module, e.g. 'TemporalModel'.
        output_features (int): The number of features of the output vectors of the module.
        border_style (str): The colour of the panel border.
    """
    summary_text = "\n".join(f"[bold]{key_name.replace('_', ' ').title():<18}[/bold] : {key_value}"
                             for key_name, key_value in module_configuration.items() if key_name != "pretrained")
    summary_text += f"\n[bold]{'Output Features':<18}[/bold] : {output_features}"
    Console(width=80).print(Panel(summary_text, title=f"[bold]{title}[/bold]", border_style=border_style))


def compute_masked_mean(features: torch.Tensor, sequence_mask: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Averages per-frame vectors over the real frames of every sequence, the padding frames being left out.

    Args:
        features (torch.Tensor): The [B, T, F] per-frame vectors.
        sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, or None when every
            frame is real.

    Returns:
        torch.Tensor: The [B, F] mean vectors.
    """
    if sequence_mask is None:
        return features.mean(dim=1)
    frame_weights = sequence_mask.to(dtype=features.dtype).unsqueeze(dim=-1)
    return (features * frame_weights).sum(dim=1) / frame_weights.sum(dim=1).clamp(min=1.0)


class MaskedMeanPooling(nn.Module):
    """
    Temporal module of the many to one task averaging the frame vectors of every sequence. It ignores the order of
    the frames, which makes it the baseline telling whether a task needs the temporal modules modelling the order.
    """

    def __init__(self, input_features: int) -> None:
        """
        Keeps the number of features of the frame vectors, unchanged by the pooling.

        Args:
            input_features (int): The number of features of the frame vectors.
        """
        super(MaskedMeanPooling, self).__init__()
        self.output_features = input_features

    def forward(self, frame_features: torch.Tensor,
                sequence_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Averages the vectors of the real frames.

        Args:
            frame_features (torch.Tensor): The [B, T, F] frame vectors.
            sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, or None when every
                frame is real.

        Returns:
            torch.Tensor: The [B, F] sequence vectors.
        """
        return compute_masked_mean(features=frame_features, sequence_mask=sequence_mask)


class MaskedMaxPooling(nn.Module):
    """
    Temporal module of the many to one task keeping, for every feature, its maximum over the real frames of every
    sequence, so that a feature present in a single frame is kept. Like the mean pooling, it ignores the frame order.
    """

    def __init__(self, input_features: int) -> None:
        """
        Keeps the number of features of the frame vectors, unchanged by the pooling.

        Args:
            input_features (int): The number of features of the frame vectors.
        """
        super(MaskedMaxPooling, self).__init__()
        self.output_features = input_features

    def forward(self, frame_features: torch.Tensor,
                sequence_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Takes the maximum of every feature over the real frames.

        Args:
            frame_features (torch.Tensor): The [B, T, F] frame vectors.
            sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, or None when every
                frame is real.

        Returns:
            torch.Tensor: The [B, F] sequence vectors.
        """
        if sequence_mask is None:
            return frame_features.max(dim=1).values
        masked_features = frame_features.masked_fill(mask=~sequence_mask.unsqueeze(dim=-1), value=float("-inf"))
        return masked_features.max(dim=1).values


class AttentionPooling(nn.Module):
    """
    Temporal module of the many to one task averaging the frame vectors of every sequence with learned weights: a
    small network scores every frame, and the softmax of the scores over the real frames weights the average, so that
    the informative frames dominate. It ignores the frame order.
    """

    def __init__(self, input_features: int, hidden_size: int, device: torch.device, dtype: torch.dtype) -> None:
        """
        Builds the frame scoring network.

        Args:
            input_features (int): The number of features of the frame vectors.
            hidden_size (int): The hidden units of the scoring network.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(AttentionPooling, self).__init__()
        self.output_features = input_features
        self.frame_scoring = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_size, device=device, dtype=dtype), nn.Tanh(),
            nn.Linear(in_features=hidden_size, out_features=1, device=device, dtype=dtype))

    def forward(self, frame_features: torch.Tensor,
                sequence_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Averages the vectors of the real frames, weighted by the softmax of their scores.

        Args:
            frame_features (torch.Tensor): The [B, T, F] frame vectors.
            sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, or None when every
                frame is real.

        Returns:
            torch.Tensor: The [B, F] sequence vectors.
        """
        frame_scores = self.frame_scoring(input=frame_features).squeeze(dim=-1)
        if sequence_mask is not None:
            frame_scores = frame_scores.masked_fill(mask=~sequence_mask, value=float("-inf"))
        frame_weights = torch.softmax(input=frame_scores, dim=1)
        return (frame_weights.unsqueeze(dim=-1) * frame_features).sum(dim=1)


class RecurrentTemporalModule(nn.Module):
    """
    Temporal module reading the frame vectors in order with a GRU or an LSTM. For many to one, the sequence vector is
    the final state of the last layer after the last real frame (both directions concatenated when bidirectional);
    for many to many, every frame keeps the output of the last layer at its position. The padding frames are skipped
    through packed sequences, which requires the real frames to come first, as the sequence dataset gives them.
    """

    def __init__(self, recurrent_type: str, input_features: int, hidden_size: int, number_layers: int,
                 dropout_rate: float, bidirectional: bool, keep_time_axis: bool, device: torch.device,
                 dtype: torch.dtype) -> None:
        """
        Builds the recurrent network.

        Args:
            recurrent_type (str): "gru" or "lstm".
            input_features (int): The number of features of the frame vectors.
            hidden_size (int): The number of features of the recurrent state.
            number_layers (int): The number of stacked recurrent layers.
            dropout_rate (float): The dropout between two stacked layers.
            bidirectional (bool): Whether to also read the sequence backwards, which looks at the future frames.
            keep_time_axis (bool): Whether to return one vector per frame (many to many) instead of per sequence.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(RecurrentTemporalModule, self).__init__()
        self.keep_time_axis = keep_time_axis
        self.number_directions = 2 if bidirectional else 1
        self.output_features = hidden_size * self.number_directions
        recurrent_class = {"gru": nn.GRU, "lstm": nn.LSTM}[recurrent_type]
        self.recurrent_network = recurrent_class(input_size=input_features, hidden_size=hidden_size,
                                                 num_layers=number_layers, batch_first=True,
                                                 dropout=dropout_rate if number_layers > 1 else 0.0,
                                                 bidirectional=bidirectional, device=device, dtype=dtype)

    def forward(self, frame_features: torch.Tensor,
                sequence_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Reads the real frames of every sequence in order.

        Args:
            frame_features (torch.Tensor): The [B, T, F] frame vectors.
            sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, which come first, or
                None when every frame is real.

        Returns:
            torch.Tensor: The [B, output_features] sequence vectors, or the [B, T, output_features] frame vectors
                (zeros on the padding frames) when keeping the time axis.
        """
        # Every frame is real: read the whole tensor, without packing (the path of the exported models)
        if sequence_mask is None:
            frame_outputs, final_state = self.recurrent_network(input=frame_features)
            if self.keep_time_axis:
                return frame_outputs
        else:
            sequence_lengths = sequence_mask.sum(dim=1).cpu()
            packed_features = nn.utils.rnn.pack_padded_sequence(input=frame_features, lengths=sequence_lengths,
                                                                batch_first=True, enforce_sorted=False)
            packed_outputs, final_state = self.recurrent_network(input=packed_features)

            # One output per frame, back in the padded layout
            if self.keep_time_axis:
                frame_outputs, _ = nn.utils.rnn.pad_packed_sequence(sequence=packed_outputs, batch_first=True,
                                                                    total_length=frame_features.shape[1])
                return frame_outputs

        # Final state of the last layer in every direction, the LSTM state also holding its cell state
        hidden_state = final_state[0] if isinstance(final_state, tuple) else final_state
        last_layer_states = hidden_state[-self.number_directions:]
        return last_layer_states.transpose(0, 1).reshape(frame_features.shape[0], self.output_features)


class TransformerTemporalModule(nn.Module):
    """
    Temporal module relating every frame to the other frames with a transformer encoder: the frame vectors are
    projected to the model size, a learned positional embedding gives them their position, and the padding frames are
    masked out of the attention. When causal, every frame only attends to itself and the frames before it. For many to
    one, the sequence vector is the mean of the outputs of the real frames.
    """

    def __init__(self, input_features: int, hidden_size: int, number_layers: int, number_heads: int,
                 dropout_rate: float, causal: bool, keep_time_axis: bool, maximum_length: int, device: torch.device,
                 dtype: torch.dtype) -> None:
        """
        Builds the input projection, the positional embedding and the encoder layers.

        Args:
            input_features (int): The number of features of the frame vectors.
            hidden_size (int): The model size of the transformer, a multiple of the number of heads.
            number_layers (int): The number of encoder layers.
            number_heads (int): The number of attention heads of every layer.
            dropout_rate (float): The dropout of the attention and feed forward layers.
            causal (bool): Whether every frame only attends to the frames before it (online use).
            keep_time_axis (bool): Whether to return one vector per frame (many to many) instead of per sequence.
            maximum_length (int): The maximum number of frames of a sequence, the size of the positional embedding.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(TransformerTemporalModule, self).__init__()
        self.causal = causal
        self.keep_time_axis = keep_time_axis
        self.output_features = hidden_size

        # Projection of the frame vectors to the model size, and learned position of every frame
        self.input_projection = nn.Linear(in_features=input_features, out_features=hidden_size, device=device,
                                          dtype=dtype)
        self.positional_embedding = nn.Parameter(data=torch.randn(size=(1, maximum_length, hidden_size),
                                                                  device=device, dtype=dtype) * 0.02)

        # Pre-normalization encoder layers with a feed forward layer four times wider than the model
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=number_heads,
                                                   dim_feedforward=4 * hidden_size, dropout=dropout_rate,
                                                   batch_first=True, norm_first=True, device=device, dtype=dtype)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=number_layers,
                                             enable_nested_tensor=False)

    def forward(self, frame_features: torch.Tensor,
                sequence_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Encodes the frames of every sequence, each frame attending to the real frames it may see.

        Args:
            frame_features (torch.Tensor): The [B, T, F] frame vectors.
            sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, or None when every
                frame is real.

        Returns:
            torch.Tensor: The [B, hidden_size] sequence vectors, or the [B, T, hidden_size] frame vectors when keeping
                the time axis.
        """
        sequence_length = frame_features.shape[1]
        frame_tokens = self.input_projection(input=frame_features) + self.positional_embedding[:, :sequence_length]

        # Forbid the attention to the future frames when causal (True marks a forbidden pair)
        attention_mask = None
        if self.causal:
            attention_mask = torch.triu(input=torch.ones(size=(sequence_length, sequence_length), dtype=torch.bool,
                                                         device=frame_features.device), diagonal=1)

        # Telling the encoder whether the mask is causal spares it a comparison of the mask values, which the ONNX
        # export cannot trace
        encoded_frames = self.encoder(src=frame_tokens, mask=attention_mask,
                                      src_key_padding_mask=None if sequence_mask is None else ~sequence_mask,
                                      is_causal=self.causal)
        if self.keep_time_axis:
            return encoded_frames
        return compute_masked_mean(features=encoded_frames, sequence_mask=sequence_mask)


class TemporalConvolutionModule(nn.Module):
    """
    Temporal module convolving the frame vectors over time: stacked 1D convolutions whose dilation doubles at every
    layer, so that the number of frames seen grows exponentially with the depth. The padding frames are zeroed before
    every convolution. When causal, the convolutions are padded on the past side only, so that every frame only sees
    the frames before it. For many to one, the sequence vector is the mean of the outputs of the real frames.
    """

    def __init__(self, input_features: int, hidden_size: int, number_layers: int, kernel_size: int,
                 dropout_rate: float, causal: bool, keep_time_axis: bool, device: torch.device,
                 dtype: torch.dtype) -> None:
        """
        Builds the stacked convolutions.

        Args:
            input_features (int): The number of features of the frame vectors.
            hidden_size (int): The number of output channels of every convolution.
            number_layers (int): The number of stacked convolutions, the layer i having a dilation of 2^i.
            kernel_size (int): The odd number of frames covered by a convolution kernel.
            dropout_rate (float): The dropout after every convolution.
            causal (bool): Whether every frame only sees the frames before it (online use).
            keep_time_axis (bool): Whether to return one vector per frame (many to many) instead of per sequence.
            device (torch.device): The device of the parameters.
            dtype (torch.dtype): The precision of the parameters.
        """
        super(TemporalConvolutionModule, self).__init__()
        self.kernel_size = kernel_size
        self.causal = causal
        self.keep_time_axis = keep_time_axis
        self.output_features = hidden_size
        self.dilations = [2 ** layer_index for layer_index in range(number_layers)]
        self.convolutions = nn.ModuleList([
            nn.Conv1d(in_channels=input_features if layer_index == 0 else hidden_size, out_channels=hidden_size,
                      kernel_size=kernel_size, dilation=dilation, device=device, dtype=dtype)
            for layer_index, dilation in enumerate(self.dilations)])
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, frame_features: torch.Tensor,
                sequence_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Convolves the frame vectors of every sequence over time.

        Args:
            frame_features (torch.Tensor): The [B, T, F] frame vectors.
            sequence_mask (Optional[torch.Tensor]): The [B, T] booleans, True for the real frames, or None when every
                frame is real.

        Returns:
            torch.Tensor: The [B, hidden_size] sequence vectors, or the [B, T, hidden_size] frame vectors (zeros on
                the padding frames) when keeping the time axis.
        """
        features = frame_features.transpose(1, 2)

        # Weight of every frame, 1 for the real frames and 0 for the padding frames, all ones without mask
        if sequence_mask is None:
            frame_weights = torch.ones_like(input=features[:, :1])
        else:
            frame_weights = sequence_mask.to(dtype=features.dtype).unsqueeze(dim=1)

        # Pad every convolution so that it keeps the number of frames, on the past side only when causal
        for convolution, dilation in zip(self.convolutions, self.dilations):
            total_padding = (self.kernel_size - 1) * dilation
            padding = (total_padding, 0) if self.causal else (total_padding // 2, total_padding - total_padding // 2)
            padded_features = nn.functional.pad(input=features * frame_weights, pad=padding)
            features = self.dropout(input=torch.relu(input=convolution(input=padded_features)))

        frame_outputs = (features * frame_weights).transpose(1, 2)
        if self.keep_time_axis:
            return frame_outputs
        return compute_masked_mean(features=frame_outputs, sequence_mask=sequence_mask)


def build_temporal_module(temporal_configuration: Dict[str, Any], input_features: int, maximum_length: int,
                          keep_time_axis: bool, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """
    Builds the temporal module of a sequence model from its 'TemporalModel' section: a pooling for many to one only,
    or a module modelling the frame order (GRU, LSTM, transformer, temporal convolution) for both sequence tasks.
    Every module maps [B, T, F] frame vectors and their [B, T] sequence mask to [B, F'] sequence vectors, or to
    [B, T, F'] frame vectors when keeping the time axis, and gives F' in its 'output_features' attribute.

    Args:
        temporal_configuration (Dict[str, Any]): The 'TemporalModel' section: 'type', 'hidden_size', 'number_layers',
            'number_heads', 'kernel_size', 'dropout_rate', 'bidirectional' and 'causal'.
        input_features (int): The number of features of the frame vectors.
        maximum_length (int): The maximum number of frames of a sequence.
        keep_time_axis (bool): Whether to return one vector per frame (many to many) instead of per sequence.
        device (torch.device): The device of the parameters.
        dtype (torch.dtype): The precision of the parameters.

    Returns:
        nn.Module: The temporal module.
    """
    temporal_type = temporal_configuration["type"]

    # Poolings, ignoring the frame order
    if temporal_type == "mean_pooling":
        return MaskedMeanPooling(input_features=input_features)
    if temporal_type == "max_pooling":
        return MaskedMaxPooling(input_features=input_features)
    if temporal_type == "attention_pooling":
        return AttentionPooling(input_features=input_features, hidden_size=temporal_configuration["hidden_size"],
                                device=device, dtype=dtype)

    # Modules modelling the frame order
    if temporal_type in ["gru", "lstm"]:
        return RecurrentTemporalModule(recurrent_type=temporal_type, input_features=input_features,
                                       hidden_size=temporal_configuration["hidden_size"],
                                       number_layers=temporal_configuration["number_layers"],
                                       dropout_rate=temporal_configuration["dropout_rate"],
                                       bidirectional=temporal_configuration["bidirectional"],
                                       keep_time_axis=keep_time_axis, device=device, dtype=dtype)
    if temporal_type == "transformer":
        return TransformerTemporalModule(input_features=input_features,
                                         hidden_size=temporal_configuration["hidden_size"],
                                         number_layers=temporal_configuration["number_layers"],
                                         number_heads=temporal_configuration["number_heads"],
                                         dropout_rate=temporal_configuration["dropout_rate"],
                                         causal=temporal_configuration["causal"], keep_time_axis=keep_time_axis,
                                         maximum_length=maximum_length, device=device, dtype=dtype)
    return TemporalConvolutionModule(input_features=input_features, hidden_size=temporal_configuration["hidden_size"],
                                     number_layers=temporal_configuration["number_layers"],
                                     kernel_size=temporal_configuration["kernel_size"],
                                     dropout_rate=temporal_configuration["dropout_rate"],
                                     causal=temporal_configuration["causal"], keep_time_axis=keep_time_axis,
                                     device=device, dtype=dtype)
