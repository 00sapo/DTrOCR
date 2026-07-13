import torch
import numpy as np

from dataclasses import dataclass
from typing import Optional, Union, List
from transformers.utils import ModelOutput


@dataclass
class DTrOCRModelOutput(ModelOutput):
    hidden_states: torch.FloatTensor = None
    past_key_values: Optional[torch.FloatTensor] = None


@dataclass
class DTrOCRLMHeadModelOutput(ModelOutput):
    logits: torch.FloatTensor = None
    loss: Optional[torch.FloatTensor] = None
    accuracy: Optional[torch.FloatTensor] = None
    past_key_values: Optional[torch.FloatTensor] = None


@dataclass
class DTrOCRProcessorOutput:
    pixel_values: Optional[torch.FloatTensor] = None
    input_ids: Optional[Union[torch.LongTensor, np.ndarray, List[int]]] = None
    attention_mask: Optional[Union[torch.FloatTensor, np.ndarray, List[int]]] = None
    labels: Optional[Union[torch.LongTensor, np.ndarray, List[int]]] = None
