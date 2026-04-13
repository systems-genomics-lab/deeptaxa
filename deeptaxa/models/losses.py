"""
Module: losses.py

Description:
    Implements custom loss functions for DeepTaxa, including FocalLoss for handling class imbalance
    in hierarchical taxonomy classification. Configurable via CLI parameters, it integrates with
    the framework's multi-level prediction tasks.

    Focal loss addresses class imbalance by down-weighting well-classified examples and focusing
    on hard ones.
"""

import torch
import torch.nn as nn
import logging
from deeptaxa.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = DEFAULT_CONFIG["focal_gamma"], 
                 weight: torch.Tensor = None, reduction: str = 'mean') -> None:
        """
        Initialize FocalLoss with configurable parameters.

        Workflow:
            1. Set gamma parameter to control focus on hard examples.
            2. Optionally apply class weights for further imbalance correction.
            3. Define reduction method (mean or sum) for loss aggregation.

        Args:
            gamma (float): Focusing parameter (higher values increase focus on hard examples).
            weight (torch.Tensor, optional): Class weights for cross-entropy.
            reduction (str): Reduction method ('mean' or 'sum').
        """
        super().__init__()
        self.gamma = gamma  # Controls the down-weighting of easy examples
        self.weight = weight  # Optional per-class weights for additional balancing
        self.reduction = reduction  # Determines how loss is aggregated across batch
    
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss for a batch of predictions.

        Workflow:
            1. Calculate cross-entropy loss per sample.
            2. Compute modulating factor (1 - pt)^gamma based on prediction confidence.
            3. Apply focal weighting and reduce loss according to specified method.

        Args:
            input (torch.Tensor): Predicted logits [batch_size, num_classes].
            target (torch.Tensor): Ground truth labels [batch_size].

        Returns:
            torch.Tensor: Reduced focal loss scalar.
        """
        # Compute cross-entropy loss without reduction
        ce_loss = nn.functional.cross_entropy(input, target, weight=self.weight, reduction='none')
        
        # Calculate probability of true class (pt) and modulating factor
        pt = torch.exp(-ce_loss)  # pt = e^(-CE), confidence in correct prediction
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss  # (1 - pt)^gamma emphasizes hard examples
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        else:
            return focal_loss.sum()

