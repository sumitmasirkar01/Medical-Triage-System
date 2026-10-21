# core/open_clip_engine.py
"""
STAT-Triage AI: Open-Vocabulary Biomedical CLIP Foundation Engine.
Wraps Microsoft BiomedCLIP (PubMedBERT-256 + ViT-B/16-224) for vision-language encoding,
zero-shot clinical classification, and text prompt embedding generation.
"""

from typing import List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip


class OpenBiomedicalCLIPEngine(nn.Module):
    """
    BiomedCLIP wrapper handling multi-modal tokenization and feature extraction.
    """

    DEFAULT_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[torch.device] = None,
    ):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"[*] Initializing BiomedCLIP Foundation Weights [{model_name}]...", flush=True)
        self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(model_name)
        self.tokenizer = open_clip.get_tokenizer(model_name)

        self.model = self.model.to(self.device)
        self.model.eval()
        print("[✓] BiomedCLIP Vision-Language Engine loaded successfully.", flush=True)

    @torch.no_grad()
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encodes preprocessed 3-channel image tensors into 512-D normalized visual vectors.
        Args:
            images: (B, 3, 224, 224) or (3, 224, 224) float tensor.
        Returns:
            (B, 512) feature tensor.
        """
        if images.ndim == 3:
            images = images.unsqueeze(0)

        images = images.to(self.device).float()
        image_features = self.model.encode_image(images)
        return image_features

    @torch.no_grad()
    def encode_text(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Tokenizes and encodes diagnostic clinical prompts into 512-D text embeddings.
        Args:
            texts: Single prompt string or list of prompt strings.
        Returns:
            (N, 512) text feature tensor.
        """
        if isinstance(texts, str):
            texts = [texts]

        tokens = self.tokenizer(texts).to(self.device)
        text_features = self.model.encode_text(tokens)
        return text_features

    @torch.no_grad()
    def compute_zero_shot_similarity(
        self,
        images: torch.Tensor,
        text_prompts: List[str],
        temperature: float = 100.0,
    ) -> torch.Tensor:
        """
        Computes zero-shot softmax probabilities across text prompts.
        """
        img_emb = F.normalize(self.encode_image(images), p=2, dim=-1)
        txt_emb = F.normalize(self.encode_text(text_prompts), p=2, dim=-1)

        logits = (img_emb @ txt_emb.T) * (1.0 / self.model.logit_scale.exp().item() if hasattr(self.model, 'logit_scale') else 10.0)
        return F.softmax(logits, dim=-1)