
"""
Vision metrics evaluation module for image similarity assessment.

This module provides evaluators for computing similarity metrics between images
including CLIP, DINO, and CSD (Content-Style Decomposition) evaluations.
All evaluators are implemented as singletons for efficient memory usage.
"""

from typing import Tuple

import clip
import torch
from CSD.loss_utils import transforms_branch0
from CSD.model import CSD_CLIP
from CSD.utils import convert_state_dict
from torchvision import transforms


class CLIPEvaluator:
    """
    CLIP-based image and text similarity evaluator.

    This class implements a singleton pattern to ensure only one instance
    of the CLIP model is loaded in memory.
    """

    _instance = None

    def __new__(cls, device: torch.device, clip_model: str = "ViT-B/32"):
        if cls._instance is None:
            cls._instance = super(CLIPEvaluator, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, device: torch.device, clip_model: str = "ViT-B/32") -> None:
        """
        Initialize the CLIP evaluator.

        Args:
            device: PyTorch device for computation (GPU/CPU)
            clip_model: CLIP model variant to use (default: "ViT-B/32")
        """
        if self.initialized:
            return

        self.device = device
        self.model, clip_preprocess = clip.load(clip_model, device=self.device)

        self.clip_preprocess = clip_preprocess
        self.preprocess = transforms.Compose(
            [transforms.Normalize(mean=[-1.0, -1.0, -1.0], std=[2.0, 2.0, 2.0])]
            + clip_preprocess.transforms[:2]
            + clip_preprocess.transforms[4:]
        )

        self.initialized = True

    @torch.no_grad()
    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        """
        Extract image features using CLIP encoder.

        Args:
            img: Input image tensor of shape [B, C, H, W]
            norm: Whether to normalize features (default: True)

        Returns:
            Image features tensor of shape [B, feature_dim]
        """
        images = self.preprocess(img).to(self.device)
        image_features = self.model.encode_image(images)

        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)

        return image_features

    @torch.no_grad()
    def get_text_features(self, text: str, norm: bool = True) -> torch.Tensor:
        """
        Extract text features using CLIP encoder.

        Args:
            text: Input text string
            norm: Whether to normalize features (default: True)

        Returns:
            Text features tensor of shape [1, feature_dim]
        """
        tokens = clip.tokenize(text, truncate=True).to(self.device)

        text_features = self.model.encode_text(tokens).detach()

        if norm:
            text_features /= text_features.norm(dim=-1, keepdim=True)

        return text_features

    def img_to_img_similarity(
        self, src_images: torch.Tensor, generated_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate image-to-image similarity using CLIP features.

        Args:
            src_images: Source images tensor of shape [B, C, H, W]
            generated_images: Generated images tensor of shape [B, C, H, W]

        Returns:
            Average similarity score tensor
        """
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)
        return (src_img_features @ gen_img_features.T).mean()

    def txt_to_img_similarity(self, text: str, generated_images: torch.Tensor) -> torch.Tensor:
        """
        Calculate text-to-image similarity using CLIP features.

        Args:
            text: Input text string
            generated_images: Generated images tensor of shape [B, C, H, W]

        Returns:
            Average similarity score tensor
        """
        text_features = self.get_text_features(text)
        gen_img_features = self.get_image_features(generated_images)

        return (text_features @ gen_img_features.T).mean()


class DINOEvaluator:
    """
    DINO-based image similarity evaluator.

    This class implements a singleton pattern to ensure only one instance
    of the DINO model is loaded in memory.
    """

    _instance = None

    def __new__(cls, device: torch.device):
        if cls._instance is None:
            cls._instance = super(DINOEvaluator, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self, device: torch.device) -> None:
        """
        Initialize the DINO evaluator.

        Args:
            device: PyTorch device for computation (GPU/CPU)
        """
        if self.initialized:
            return

        self.device = device
        self.model = torch.hub.load("facebookresearch/dino:main", "dino_vits16").to(device)
        self.model.eval()

        # Define preprocessing
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.initialized = True

    @torch.no_grad()
    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        """
        Extract image features using DINO encoder.

        Args:
            img: Input image tensor of shape [B, C, H, W]
            norm: Whether to normalize features (default: True)

        Returns:
            Image features tensor of shape [B, feature_dim]
        """
        img = self.preprocess(img)
        features = self.model(img)

        if norm:
            features /= features.clone().norm(dim=-1, keepdim=True)

        return features

    def img_to_img_similarity(
        self, src_images: torch.Tensor, generated_images: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate image-to-image similarity using DINO features.

        Args:
            src_images: Source images tensor of shape [B, C, H, W]
            generated_images: Generated images tensor of shape [B, C, H, W]

        Returns:
            Average similarity score tensor
        """
        src_img_features = self.get_image_features(src_images)
        gen_img_features = self.get_image_features(generated_images)
        return (src_img_features @ gen_img_features.T).mean()


class CSDEvaluator:
    _instance = None

    def __new__(
        cls, device, model_path="evaluator/CSD/csd_models/pytorch_model.bin", arch="vit_large"
    ):
        if cls._instance is None:
            cls._instance = super(CSDEvaluator, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(
        self, device, model_path="evaluator/CSD/csd_models/pytorch_model.bin", arch="vit_large"
    ):
        print(f"Loading CSD model from {model_path}")
        if self.initialized:
            return

        self.device = device
        # Load CSD model
        self.model = CSD_CLIP(arch, "default")
        checkpoint = torch.load(model_path, map_location=device)
        state_dict = convert_state_dict(checkpoint["model_state_dict"])
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()
        self.model = self.model.to(device)

        self.preprocess = transforms_branch0
        self.initialized = True

    @torch.no_grad()
    def extract_embeddings(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract content and style embeddings from image."""
        # If input is a batch of images, process each one
        if len(img.shape) == 4:  # [B, C, H, W]
            # Convert tensor to PIL for CSD preprocessing
            imgs = [transforms.ToPILImage()(img[i]) for i in range(img.shape[0])]
            # Preprocess and get embeddings for batch
            img_tensors = (
                torch.stack([self.preprocess(img).unsqueeze(0) for img in imgs])
                .squeeze(1)
                .to(self.device)
            )
            content_embeddings, _, style_embeddings = self.model(img_tensors)
            return content_embeddings.squeeze(), style_embeddings.squeeze()
        else:  # Single image [C, H, W]
            if isinstance(img, torch.Tensor):
                img = transforms.ToPILImage()(img)
            img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
            content_embedding, _, style_embedding = self.model(img_tensor)
            return content_embedding.squeeze(), style_embedding.squeeze()

    def calculate_similarities(
        self, src_images: torch.Tensor, trg_image: torch.Tensor
    ) -> Tuple[float, float]:
        """
        Calculate average content and style similarities between multiple source images and one target image.
        Args:
            src_images: Source images tensor of shape [B, C, H, W]
            trg_image: Single target image tensor of shape [C, H, W]
        Returns:
            Tuple of (content_similarity, style_similarity) averaged over all source images
        """
        # Extract embeddings for source images
        src_content_embs, src_style_embs = self.extract_embeddings(src_images)  # [B, D]

        # Extract embeddings for target image
        trg_content_emb, trg_style_emb = self.extract_embeddings(trg_image)  # [D]

        # Ensure embeddings have correct shape for similarity calculation
        if len(src_content_embs.shape) == 1:
            src_content_embs = src_content_embs.unsqueeze(0)
            src_style_embs = src_style_embs.unsqueeze(0)

        # Calculate similarities between each source image and the target
        content_sims = torch.nn.functional.cosine_similarity(
            src_content_embs, trg_content_emb.unsqueeze(0), dim=1  # [B, D]  # [1, D]
        )

        style_sims = torch.nn.functional.cosine_similarity(
            src_style_embs, trg_style_emb.unsqueeze(0), dim=1  # [B, D]  # [1, D]
        )

        # Take mean of similarities
        content_sim = content_sims.mean().item()
        style_sim = style_sims.mean().item()

        return content_sim, style_sim
