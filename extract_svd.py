
"""
SVD-based projection matrix extraction for CLIP models.

This module provides functionality to extract projection matrices from CLIP text
or image embeddings using Singular Value Decomposition (SVD). It supports both
top and least significant singular vectors for creating projection matrices
that can be used to modify embedding spaces.
"""

import os
from typing import List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


def encode_text_features(
    text_list: List[str], model: CLIPModel, processor: CLIPProcessor, device: str = "cpu"
) -> torch.Tensor:
    """
    Encode text descriptions using CLIP model.

    Args:
        text_list: List of text strings to encode
        model: CLIP model for text encoding
        processor: CLIP processor for text preprocessing
        device: Device to run the model on (default: "cpu")

    Returns:
        Text embeddings tensor of shape (n_texts, embedding_dim)
    """
    with torch.no_grad():
        inputs = processor(text=text_list, images=None, return_tensors="pt", padding=True)
        text_embeddings = model.get_text_features(
            **{k: v.to(device) for k, v in inputs.items() if k.startswith("input")}
        )
    return text_embeddings


def encode_image_features(
    image_paths: List[str], model: CLIPModel, processor: CLIPProcessor, device: str = "cpu"
) -> torch.Tensor:
    """
    Encode images using CLIP model.

    Args:
        image_paths: List of image file paths to encode
        model: CLIP model for image encoding
        processor: CLIP processor for image preprocessing
        device: Device to run the model on (default: "cpu")

    Returns:
        Image embeddings tensor of shape (n_images, embedding_dim)
    """
    embeddings_list = []

    with torch.no_grad():
        for img_path in image_paths:
            image = Image.open(img_path).convert("RGB")
            inputs = processor(images=image, text=None, return_tensors="pt", padding=True)
            image_embeddings = model.get_image_features(
                **{k: v.to(device) for k, v in inputs.items() if k.startswith("pixel")}
            )
            embeddings_list.append(image_embeddings)

    return torch.cat(embeddings_list, dim=0)


def compute_projection_matrix(
    input_list: List[str],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: str = "cpu",
    r: int = 50,
    input_type: str = "text",
    use_least: bool = False,
) -> np.ndarray:
    """
    Constructs the projection matrix given a list of texts or image paths.

    Args:
        input_list: List of text strings or image file paths
        model: CLIP model for encoding
        processor: CLIP processor for preprocessing
        device: Device to run the model on (default: "cpu")
        r: Number of singular vectors to use (default: 50)
        input_type: Type of input, either "text" or "image" (default: "text")
        use_least: If True, use the least significant singular vectors (default: False)

    Returns:
        Projection matrix of shape (embedding_dim, embedding_dim)
    """
    model = model.to(device)

    if input_type == "text":
        # Clean the text list - strip whitespace and remove empty lines
        input_list = [text.strip() for text in input_list if text.strip()]
        embeddings = encode_text_features(input_list, model, processor, device)
    else:  # input_type == "image"
        embeddings = encode_image_features(input_list, model, processor, device)

    E = embeddings.cpu().numpy().astype(np.float32)
    print(f"Embeddings shape: {E.shape}")
    U, Sigma, Vt = np.linalg.svd(E, full_matrices=False)
    print(f"Singular values: {Sigma}")

    V_r = Vt[-r:, :] if use_least else Vt[:r, :]

    P_c = V_r.T @ V_r
    return P_c


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """
    Computes cosine similarity between two 1-D numpy vectors.

    Args:
        u: First vector
        v: Second vector

    Returns:
        Cosine similarity value between -1 and 1

    Raises:
        ValueError: If vectors have different shapes or are empty
    """
    if u.shape != v.shape or u.size == 0:
        raise ValueError("Vectors must have the same shape and be non-empty")

    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))


def save_projection_matrix(P: np.ndarray, filepath: str) -> None:
    """
    Save projection matrix to a file using PyTorch.

    Args:
        P: Projection matrix as numpy array
        filepath: Path to save the matrix

    Raises:
        OSError: If unable to create directory or save file
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Convert numpy array to torch tensor
        P_tensor = torch.from_numpy(P)
        torch.save(P_tensor, filepath.replace(".npy", ".pt"))
        print(f"Saved projection matrix to {filepath.replace('.npy', '.pt')}")
    except OSError as e:
        raise OSError(f"Failed to save projection matrix: {e}")


def load_projection_matrix(filepath: str) -> np.ndarray:
    """
    Load projection matrix from a file using PyTorch.

    Args:
        filepath: Path to the saved matrix file

    Returns:
        Projection matrix as numpy array

    Raises:
        FileNotFoundError: If the file doesn't exist
        RuntimeError: If the file cannot be loaded
    """
    try:
        P_tensor = torch.load(filepath.replace(".npy", ".pt"))
        P = P_tensor.numpy()
        print(f"Loaded projection matrix from {filepath.replace('.npy', '.pt')}")
        return P
    except FileNotFoundError:
        raise FileNotFoundError(f"Projection matrix file not found: {filepath}")
    except RuntimeError as e:
        raise RuntimeError(f"Failed to load projection matrix: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compute projection matrix from text or images")
    parser.add_argument(
        "--input_type",
        type=str,
        choices=["text", "image"],
        default="text",
        help="Type of input to process (text or image)",
    )
    parser.add_argument("--top_r", type=int, default=10, help="Number of singular vectors to use")
    parser.add_argument(
        "--input_dir_data",
        type=str,
        default="benchmark_data/test",
        help="Input directory containing data subdirectories (default: benchmark_data/test)",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_configs = [
        {
            "name": "openai/clip-vit-large-patch14",
            "proj_suffix": f"proj_top_{args.top_r}.pt",
            "proj_least_suffix": f"proj_least_{args.top_r}.pt",
        },
        {
            "name": "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
            "proj_suffix": f"proj_2_top_{args.top_r}.pt",
            "proj_least_suffix": f"proj_least_2_{args.top_r}.pt",
        },
    ]

    dir_data = args.input_dir_data
    data_paths = [
        f"{dir_data}/{d}" for d in os.listdir(dir_data) if os.path.isdir(f"{dir_data}/{d}")
    ]

    for config in tqdm(model_configs, desc="Processing models"):
        print(f"\nProcessing model: {config['name']}")

        model = CLIPModel.from_pretrained(config["name"]).to(device)
        processor = CLIPProcessor.from_pretrained(config["name"])

        for data_path in tqdm(data_paths, desc="Processing data paths"):
            print(f"\nProcessing {data_path}")

            if args.input_type == "text":
                if not os.path.exists(f"{data_path}/variation_content.txt"):
                    print(
                        f"Skipping {data_path} because it does not have a variation_content.txt file"
                    )
                    continue

                with open(f"{data_path}/variation_content.txt", "r") as f:
                    variations = [line.strip() for line in f.readlines() if line.strip()]

                print("Loaded text variations:", variations)
            else:  # input_type == "image"
                img_dir = f"{data_path}/images"
                if not os.path.exists(img_dir):
                    print(f"Warning: Image directory not found at {img_dir}")
                    continue
                variations = [
                    os.path.join(img_dir, f)
                    for f in os.listdir(img_dir)
                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
                ]
                print(f"Found {len(variations)} images in {img_dir}")

            print(f"Computing projection matrix using top {args.top_r} singular vectors...")
            P = compute_projection_matrix(
                variations,
                model,
                processor,
                device=device,
                r=args.top_r,
                input_type=args.input_type,
                use_least=False,
            )
            save_projection_matrix(P, f"{data_path}/{config['proj_suffix']}")

            print(f"Computing projection matrix using least {args.top_r} singular vectors...")
            P_least = compute_projection_matrix(
                variations,
                model,
                processor,
                device=device,
                r=args.top_r,
                input_type=args.input_type,
                use_least=True,
            )
            save_projection_matrix(P_least, f"{data_path}/{config['proj_least_suffix']}")

            if config == model_configs[0]:
                P_test = load_projection_matrix(f"{data_path}/{config['proj_suffix']}")

                if args.input_type == "text":
                    text1 = "An Aegean"
                    text2 = "A Brazilian Shorthair"

                    with torch.no_grad():
                        emb1 = encode_text_features([text1], model, processor, device)[0]
                        emb2 = encode_text_features([text2], model, processor, device)[0]

                    emb1_np = emb1.cpu().numpy()
                    emb2_np = emb2.cpu().numpy()

                    print(f"Text 1: {text1}")
                    print(f"Text 2: {text2}")

                else:  # input_type == "image"
                    if len(variations) >= 2:
                        with torch.no_grad():
                            emb1 = encode_image_features([variations[0]], model, processor, device)[
                                0
                            ]
                            emb2 = encode_image_features([variations[1]], model, processor, device)[
                                0
                            ]

                        emb1_np = emb1.cpu().numpy()
                        emb2_np = emb2.cpu().numpy()

                        print(f"Image 1: {variations[0]}")
                        print(f"Image 2: {variations[1]}")
                    else:
                        print("Not enough images for testing similarity")
                        continue

                original_sim = cosine_similarity(emb1_np, emb2_np)
                emb1_proj = emb1_np @ P_test
                emb2_proj = emb2_np @ P_test
                projected_sim = cosine_similarity(emb1_proj, emb2_proj)

                print(f"Cosine similarity BEFORE projection:  {original_sim:.4f}")
                print(f"Cosine similarity AFTER projection:   {projected_sim:.4f}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
