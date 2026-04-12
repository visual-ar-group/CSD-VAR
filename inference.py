
"""
Evaluation script for content-style disentanglement using Switti pipeline.

This script evaluates the disentanglement capabilities of trained Switti models
by generating images from various prompts and saving results with comprehensive
metadata for further analysis.
"""

import argparse
import json
import os
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import safetensors
import torch
import yaml
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import make_grid

from models import SwittiDisentangledPipeline

EVAL_PROMPT_PATH = "benchmark_data/benchmark_prompts.json"


def add_caption_to_image(image: torch.Tensor, caption: str, font_size: int = 20) -> torch.Tensor:
    """
    Add caption text to an image tensor.

    Args:
        image: Input image tensor
        caption: Text caption to add
        font_size: Font size for the caption (default: 20)

    Returns:
        Image tensor with caption added
    """
    # Convert tensor to PIL Image if needed
    if isinstance(image, torch.Tensor):
        image = (image * 255).clamp(0, 255).to(torch.uint8)
        image = image.permute(1, 2, 0).cpu().numpy()
        image = Image.fromarray(image)

    # Create new image with space for caption
    margin = 10
    width = image.width
    height = image.height + font_size + 2 * margin
    new_image = Image.new("RGB", (width, height), "white")
    new_image.paste(image, (0, 0))

    # Add caption
    draw = ImageDraw.Draw(new_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
    except:
        font = ImageFont.load_default()

    # Center the text
    text_width = draw.textlength(caption, font=font)
    x = (width - text_width) // 2
    y = height - font_size - margin

    draw.text((x, y), caption, fill="black", font=font)

    # Convert back to tensor
    new_image = torch.from_numpy(np.array(new_image)).permute(2, 0, 1).float() / 255.0
    return new_image


def get_number_of_vecs(path_ckpt: str) -> int:
    """
    Get the number of vectors from a checkpoint file.

    Args:
        path_ckpt: Path to the checkpoint file

    Returns:
        Number of vectors in the checkpoint
    """
    dict_vecs = safetensors.torch.load_file(path_ckpt, device="cpu")
    # get first key
    first_key = next(iter(dict_vecs))

    return dict_vecs[first_key].shape[0]


def get_identifier(num_vecs: int, is_content: bool = True) -> str:
    """
    Generate identifier string based on number of vectors and type.

    Args:
        num_vecs: Number of vectors
        is_content: Whether this is for content (True) or style (False)

    Returns:
        Identifier string with appropriate placeholders
    """
    if is_content:
        placeholders = ["<object>"]
        for i in range(num_vecs - 1):
            placeholders.append(f"<object>_{i+1}")
        return " ".join(placeholders)
    else:
        placeholders = ["<style>"]
        for i in range(num_vecs - 1):
            placeholders.append(f"<style>_{i+1}")
        return " ".join(placeholders)


def parse_caption_from_csv(csv_path: str) -> Tuple[str, str]:
    """
    Parse caption from CSV file to extract content and style descriptions.

    Example: from "A white cat {} in fantastic {} style"
    Returns: ("white cat", "fantastic")

    Args:
        csv_path: Path to the CSV file containing captions

    Returns:
        Tuple of (content_description, style_description)
    """
    # Read CSV file
    df = pd.read_csv(csv_path)

    # Get the first caption_full (assuming we want the first row)
    caption = df["caption_full"].iloc[0]

    # Extract content (between 'A' and first '{}')
    content_start = caption.find("A ") + 2
    content_end = caption.find("{}")
    content_desc = caption[content_start:content_end].strip()

    # Extract style (between 'in' and second '{}')
    style_start = caption.find("in ") + 3
    style_end = caption.find("{}", content_end + 2)  # Start search after first {}
    style_desc = caption[style_start:style_end].strip()

    return content_desc, style_desc


def get_eval_prompts(
    eval_prompts_path: str,
    identifier_content: str,
    identifier_style: str,
    captions_path: Optional[str] = None,
    is_live_subject: bool = False,
) -> Tuple[List[str], Optional[str], Optional[str], List[str], List[str]]:
    """
    Get evaluation prompts with proper formatting and identifiers.

    Args:
        eval_prompts_path: Path to the evaluation prompts JSON file
        identifier_content: Content identifier string
        identifier_style: Style identifier string
        captions_path: Optional path to captions CSV file
        is_live_subject: Whether this is a live subject evaluation

    Returns:
        Tuple containing (prompts, content_desc, style_desc, prompt_types, mode_prompts)
    """
    with open(eval_prompts_path, "r") as f:
        eval_prompts = json.load(f)

    content_prompts_recon = eval_prompts["content"]["recontextualization"]
    content_prompts_style = eval_prompts["content"]["stylization"]
    content_prompts_nonrigid = eval_prompts["content"]["non-rigid"]

    content_prompts = []
    prompt_types = []

    content_prompts.extend(content_prompts_recon)
    prompt_types.extend(["content_recontextualization"] * len(content_prompts_recon))

    content_prompts.extend(content_prompts_style)
    prompt_types.extend(["content_stylization"] * len(content_prompts_style))

    if is_live_subject:
        content_prompts.extend(content_prompts_nonrigid)
        prompt_types.extend(["content_non-rigid"] * len(content_prompts_nonrigid))

    style_prompts = eval_prompts["style"]
    prompt_types.extend(["style"] * len(style_prompts))

    content_desc = None
    style_desc = None
    if captions_path is not None:
        content_desc, style_desc = parse_caption_from_csv(captions_path)

    mode_prompts = []
    for i in range(len(content_prompts)):
        if captions_path is not None:
            content_prompts[i] = content_prompts[i].replace("{}", f"{content_desc} {{}}")
        content_prompts[i] = content_prompts[i].format(identifier_content)
        mode_prompts.append("content")
        print(content_prompts[i])

    for i in range(len(style_prompts)):
        if captions_path is not None:
            style_prompts[i] = style_prompts[i].replace("{}", f"{style_desc} {{}}")
        style_prompts[i] = style_prompts[i].format(identifier_style)
        mode_prompts.append("style")
        print(style_prompts[i])

    prompts = content_prompts + style_prompts

    return prompts, content_desc, style_desc, prompt_types, mode_prompts


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Evaluate content-style disentanglement")
    parser.add_argument(
        "--prompts_path", type=str, default=EVAL_PROMPT_PATH, help="Path to prompts file"
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run evaluation on")
    parser.add_argument("--model_path", type=str, default="yresearch/Switti", help="Path to model")
    parser.add_argument("--step", type=int, default=80, help="Step number to evaluate")
    parser.add_argument("--content_path_ckpt", type=str, required=True)
    parser.add_argument("--style_path_ckpt", type=str, required=True)
    parser.add_argument("--content_name", type=str, default="sketch", help="Content name")
    parser.add_argument("--style_name", type=str, default="painting", help="Style name")
    parser.add_argument("--cfg", type=float, default=6.0, help="Classifier free guidance scale")
    parser.add_argument("--top_k", type=int, default=400, help="Top k sampling")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top p sampling")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument(
        "--images_per_prompt", type=int, default=4, help="Number of images to generate per prompt"
    )
    parser.add_argument("--captions_path", type=str, default=None, help="Path to captions CSV file")
    parser.add_argument("--style_scale_end", type=int, default=3, help="Style scale end")
    parser.add_argument(
        "--prompt_config_path", type=str, default=None, help="Path to prompt configuration file"
    )
    parser.add_argument(
        "--proj_path", type=str, default=None, help="Path to projection matrix file"
    )
    parser.add_argument(
        "--proj_2_path", type=str, default=None, help="Path to projection matrix file"
    )
    parser.add_argument(
        "--disable_load_virtual_token", action="store_true", help="disable load virtual token"
    )
    parser.add_argument("--is_live_subject", action="store_true", help="is live subject")
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--save_grid",
        action="store_true",
        default=False,
        help="Save grid visualization as JPG (default: False)",
    )
    return parser.parse_args()


def create_image_grid(
    images: List[torch.Tensor], prompts: List[str], images_per_prompt: int, font_size: int = 20
) -> torch.Tensor:
    """ƒ
    Create a grid of images with captions.

    Args:
        images: List of image tensors
        prompts: List of prompt strings
        images_per_prompt: Number of images per prompt
        font_size: Font size for captions (default: 20)

    Returns:
        Grid tensor containing all images with captions
    """
    num_prompts = len(prompts)

    # First add captions to all images
    captioned_images = []
    for i, img in enumerate(images):
        prompt_idx = i // images_per_prompt
        img_idx = i % images_per_prompt + 1
        caption = f"{prompts[prompt_idx]} ({img_idx}/{images_per_prompt})"
        captioned_img = add_caption_to_image(img, caption, font_size)
        captioned_images.append(captioned_img)

    # Convert to tensor and create grid
    image_tensor = torch.stack(captioned_images)
    grid = make_grid(image_tensor, nrow=images_per_prompt, padding=10)

    return grid


def main() -> None:
    """
    Main function to orchestrate the evaluation process.

    Loads the Switti model, processes prompts, generates images, and saves
    results with comprehensive metadata for further analysis.
    """
    args = parse_args()

    print("Args:", args)
    device = args.device

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    print(f"Random seed set to {args.seed} for reproducibility")

    # Load prompt config if provided
    prompt_config = None
    if args.prompt_config_path is not None:
        print(f"Loading prompt config from {args.prompt_config_path}")
        with open(args.prompt_config_path, "r") as f:
            prompt_config = yaml.safe_load(f)

    print(f"Loading model from {args.model_path} on device {device}")

    if args.disable_load_virtual_token:
        print("Loading virtual token is disabled")
        # Configuration for prompt tuning
        prompt_config["selected_scales"] = [10]
        prompt_config["num_virtual_tokens_per_scale"] = 1
        prompt_config["add_mode"] = "selected_single_scales"
        prompt_config["deep"] = False
        prompt_config["init_zeros"] = False

    print("Prompt config:", prompt_config)

    pipe = SwittiDisentangledPipeline.from_pretrained(
        args.model_path,
        device=device,
        torch_dtype=torch.bfloat16,
        prompt_config=prompt_config,
        proj_path=args.proj_path,
        proj_2_path=args.proj_2_path,
    )

    # Construct paths
    content_path_ckpt = f"{args.content_path_ckpt}"
    style_path_ckpt = f"{args.style_path_ckpt}"

    # Load content embeddings
    placeholder_token_content_ids = pipe.load_textual_inversion(
        f"{content_path_ckpt}/learned_embeds-steps-{args.step}_content.safetensors",
        text_encoder=pipe.text_encoder.transformer,
        tokenizer=pipe.text_encoder.tokenizer,
    )
    placeholder_token_content_ids_2 = pipe.load_textual_inversion(
        f"{content_path_ckpt}/learned_embeds_2-steps-{args.step}_content.safetensors",
        text_encoder=pipe.text_encoder_2.transformer,
        tokenizer=pipe.text_encoder_2.tokenizer,
    )

    # Load style embeddings
    placeholder_token_style_ids = pipe.load_textual_inversion(
        f"{style_path_ckpt}/learned_embeds-steps-{args.step}_style.safetensors",
        text_encoder=pipe.text_encoder.transformer,
        tokenizer=pipe.text_encoder.tokenizer,
    )
    placeholder_token_style_ids_2 = pipe.load_textual_inversion(
        f"{style_path_ckpt}/learned_embeds_2-steps-{args.step}_style.safetensors",
        text_encoder=pipe.text_encoder_2.transformer,
        tokenizer=pipe.text_encoder_2.tokenizer,
    )

    pipe.kwargs_bonus["placeholder_token_content_ids"] = placeholder_token_content_ids
    pipe.kwargs_bonus["placeholder_token_content_ids_2"] = placeholder_token_content_ids_2
    pipe.kwargs_bonus["placeholder_token_style_ids"] = placeholder_token_style_ids
    pipe.kwargs_bonus["placeholder_token_style_ids_2"] = placeholder_token_style_ids_2

    virtual_token_path = f"{args.content_path_ckpt}/virtual_token_embed-steps-{args.step}.pt"
    if not args.disable_load_virtual_token and os.path.exists(virtual_token_path):
        virtual_token_state = torch.load(virtual_token_path)
        pipe.switti.virtual_token_embed.load_state_dict(virtual_token_state)
    else:
        print(f"Warning: Virtual token embeddings not found at {virtual_token_path}")

    # Define identifiers and get prompts
    identifier_content = get_identifier(
        get_number_of_vecs(
            f"{content_path_ckpt}/learned_embeds-steps-{args.step}_content.safetensors"
        ),
        is_content=True,
    )
    identifier_style = get_identifier(
        get_number_of_vecs(f"{style_path_ckpt}/learned_embeds-steps-{args.step}_style.safetensors"),
        is_content=False,
    )
    eval_prompts, content_desc, style_desc, prompt_types, mode_prompts = get_eval_prompts(
        args.prompts_path,
        identifier_content,
        identifier_style,
        args.captions_path,
        args.is_live_subject,
    )

    # Create base result directory with timestamp
    result_dir = os.path.join(
        args.output_dir,
        f'{args.content_name}+{args.style_name}_{datetime.now().strftime("%Y%m%d%H%M%S")}',
    )
    os.makedirs(result_dir, exist_ok=True)

    # Generate and save images for each prompt
    for prompt_idx, (prompt, prompt_type, mode_prompt) in enumerate(
        zip(eval_prompts, prompt_types, mode_prompts)
    ):
        # Create config for this prompt
        config = {
            "gen_prompt": prompt,
            "cfg": args.cfg,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "step": args.step,
            "style_scale_end": args.style_scale_end,
            "prompt_type": prompt_type,
        }

        # Create prompt-specific directory
        prompt_result_dir = os.path.join(
            result_dir, "output", "ours", f"prompt_{prompt_idx}_{prompt_type}"
        )
        os.makedirs(prompt_result_dir, exist_ok=True)

        # Set up prompts for content and style
        content_prompt = prompt
        style_prompt = prompt

        # Generate batch of images for this prompt
        print(
            f"Generating {args.images_per_prompt} images for prompt {prompt_idx + 1}/{len(eval_prompts)}: {prompt}"
        )

        # Create batches of prompts
        batch_prompts = [prompt] * args.images_per_prompt
        batch_content_prompts = [content_prompt] * args.images_per_prompt
        batch_style_prompts = [style_prompt] * args.images_per_prompt

        # Generate all images at once
        images = pipe(
            batch_prompts,
            batch_content_prompts,
            batch_style_prompts,
            cfg=args.cfg,
            top_k=args.top_k,
            top_p=args.top_p,
            return_pil=True,
            style_scale_end=args.style_scale_end,
            mode=mode_prompt,
        )

        # Save each generated image
        for i, image in enumerate(images):
            image.save(os.path.join(prompt_result_dir, f"{i}.png"))

        # Save config for this prompt
        config_name = f"prompt_{prompt_idx}_params.json"
        config_path = os.path.join(result_dir, config_name)
        with open(config_path, "w") as f:
            json.dump(config, f)

    # Create and save grid visualization in the result directory (optional)
    if args.save_grid:
        print("Creating grid visualization...")
        all_images = []
        for prompt_idx, prompt_type in enumerate(prompt_types):
            # Include prompt_type in the directory path
            prompt_dir = os.path.join(
                result_dir, "output", "ours", f"prompt_{prompt_idx}_{prompt_type}"
            )
            for i in range(args.images_per_prompt):
                img_path = os.path.join(prompt_dir, f"{i}.png")
                img = Image.open(img_path)
                img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
                all_images.append(img_tensor)

        grid = create_image_grid(all_images, eval_prompts, args.images_per_prompt)
        grid_image = Image.fromarray((grid.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        grid_path = os.path.join(result_dir, "grid_visualization.jpg")
        grid_image.save(grid_path, "JPEG", quality=95)
        print(f"Grid visualization saved to {grid_path}")
    else:
        print("Grid visualization skipped (use --save_grid to enable)")

    print(f"Results saved to {result_dir}")


if __name__ == "__main__":
    main()
