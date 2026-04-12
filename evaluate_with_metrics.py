"""
Evaluation script for calculating metrics on generated images.

This script processes result directories containing generated images and calculates
various similarity metrics including CLIP, DINO, and CSD (Content-Style Decomposition)
similarities between source and generated images.
"""

import argparse
import json
import os
from typing import Dict

import pandas as pd
import torch

from evaluator.personalized import PersonalizedBase
from vis.metrics import CLIPEvaluator, CSDEvaluator, DINOEvaluator


def process_single_result(result_dir: str, src_dir: str, device: torch.device) -> pd.DataFrame:
    """
    Process a single result directory and calculate metrics.

    Args:
        result_dir: Path to the result directory containing generated images
        src_dir: Path to the source images directory
        device: PyTorch device for computation (GPU/CPU)

    Returns:
        DataFrame containing calculated metrics for all prompts and images

    Raises:
        FileNotFoundError: If required directories or files don't exist
        ValueError: If prompt parameters are invalid
    """
    # Validate input paths
    if not os.path.exists(result_dir):
        raise FileNotFoundError(f"Result directory not found: {result_dir}")
    if not os.path.exists(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")

    # Initialize evaluators
    clip_evaluator = CLIPEvaluator(device)
    dino_evaluator = DINOEvaluator(device)
    csd_evaluator = CSDEvaluator(device)

    # Load source images
    src_data_loader = PersonalizedBase(src_dir, size=256, flip_p=0.0, set="eval")
    src_images = [
        torch.from_numpy(src_data_loader[i]["image"]).permute(2, 0, 1)
        for i in range(src_data_loader.num_images)
    ]
    src_images = torch.stack(src_images, axis=0).to(device)

    # Get all prompt directories
    output_dir = os.path.join(result_dir, "output", "ours")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    prompt_folders = [
        folder
        for folder in os.listdir(output_dir)
        if os.path.isdir(os.path.join(output_dir, folder)) and folder.startswith("prompt_")
    ]

    if not prompt_folders:
        raise ValueError(f"No prompt folders found in {output_dir}")

    metrics_data = []

    for prompt_folder in sorted(prompt_folders):
        # Extract prompt index and type from folder name (e.g., "prompt_0_content")
        folder_parts = prompt_folder.split("_")
        if len(folder_parts) < 3:
            print(f"Warning: Skipping malformed folder name: {prompt_folder}")
            continue

        prompt_idx = int(folder_parts[1])
        prompt_type = "_".join(
            folder_parts[2:]
        )  # Join remaining parts in case type contains underscores

        # Get prompt from params file
        params_file = os.path.join(result_dir, f"prompt_{prompt_idx}_params.json")
        if not os.path.exists(params_file):
            print(f"Warning: Params file not found for prompt {prompt_idx}, skipping")
            continue

        try:
            with open(params_file, "r") as f:
                params = json.load(f)
            prompt = params["gen_prompt"]
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Error reading params for prompt {prompt_idx}: {e}")
            continue

        # Load target images
        prompt_dir = os.path.join(output_dir, prompt_folder)
        trg_data_loader = PersonalizedBase(prompt_dir, size=256, flip_p=0.0)

        trg_images = [
            torch.from_numpy(trg_data_loader[i]["image"]).permute(2, 0, 1)
            for i in range(trg_data_loader.num_images)
        ]
        trg_images = torch.stack(trg_images, axis=0).to(device)

        # Calculate similarities
        sim_img_dino = dino_evaluator.img_to_img_similarity(src_images, trg_images)
        sim_img_clip = clip_evaluator.img_to_img_similarity(src_images, trg_images)
        sim_text = clip_evaluator.txt_to_img_similarity(prompt, trg_images)

        # Calculate CSD similarities
        csd_content_sim, csd_style_sim = csd_evaluator.calculate_similarities(
            src_images[0], trg_images[0]
        )

        # Convert to float
        sim_img_clip = float(sim_img_clip.cpu().numpy())
        sim_img_dino = float(sim_img_dino.cpu().numpy())
        sim_text = float(sim_text.cpu().numpy())

        # Store metrics for each image
        for img_idx in range(len(trg_images)):
            metrics_data.append(
                {
                    "prompt_idx": prompt_idx,
                    "prompt_type": prompt_type,
                    "image_idx": img_idx,
                    "prompt": prompt,
                    "img_sim_dino": sim_img_dino,
                    "img_sim_clip": sim_img_clip,
                    "txt_sim": sim_text,
                    "csd_content": csd_content_sim,
                    "csd_style": csd_style_sim,
                }
            )

        # Calculate average for this prompt
        avg_metrics = {
            "prompt_idx": prompt_idx,
            "prompt_type": prompt_type,
            "image_idx": "average",
            "prompt": prompt,
            "img_sim_dino": sim_img_dino,
            "img_sim_clip": sim_img_clip,
            "txt_sim": sim_text,
            "csd_content": csd_content_sim,
            "csd_style": csd_style_sim,
        }
        metrics_data.append(avg_metrics)

    # Convert to DataFrame
    df = pd.DataFrame(metrics_data)

    if df.empty:
        raise ValueError("No valid metrics data generated")

    # Calculate overall averages
    overall_avg = df[df["image_idx"] != "average"].mean(numeric_only=True)
    overall_metrics = {
        "prompt_idx": "overall",
        "prompt_type": "overall",
        "image_idx": "average",
        "prompt": "overall_average",
        "img_sim_dino": overall_avg["img_sim_dino"],
        "img_sim_clip": overall_avg["img_sim_clip"],
        "txt_sim": overall_avg["txt_sim"],
        "csd_content": overall_avg["csd_content"],
        "csd_style": overall_avg["csd_style"],
    }
    df = pd.concat([df, pd.DataFrame([overall_metrics])], ignore_index=True)

    return df


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the evaluation script.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Calculate metrics for generated images using CLIP, DINO, and CSD evaluators"
    )
    parser.add_argument(
        "--result_dir", type=str, required=True, help="Base directory containing result folders"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing source images (default: data)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)"
    )

    return parser.parse_args()


def main() -> None:
    """
    Main function to orchestrate the evaluation process.

    Processes all result directories and calculates comprehensive metrics
    for generated images, saving results in both CSV and summary formats.
    """
    args = parse_args()

    # Validate input directory
    if not os.path.exists(args.result_dir):
        raise FileNotFoundError(f"Result directory not found: {args.result_dir}")

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize dictionaries to store sums and counts for final averaging
    final_sums: Dict[str, Dict[str, float]] = {
        "content": {
            "img_sim_dino": 0.0,
            "img_sim_clip": 0.0,
            "txt_sim": 0.0,
            "csd_content": 0.0,
            "csd_style": 0.0,
            "count": 0.0,
        },
        "style": {
            "img_sim_dino": 0.0,
            "img_sim_clip": 0.0,
            "txt_sim": 0.0,
            "csd_content": 0.0,
            "csd_style": 0.0,
            "count": 0.0,
        },
        "content_style": {
            "img_sim_dino": 0.0,
            "img_sim_clip": 0.0,
            "txt_sim": 0.0,
            "csd_content": 0.0,
            "csd_style": 0.0,
            "count": 0.0,
        },
    }

    # Process each result directory
    result_folders = [
        f for f in os.listdir(args.result_dir) if os.path.isdir(os.path.join(args.result_dir, f))
    ]

    if not result_folders:
        print(f"No result folders found in {args.result_dir}")
        return

    for result_folder in result_folders:
        result_path = os.path.join(args.result_dir, result_folder)

        print(f"Processing {result_folder}...")

        try:
            # Get concept name from folder name (remove timestamp)
            concept = result_folder.split("_")
            concept = "_".join(concept[:-1])

            # Set source directory for this concept
            src_dir = os.path.join(args.data_dir, concept, "imgs")

            # Calculate metrics
            df = process_single_result(result_path, src_dir, device)

            # Create metrics directory
            metrics_dir = os.path.join(result_path, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)

            # Save metrics as CSV
            csv_path = os.path.join(metrics_dir, f"{concept}_metrics.csv")
            df.to_csv(csv_path, index=False)

            # Also save in original format
            txt_path = os.path.join(metrics_dir, f"avg_{concept}_metrics.csv")
            with open(txt_path, "w") as f:
                f.write("type,method,img_sim_dino,img_sim_clip,txt_sim,csd_content,csd_style\n")

                # Write individual prompt metrics
                for _, row in df[df["image_idx"] == "average"].iterrows():
                    if row["prompt_idx"] != "overall":
                        # Use full prompt type name (e.g. prompt_0_content_recontextualization)
                        full_type = f"prompt_{row['prompt_idx']}_{row['prompt_type']}"
                        f.write(
                            f"{full_type},our,{row['img_sim_dino']:.4f},{row['img_sim_clip']:.4f},"
                            f"{row['txt_sim']:.4f},{row['csd_content']:.4f},{row['csd_style']:.4f}\n"
                        )

                # Calculate and write averages for each prompt type
                prompt_types = ["content", "style", "content_style"]
                for prompt_type in prompt_types:
                    if prompt_type == "content":
                        # For content type, exclude content_style
                        type_df = df[
                            (df["prompt_type"].str.startswith("content"))
                            & (~df["prompt_type"].str.startswith("content_style"))
                            & (df["image_idx"] != "average")
                            & (df["prompt_idx"] != "overall")
                        ]
                    else:
                        type_df = df[
                            (df["prompt_type"].str.startswith(prompt_type))
                            & (df["image_idx"] != "average")
                            & (df["prompt_idx"] != "overall")
                        ]

                    if not type_df.empty:
                        num_prompts = len(type_df)
                        print(
                            f"Calculating average for {prompt_type} type using {num_prompts} prompts"
                        )
                        type_avg = type_df.mean(numeric_only=True)
                        f.write(
                            f"average_{prompt_type},our,{type_avg['img_sim_dino']:.4f},"
                            f"{type_avg['img_sim_clip']:.4f},{type_avg['txt_sim']:.4f},"
                            f"{type_avg['csd_content']:.4f},{type_avg['csd_style']:.4f}\n"
                        )

            print(f"Saved metrics to {csv_path} and {txt_path}")

            # Add the averages to final sums
            avg_file = os.path.join(metrics_dir, f"avg_{concept}_metrics.csv")
            df_avg = pd.read_csv(avg_file)

            for _, row in df_avg.iterrows():
                if row["type"].startswith("average_"):
                    prompt_type = row["type"].replace("average_", "")
                    if prompt_type in final_sums:
                        final_sums[prompt_type]["img_sim_dino"] += row["img_sim_dino"]
                        final_sums[prompt_type]["img_sim_clip"] += row["img_sim_clip"]
                        final_sums[prompt_type]["txt_sim"] += row["txt_sim"]
                        final_sums[prompt_type]["csd_content"] += row["csd_content"]
                        final_sums[prompt_type]["csd_style"] += row["csd_style"]
                        final_sums[prompt_type]["count"] += 1

        except Exception as e:
            print(f"Error processing {result_folder}: {e}")
            continue

    # Calculate and save final averages
    summary_path = os.path.join(args.result_dir, "summary_metrics.csv")
    with open(summary_path, "w") as f:
        f.write("method,csd-c,clip-i,csd-s,dino,clip-t\n")

        # Calculate averages for each metric type
        if final_sums["content"]["count"] > 0 and final_sums["style"]["count"] > 0:
            # CSD-C: csd_content from average_content
            csd_c = final_sums["content"]["csd_content"] / final_sums["content"]["count"]

            # CLIP-I: img_sim_clip from average_content
            clip_i = final_sums["content"]["img_sim_clip"] / final_sums["content"]["count"]

            # CSD-S: csd_style from average_style
            csd_s = final_sums["style"]["csd_style"] / final_sums["style"]["count"]

            # DINO: img_sim_dino from average_style
            dino = final_sums["style"]["img_sim_dino"] / final_sums["style"]["count"]

            # CLIP-T: average txt_sim from both average_content and average_style
            content_txt_sim = final_sums["content"]["txt_sim"] / final_sums["content"]["count"]
            style_txt_sim = final_sums["style"]["txt_sim"] / final_sums["style"]["count"]
            clip_t = (content_txt_sim + style_txt_sim) / 2

            f.write(f"our,{csd_c:.4f},{clip_i:.4f},{csd_s:.4f},{dino:.4f},{clip_t:.4f}\n")

    print(f"Saved summary metrics to {summary_path}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
