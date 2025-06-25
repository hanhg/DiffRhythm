import argparse

import sys
import os
import torch
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../infer")))

from infer_utils import prepare_model

def save_all_models(cfm, tokenizer, muq, vae, device, save_dir="saved_models"):
    os.makedirs(save_dir, exist_ok=True)

    # Save CFM weights
    torch.save(cfm.state_dict(), os.path.join(save_dir, "cfm_model_state.pt"))

    # Save tokenizer (optional, for completeness)
    with open(os.path.join(save_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    # Save MuQ (Hugging Face format: config.json, pytorch_model.bin, etc.)
    muq.save_pretrained(os.path.join(save_dir, "muq_model"))

    # Save VAE TorchScript model
    vae.save(os.path.join(save_dir, "vae_model_script.pt"))

    print(f"✅ All models saved in '{save_dir}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--audio-length",
        type=int,
        default=95,
        choices=[95, 285],
        help="length of generated song",
    )  # length of target song
    parser.add_argument(
        "--repo-id", type=str, default="ASLP-lab/DiffRhythm-base", help="target model"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="saved_models/example",
        help="output directory fo generated song",
    )  # output directory of target song

    args = parser.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.mps.is_available():
        device = "mps"

    audio_length = args.audio_length
    if audio_length == 95:
        max_frames = 2048
    elif audio_length == 285:  # current not available
        max_frames = 6144

    cfm, tokenizer, muq, vae = prepare_model(max_frames, device, repo_id=args.repo_id)

    # Save them all
    save_all_models(cfm, tokenizer, muq, vae, device, save_dir=args.save_dir)