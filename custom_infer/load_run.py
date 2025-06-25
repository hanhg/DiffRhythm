import torch
import json
import pickle
import sys
import os

import argparse
import time
import random

import torch
import torchaudio
from einops import rearrange

sys.path.append(os.getcwd())
from model import DiT, CFM
from muq import MuQMuLan  # or correct import path for MuQ

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../infer")))
from infer_utils import (
    load_checkpoint, 
    CNENTokenizer,
    decode_audio,
    get_lrc_token,
    get_negative_style_prompt,
    get_reference_latent,
    get_style_prompt,
)
from infer import inference

print("Current working directory:", os.getcwd())

def load_models(max_frames, device, save_dir="saved_models"):
    # === 1. Load CFM ===
    st = time.time()
    config_path = "./config/diffrhythm-1b.json"  # Same config used during saving
    with open(config_path) as f:
        model_config = json.load(f)

    
    cfm = CFM(
        transformer=DiT(**model_config["model"], max_frames=max_frames),
        num_channels=model_config["model"]["mel_dim"],
        max_frames=max_frames
    ).to(device)
    
    cfm_weights_path = os.path.join(save_dir, "cfm_model_state.pt")
    cfm.load_state_dict(torch.load(cfm_weights_path, map_location=device))
    cfm.eval()
    cfm = cfm.to(device).float()

    et = time.time()
    print(f"Loading CFM Cost: {et-st}s")

    # === 2. Load Tokenizer ===
    st = time.time()
    with open(os.path.join(save_dir, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)
    et = time.time()
    print(f"Loading Tokenizer Cost: {et-st}s")
    # === 3. Load MuQ ===
    st = time.time()
    muq = MuQMuLan.from_pretrained(os.path.join(save_dir, "muq_model"))
    muq = muq.to(device).eval()
    et = time.time()
    print(f"Loading MuQ Cost: {et-st}s")

    # === 4. Load VAE ===
    st = time.time()
    vae = torch.jit.load(os.path.join(save_dir, "vae_model_script.pt"))
    vae = vae.to(device).eval()
    et = time.time()
    print(f"Loading VAE Cost: {et-st}s")

    print("✅ All models loaded successfully.")
    return cfm, tokenizer, muq, vae


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lrc-path",
        type=str,
        help="lyrics of target song",
    )  # lyrics of target song
    parser.add_argument(
        "--ref-prompt",
        type=str,
        help="reference prompt as style prompt for target song",
        required=False,
    )  # reference prompt as style prompt for target song
    parser.add_argument(
        "--ref-audio-path",
        type=str,
        help="reference audio as style prompt for target song",
        required=False,
    )  # reference audio as style prompt for target song
    parser.add_argument(
        "--chunked",
        action="store_true",
        help="whether to use chunked decoding",
    )  # whether to use chunked decoding
    parser.add_argument(
        "--audio-length",
        type=int,
        default=95,
        choices=[95, 285],
        help="length of generated song",
    )  # length of target song
    parser.add_argument(
        "--model-dir",
        type=str,
        default="saved_models/example",
        help="output directory fo generated song",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="custom_infer/example/output",
        help="output directory fo generated song",
    )  # output directory of target song
    parser.add_argument(
        "--edit",
        action="store_true",
        help="whether to open edit mode",
    )  # edit flag
    parser.add_argument(
        "--ref-song",
        type=str,
        required=False,
        help="reference prompt as latent prompt for editing",
    )  # reference prompt as latent prompt for editing
    parser.add_argument(
        "--edit-segments",
        type=str,
        required=False,
        help="Time segments to edit (in seconds). Format: `[[start1,end1],...]`. "
             "Use `-1` for audio start/end (e.g., `[[-1,25], [50.0,-1]]`)."
    )  # edit segments of target song
    parser.add_argument(
        "--batch-infer-num",
        type=int,
        default=1,
        required=False,
        help="number of songs per batch",
    )  # number of songs per batch
    args = parser.parse_args()

    assert (
        args.ref_prompt or args.ref_audio_path
    ), "either ref_prompt or ref_audio_path should be provided"
    assert not (
        args.ref_prompt and args.ref_audio_path
    ), "only one of them should be provided"
    if args.edit:
        assert (
            args.ref_song and args.edit_segments
        ), "reference song and edit segments should be provided for editing"

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

    cfm, tokenizer, muq, vae = load_models(max_frames, device, save_dir=args.model_dir)

    if args.lrc_path:
        with open(args.lrc_path, "r", encoding='utf-8') as f:
            lrc = f.read()
    else:
        lrc = ""
    lrc_prompt, start_time = get_lrc_token(max_frames, lrc, tokenizer, device)
    start_time = start_time.to(device).float()

    if args.ref_audio_path:
        style_prompt = get_style_prompt(muq, args.ref_audio_path)
    else:
        style_prompt = get_style_prompt(muq, prompt=args.ref_prompt)

    negative_style_prompt = get_negative_style_prompt(device)

    latent_prompt, pred_frames = get_reference_latent(device, max_frames, args.edit, args.edit_segments, args.ref_song, vae)
    latent_prompt = latent_prompt.to(device).float()

    s_t = time.time()
    generated_songs = inference(
        cfm_model=cfm,
        vae_model=vae,
        cond=latent_prompt,
        text=lrc_prompt,
        duration=max_frames,
        style_prompt=style_prompt,
        negative_style_prompt=negative_style_prompt,
        start_time=start_time,
        pred_frames=pred_frames,
        chunked=args.chunked,
        batch_infer_num=args.batch_infer_num
    )
    e_t = time.time() - s_t
    print(f"inference cost {e_t:.2f} seconds")
    
    generated_song = random.sample(generated_songs, 1)[0]

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "output.wav")
    torchaudio.save(output_path, generated_song, sample_rate=44100)

    
