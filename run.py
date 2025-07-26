import subprocess
import os
# Absolute path to the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change working directory to a specific path
os.chdir(script_dir)

subprocess.run([
    "python", "custom_infer/load_run.py",
    "--lrc-path", "custom_infer/Custom_Lyrics/Still_in_the_Room.lrc",
    "--ref-prompt", (
        "up‑tempo pop‑punk/new‑wave driving beat, C minor, Cm–Ab–Eb–Bb, "
        "steady chord change every measure, builds from mid‑level verses to "
        "loud, energetic choruses with brief pullbacks in bridge, 149 bpm"
    ),
    "--audio-length", "95",
    "--model-dir", "saved_models/model95-1_2",
    "--output-dir", "Outputs/Output1",
    "--chunked"
])
