import subprocess
import os

# Build the path to infer.py relative to this script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
infer_script = os.path.abspath(os.path.join(script_dir, "..", "infer", "infer.py"))

# Build the command
cmd = [
    "python",
    infer_script,
    "--lrc-path", "infer/example/eg_en.lrc",
    "--ref-prompt", "classical genres, hopeful mood, piano.",
    "--audio-length", "95",
    "--repo-id", "ASLP-lab/DiffRhythm-1_2",
    "--output-dir", "infer/example/output_en",
    "--chunked"
]

# Run the command and print output live
subprocess.run(cmd)
input("\nPress Enter to continue...")  # simulates 'pause'