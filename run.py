import subprocess
import os
# Absolute path to the folder where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
# Change working directory to a specific path
os.chdir(script_dir)

subprocess.run([
    "python", "custom_infer/load_run.py",
    "--lrc-path", "infer/example/Custom_Lyrics/badIdeaRight.lrc",
    "--ref-audio-path", "infer/example/Custom_Wav/badIdeaRight.mp3",
    "--audio-length", "95",
    "--model-dir", "saved_models/model95-1_2",
    "--output-dir", "/content/Bad_Idea",
    "--chunked"
])
