# offlineisbetter
# Copyright (c) 2026- offlineisbetter

from pathlib import Path
import json
import sys
import time

import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort

def main():
    # Get checkpoint name
    if len(sys.argv) < 2:
        print("offlinedemo [checkpoint-dir]")
        return
    checkpoint = Path(sys.argv[1])

    # Load checkpoint
    session = ort.InferenceSession(
        checkpoint / "model.onnx",
        providers=["CPUExecutionProvider"],
    )
    print("Model loaded!")

    # Load classes
    with open(checkpoint / "offlineisbetter.json", "r") as f:
        classes = json.load(f)
    classes = {v: k for k, v in classes.items()}

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only = True,
    )
    print("Tokenizer loaded!")
    print()

    # Ask user for input
    user_input = input("offlineisbetter >> ")
    
    # Tokenize and run inference
    start = time.perf_counter()
    tokens = tokenizer(user_input)
    result = session.run(None, tokens)[0][0]
    idx = int(np.argmax(result))
    c = classes[idx]
    duration = time.perf_counter() - start
    
    # Report to user
    print(f"LOGITS:  {result}")
    print(f"CLASS:   {c}")
    print(f"LATENCY: {duration*1000:.3f} ms")
