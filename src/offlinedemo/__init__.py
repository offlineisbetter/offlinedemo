# offlineisbetter
# Copyright (c) 2026- offlineisbetter

from pathlib import Path
import sys
import time

from transformers import AutoTokenizer
import onnxruntime as ort

def main():
    # Get checkpoint name
    if len(sys.argv) < 2:
        print("offlinedemo [checkpoint]")
        return
    checkpoint = Path(sys.argv[1])

    # Load checkpoint
    session = ort.InferenceSession(
        checkpoint / "model.onnx",
        providers=["CPUExecutionProvider"],
    )
    print("Model loaded!")
    print()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        local_files_only = True,
    )

    # Ask user for input
    user_input = input("offlineisbetter >> ")
    
    # Tokenize and run inference
    start = time.perf_counter()
    tokens = tokenizer(user_input)
    result = session.run(None, tokens)[0]
    duration = time.perf_counter() - start
    
    # Report to user
    print(f"LOGITS:  {result}")
    print(f"LATENCY: {duration*1000:.3f} ms")
