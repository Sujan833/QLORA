"""
Inference Script for Generating Responses with Fine-Tuned Model.
"""

import argparse
import sys
from pathlib import Path

# Ensure src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qlora_engine.config.loader import load_config
from qlora_engine.inference.engine import InferenceEngine
from qlora_engine.utils.logging import get_logger

logger = get_logger("scripts.inference")


def main():
    parser = argparse.ArgumentParser(description="Run inference using fine-tuned QLoRA model.")
    parser.add_argument("--config", type=str, default="configs/cybersecurity.yaml", help="Path to config YAML file")
    parser.add_argument("--prompt", type=str, default="What is SQL injection?", help="Input prompt question")
    parser.add_argument("--adapter", type=str, default=None, help="Path to saved adapter directory")
    args = parser.parse_args()

    config = load_config(args.config)
    adapter_path = args.adapter or config.get("output", {}).get("directory", "outputs/cybersecurity_qwen3b")

    logger.info(f"Initializing Inference Engine (Model: {config['model']['name']})...")
    engine = InferenceEngine(
        base_model_name=config["model"]["name"],
        adapter_path=adapter_path if Path(adapter_path).exists() else None,
    )

    messages = [{"role": "user", "content": args.prompt}]
    logger.info(f"Prompt: {args.prompt}")

    response = engine.generate(
        messages,
        temperature=float(config.get("inference", {}).get("temperature", 0.7)),
        top_p=float(config.get("inference", {}).get("top_p", 0.9)),
        max_new_tokens=int(config.get("inference", {}).get("max_new_tokens", 256)),
    )

    logger.info("=" * 60)
    logger.info("GENERATED RESPONSE:")
    logger.info("=" * 60)
    print("\n" + response + "\n")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
