import os
import argparse
from optimum.exporters.onnx import main_export
from transformers import AutoConfig, AutoTokenizer

def convert_nllb_to_onnx(model_id="facebook/nllb-200-distilled-600M", output_dir="onnx_model"):
    """
    Convert NLLB-200 PyTorch model to ONNX format
    
    Args:
        model_id: Hugging Face model ID
        output_dir: Directory to save the ONNX model
    """
    print(f"Converting {model_id} to ONNX format...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load the tokenizer and config
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = AutoConfig.from_pretrained(model_id)

    # Export as a single model (not encoder/decoder separate)
    main_export(
        model_name_or_path=model_id,
        dtype="fp16",
        device = "cuda",
        tokenizer=tokenizer,
        config=config,
        opset=14,
        output=output_dir,
        task="translation",
        monolith=True  # Export as a single model instead of separate encoder/decoder
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NLLB-200 model to ONNX format")
    parser.add_argument(
        "--model_id", 
        default="facebook/nllb-200-distilled-600M", 
        help="Hugging Face model ID"
    )
    parser.add_argument(
        "--output_dir", 
        default="onnx_model", 
        help="Directory to save the ONNX model"
    )
    
    args = parser.parse_args()
    convert_nllb_to_onnx(args.model_id, args.output_dir)