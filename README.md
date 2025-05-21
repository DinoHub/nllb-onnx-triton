# NLLB-200 ONNX Model with Triton Inference Server

This repository contains the necessary code to:
1. Convert Facebook's NLLB-200-distilled-600M transformer model to ONNX format
2. Host the ONNX model on NVIDIA's Triton Inference Server
3. Provide sample scripts to interface with the deployed model

## Overview

The No Language Left Behind (NLLB) model is a multilingual machine translation model that supports 200 languages. The distilled 600M version is a smaller, optimized version that maintains good translation quality while requiring fewer computational resources.

Converting this model to ONNX format and serving it via Triton Inference Server offers several advantages:
- Improved inference performance
- Platform/framework independence
- Scalable serving infrastructure

## Repository Structure

```
.
├── README.md
├── convert_to_onnx/
│   ├── convert_nllb_to_onnx.py
│   └── requirements.txt
├── triton_model_repository/
│   └── nllb_onnx/
│       ├── config.pbtxt
│       └── 1/
│           └── model.onnx
└── client/
    ├── requirements.txt
    └── nllb_client.py
```

## Clone the Repository

```bash
git clone https://github.com/DinoHub/nllb-onnx-triton.git
cd nllb-onnx-triton
```

## 1. Converting NLLB to ONNX

The conversion script transforms the PyTorch NLLB model into ONNX format with proper optimization.

### Prerequisites

```bash
cd convert_to_onnx
pip install -r requirements.txt
```

### Running the Conversion

```bash
python convert_nllb_to_onnx.py --model_id facebook/nllb-200-3.3B --output_dir ../triton_model_repository/nllb_onnx/1/
```

The script will:
1. Download the model from Hugging Face
2. Convert the encoder and decoder components to ONNX
3. Apply optimizations for inference
4. Save the model in the Triton model repository structure

## 2. Hosting on Triton Inference Server

### Prerequisites

- Docker
- NVIDIA GPU with appropriate drivers (for GPU acceleration)

### Starting the Triton Server

```bash
cd ..
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 -p 8002:8002 \
    -v $(pwd)/triton_model_repository:/models \
    nvcr.io/nvidia/tritonserver:25.04-py3 tritonserver \
    --model-repository=/models
```

## 3. Using the Client

The client script demonstrates how to send translation requests to the deployed model.

### Prerequisites

```bash
cd client
pip install -r requirements.txt
```

### Sample Usage

```bash
python nllb_client.py \
    --text "Hello, how are you?" \
    --source_lang "eng_Latn" \
    --target_lang "fra_Latn"
```


## Supported Languages

NLLB-200 supports 200 languages. Some common language codes include:
- `eng_Latn`: English (Latin script)
- `fra_Latn`: French
- `deu_Latn`: German
- `spa_Latn`: Spanish
- `zho_Hans`: Chinese (Simplified)
- `rus_Cyrl`: Russian
- `ara_Arab`: Arabic

For a complete list of supported languages, refer to the [NLLB documentation](https://github.com/facebookresearch/fairseq/tree/nllb).

## References

- [NLLB-200 Model](https://huggingface.co/facebook/nllb-200-distilled-600M)
- [NVIDIA Triton Inference Server](https://github.com/triton-inference-server/server)
- [ONNX Runtime](https://onnxruntime.ai/)

## License

The code in this repository is licensed under the MIT License. The NLLB model itself is licensed under the CC-BY-NC 4.0 license.
