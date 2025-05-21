import argparse
import numpy as np
import tritonclient.http as httpclient
from transformers import AutoTokenizer
import time

class NLLBTritonClient:
    """Client for interacting with NLLB model deployed on Triton Inference Server"""
    
    def __init__(self, url="localhost:8000", model_name="nllb_onnx", model_version="1"):
        """
        Initialize the client
        
        Args:
            url: Triton server URL
            model_name: Name of the model in Triton
            model_version: Version of the model to use
        """
        self.url = url
        self.model_name = model_name
        self.model_version = model_version
        
        # Connect to Triton server
        self.client = httpclient.InferenceServerClient(url=url, verbose=False)
        
        # Load NLLB tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
        
    def translate(self, text, source_lang, target_lang, max_length=100):
        """
        Translate text from source language to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code (e.g., 'eng_Latn')
            target_lang: Target language code (e.g., 'fra_Latn')
            max_length: Maximum length of generated translation
            
        Returns:
            Translated text
        """
        if source_lang not in self.get_supported_languages():
            raise ValueError(f"Invalid source language: {source_lang}")
        if target_lang not in self.get_supported_languages():
            raise ValueError(f"Invalid target language: {target_lang}")
        
        # Set target language for generation
        self.tokenizer.src_lang = source_lang
        self.tokenizer.tgt_lang = target_lang

        # Format input with language tokens
        tokenized_input = self.tokenizer(text, return_tensors="np")
        input_ids = tokenized_input["input_ids"].astype(np.int64)
        attention_mask = tokenized_input["attention_mask"].astype(np.int64)
        decoder_input_ids = np.array([[self.tokenizer.convert_tokens_to_ids("</s>"), self.tokenizer.convert_tokens_to_ids(target_lang)]], dtype=np.int64)

        # Set up the inputs for Triton
        inputs = [
            httpclient.InferInput("input_ids", input_ids.shape, "INT64"),
            httpclient.InferInput("attention_mask", attention_mask.shape, "INT64"),
            httpclient.InferInput("decoder_input_ids", decoder_input_ids.shape, "INT64")
        ]
        
        inputs[0].set_data_from_numpy(input_ids)
        inputs[1].set_data_from_numpy(attention_mask)

        start_time = time.time()
        
        # Auto-regressive generation loop
        for _ in range(max_length):
            inputs[2].set_shape(decoder_input_ids.shape)
            inputs[2].set_data_from_numpy(decoder_input_ids)
            
            # Set up the outputs
            outputs = [httpclient.InferRequestedOutput("logits")]
            
            # Send request to Triton
            response = self.client.infer(
                model_name=self.model_name,
                model_version=self.model_version,
                inputs=inputs,
                outputs=outputs
            )

            # Get the response
            logits = response.as_numpy("logits")
            token_ids = np.argmax(logits, axis=-1)

            # If end of sequence token is generated, stop
            if token_ids[0, -1] == self.tokenizer.eos_token_id:
                print("EOS token encountered, stopping generation")
                break

            # Append the predicted token to decoder_input_ids
            decoder_input_ids = np.concatenate([decoder_input_ids, token_ids[:, -1:]], axis=1)

        inference_time = time.time() - start_time
        
        # Decode the translated text
        translated_text = self.tokenizer.decode(token_ids[0], skip_special_tokens=True)
        
        return {
            "translated_text": translated_text,
            "inference_time_seconds": inference_time
        }
    
    def get_supported_languages(self):
        """Return list of supported language codes"""
        return list(self.tokenizer.additional_special_tokens)

def main():
    parser = argparse.ArgumentParser(description="NLLB Translation Client for Triton")
    parser.add_argument("--text", required=True, help="Text to translate")
    parser.add_argument("--source_lang", required=True, help="Source language code")
    parser.add_argument("--target_lang", required=True, help="Target language code")
    parser.add_argument("--url", default="localhost:8000", help="Triton server URL")
    parser.add_argument("--model_name", default="nllb_onnx", help="Model name in Triton")
    parser.add_argument("--model_version", default="1", help="Model version")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum output length")
    
    args = parser.parse_args()
    
    # Create client and translate
    client = NLLBTritonClient(
        url=args.url,
        model_name=args.model_name,
        model_version=args.model_version
    )
    
    result = client.translate(
        text=args.text,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_length=args.max_length
    )
    
    print(f"\nTranslation from {args.source_lang} to {args.target_lang}:")
    print(f"Original: {args.text}")
    print(f"Translated: {result['translated_text']}")
    print(f"Inference time: {result['inference_time_seconds']:.3f} seconds")
    

if __name__ == "__main__":
    main()