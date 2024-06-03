"""
MIT License

Author: Chandra Sakthivel
Date: 2024-06-02

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

class ChatModel:
    def __init__(self, model_id: str, hf_token: str = None, device="cpu", quantization="fp32"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
        
        # Prepare model configuration based on quantization and device
        if device == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",  # Automatically distribute the model on available GPUs
                torch_dtype=torch.float16 if quantization == "fp16" else torch.float32,
                use_auth_token=hf_token,
                trust_remote_code=True
            ).to(self.device)
        else:
            # Handling CPU quantization options
            if quantization == "4bit":
                # Placeholder for 4-bit quantization setup
                print("4-bit quantization is selected, which is not supported natively.")
                # Simulate loading a quantized model (this is a placeholder)
                self.model = self.load_quantized_model(model_id, quantization, hf_token)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float32,
                    use_auth_token=hf_token,
                    trust_remote_code=True
                )

        self.model.eval()

    def load_quantized_model(self, model_id, quantization, hf_token):
        # Placeholder function to simulate the loading of a quantized model
        # This should be replaced with actual quantization code
        print(f"Simulating the loading of a {quantization} quantized model.")
        # Return a standard model as a stand-in
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            use_auth_token=hf_token,
            trust_remote_code=True
        )

    def inference(self, question: str, context: str = None, max_new_tokens: int = 512, concise: bool = False):
        prompt = self.build_prompt(question, context, concise)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(input_ids=inputs.input_ids, max_new_tokens=max_new_tokens, do_sample=False)
        return self.post_process(self.tokenizer.decode(outputs[0], skip_special_tokens=True))

    def build_prompt(self, question, context, concise):
        if context:
            return f"Using the information contained from the context, give a {'concise' if concise else 'detailed'} answer to the question. Context: {context}. Question: {question}"
        return f"Give a detailed answer to the question. Question: {question}"

    def post_process(self, text: str) -> str:
        corrections = {" .": ".", " ,": ",", " ’": "’", " ?": "?", " !": "!", " :": ":", " ;": ";"}
        for old, new in corrections.items():
            text = text.replace(old, new)
        return ' '.join(text.split())
