import torch
from transformers import pipeline
from modelscope import snapshot_download

import os
os.environ['HF_HOME'] = '/blabla/cache/'
os.environ['MODELSCOPE_CACHE'] = '/tmp'
model_dir = snapshot_download('LLM-Research/Llama-3.2-1B-Instruct')

pipe = pipeline(
    "text-generation",
    model=model_dir,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    cache_dir="/tmp"
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
]

try:
    while True:
        user_input = input("\nEnter your message (q to quit): ")
        if user_input.lower() == 'q':
            break
            
        messages.append({"role": "user", "content": user_input})
        outputs = pipe(messages, max_new_tokens=256)
        response = outputs[0]["generated_text"][-1]
        
        print("\nBot:", response)
        messages.append({"role": "assistant", "content": response})
        
except KeyboardInterrupt:
    print("\nGoodbye!")
