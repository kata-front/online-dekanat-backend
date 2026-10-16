import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'

print('Загружаем токенизатор...')

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir='models/llm',
)

print('Загружаем модель...')

model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir='models/llm',
        dtype=torch.float32,
    )
    .to('cpu')
    .eval()
)

print('Модель готова.')
print('Устройство:', next(model.parameters()).device)
