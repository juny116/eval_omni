import sys
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

sys.path.append('/home/juny116/Workspace/Emu3/emu3')

from PIL import Image
from mllm.processing_emu3 import Emu3Processor

tok = AutoTokenizer.from_pretrained("BAAI/Emu3-Chat-hf")
tok.chat_template = open("tokenizers/emu3.jinja").read()
# tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B")
# tok.chat_template = open("tokenizers/qwen.jinja").read()
# with open("qwen_template.jinja", "w", encoding="utf-8") as f:
#     f.write(tok.chat_template)
dummy_messages = [
    {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "I'm fine, thank you!"}]},
]
formatted = tok.apply_chat_template(dummy_messages, tokenize=False, add_generation_prompt=False)
print(formatted)
# processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")

# inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)

# You are a helpful assistant. USER: {image_prompt}{text_prompt}. ASSISTANT:
# image_processor = AutoImageProcessor.from_pretrained("BAAI/Emu3-VisionTokenizer", trust_remote_code=True)
# image_tokenizer = AutoModel.from_pretrained("BAAI/Emu3-VisionTokenizer", trust_remote_code=True).eval()
# processor = Emu3Processor(image_processor, image_tokenizer, tok)

# image = Image.open("demo.png")

# inputs = processor(
#     text="Describe the image in detail.",
#     image=image,
#     mode='U',
#     return_tensors="pt",
#     padding="longest",
# )
