from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

def make_prompt(user_input):
    return f"<|user|>\n{user_input}\n<|assistant|>\n"

PEFT_MODEL_PATH = r"E:\custome_fineturing_PhoGPT_4B\result_fineturning\experiments\checkpoint-1000"

# Load config adapter để lấy base model path
config = PeftConfig.from_pretrained(PEFT_MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    trust_remote_code=True,
    device_map="auto",
    offload_folder="./offload",
    torch_dtype=torch.float16,
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token

# Load adapter PEFT fine-tuned
model = PeftModel.from_pretrained(model, PEFT_MODEL_PATH)
model.eval()

prompt = make_prompt("Điểm thi của đại học Kinh Tế Quốc Dân là bao nhiêu?")

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)

