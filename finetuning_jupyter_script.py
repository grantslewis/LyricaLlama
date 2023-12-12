# %%
from huggingface_hub import notebook_login, login

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
# os.environ['TENSORBOARD_BINARY'] = '/home/jupyter-grantsl/.conda/envs/llm-gpu-quant/bin/tensorboard' 

login(token='YOUR_TOKEN_HERE', add_to_git_credential=True)

# %%
import torch
from datasets import Dataset, DatasetDict, load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
    Trainer,
    TrainerCallback
)
from peft import LoraConfig
from trl import SFTTrainer
import pandas as pd
from sklearn.model_selection import train_test_split
import json
from tensorboard import program
import shutil
from peft import AutoPeftModelForCausalLM



# %%
if torch.cuda.is_available():
    print("Post-environment: Visible CUDA Devices:")
    for i in range(torch.cuda.device_count()):
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
else:
    print("No CUDA devices are available.")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(DEVICE)

device = torch.device(DEVICE)

# %%
MODEL_NAME = "meta-llama/Llama-2-7b-hf"
SOURCE_DATA = "./lyrics-data-combined.tsv"
NEW_MODEL_NAME = "LyricaLlama"

OUTPUT_DIR = "experiments"

DEFAULT_SYSTEM_PROMPT = """
You are a creative, world-famous expert lyricist. Write lyrics for a song, given just a title, artist name, possible genres, and any additional information provided.
""".strip()

# %%

def dataset_builder(data): #, system_prompt=DEFAULT_SYSTEM_PROMPT):  
    def generate_input(row):
        song_name = row['SName']
        artist = row['Artist']
        genres = row['Genres']
        genre_info = ""
        if genres != "" and genres is not None:
            genre_info = ', '.join(genres)
            genre_info = f" using the following genres: {genre_info}"
        
        return f"""Write lyrics for a song titled "{song_name}" to be performed by {artist}{genre_info}."""    
    
    def generate_text(row, system_prompt=DEFAULT_SYSTEM_PROMPT):
        inputs = row['input']
        lyrics = row['lyrics']
        return f"""### Instruction: {system_prompt}

### Input:
{inputs}

### Response:
{lyrics}
""".strip()   
    
    output_data = data.copy()
    output_data['input'] = output_data.apply(generate_input, axis=1)
    output_data['lyrics'] = output_data['Lyric']
    # output_data['text'] = output_data.apply(generate_training_prompt, axis=1)
    
    output_data['text'] = output_data.apply(generate_text, axis=1)
    
    train_df, test_df = train_test_split(output_data, test_size=0.2, random_state=42)
    
    
    train_df = train_df[['input', 'lyrics', 'text']]
        

    # Further split the training data into training and validation set (90% train, 10% validation of the original train set)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    # output_data = Dataset.from_pandas(output_data)
    # return output_data
    return dataset_dict


instruction_df = pd.read_csv(SOURCE_DATA, sep='\t')
# instruction_df = instruction_df.iloc[:200]

non_nan_cols = ['SName', 'Lyric', 'Artist']

selection_df = instruction_df.copy()
print(selection_df.shape)
for col in non_nan_cols:
    selection_df = selection_df[~selection_df[col].isna()]


# Reducing the size of the dataset so that training wouldn't take too long
test_size = 0.75 #0.95 #0.5 # 0.25

selection_df, removed_df = train_test_split(selection_df, test_size=test_size, random_state=42)

# select all rows with "Taylor Swift" in the Artist column in remove_df
ts_df = removed_df[removed_df['Artist'].str.contains("Taylor Swift")]
print(ts_df.shape)

# selection_df = selection_df.append(ts_df)
selection_df = pd.concat([selection_df, ts_df])


selection_df = selection_df.fillna("")

print(selection_df.shape)


print(instruction_df.columns)
# print(min([len(val) for val in instruction_df['text'].tolist()]))




for col in selection_df.columns:
    print(col)
    lens = [len(val) for val in selection_df[col].tolist()]
    print(col, min(lens), max(lens))

dataset = dataset_builder(selection_df)
print(dataset)

# %%
print(instruction_df.head())

# %%
def create_model_and_tokenizer():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        use_safetensors=True,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto",
        # device_map="cpu",
        # force_download=True
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    return model, tokenizer

model, tokenizer = create_model_and_tokenizer()
model.config.use_cache = False

model.config.quantization_config.to_dict()

# %%
lora_r = 16
lora_alpha = 64
lora_dropout = 0.1
lora_target_modules = [
    "q_proj",
    "up_proj",
    "o_proj",
    "k_proj",
    "down_proj",
    "gate_proj",
    "v_proj",
]


peft_config = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=lora_target_modules,
    bias="none",
    task_type="CAUSAL_LM",
)

# %%
tracking_address = os.path.join(OUTPUT_DIR, 'runs') # the path of your log file.

tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', tracking_address])
url = tb.launch()
print(f"Tensorflow listening on {url}")

# %%
def cleanup_checkpoints(output_dir, keep_last_n=5):
    """
    Keeps the `keep_last_n` most recent checkpoints in the output directory and deletes the rest.
    """
    checkpoint_dirs = [os.path.join(output_dir, d) for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if len(checkpoint_dirs) <= keep_last_n:
        return  # No cleanup needed if <= keep_last_n checkpoints

    # Sort checkpoints by modification time
    checkpoint_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)

    # Delete older checkpoints
    for checkpoint_dir in checkpoint_dirs[keep_last_n:]:
        shutil.rmtree(checkpoint_dir)

# %%

train_steps_per_epoch = len(dataset["train"]) // (2 * 2)  # Assuming per_device_train_batch_size=4 and gradient_accumulation_steps=4

# Number of steps you want to save and evaluate
save_and_eval_steps = train_steps_per_epoch // 20 #30

# Define TrainingArguments with the calculated save_and_eval_steps
training_arguments = TrainingArguments(
    per_device_train_batch_size=3,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    logging_steps=1,
    learning_rate=1e-4,
    fp16=True,
    max_grad_norm=0.3,
    num_train_epochs=2,
    evaluation_strategy="steps",
    eval_steps=save_and_eval_steps,
    save_strategy="steps",
    save_steps=save_and_eval_steps,
    warmup_ratio=0.05,
    group_by_length=True,
    output_dir=OUTPUT_DIR,
    report_to="tensorboard",
    save_safetensors=True,
    lr_scheduler_type="cosine",
    seed=42,
)

# Custom callback to save the model at each evaluation
class SaveCallback(TrainerCallback):
    def on_evaluate(self, args, state, control, **kwargs):
        trainer.save_model()
        cleanup_checkpoints(args.output_dir, keep_last_n=5)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_arguments,
    callbacks=[SaveCallback()],
)

# %%
# Check for the last checkpoint in the output directory
parent_dir = training_arguments.output_dir
# parent_dir = './test/experiments'
chkpts = [file for file in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, file)) and 'checkpoint-' in file]
print(chkpts) #, len(chkpts))

resume = True if len(chkpts) > 0 else None
print(resume)
    
    
print('Beginning Training')
# Start training from the last checkpoint if it exists
trainer.train(resume_from_checkpoint=resume) #latest_checkpoint)
print('Finished Training')

# %%
trainer.save_model()

print(trainer.model)

# %%

trained_model = AutoPeftModelForCausalLM.from_pretrained(
    OUTPUT_DIR,
    low_cpu_mem_usage=True,
)

# Assuming trained_model is the correct object
merged_model = trained_model.merge_and_unload()
merged_model.save_pretrained(NEW_MODEL_NAME, safe_serialization=True)
tokenizer.save_pretrained(NEW_MODEL_NAME)

# %%

# Path to your model and tokenizer
model_path = f"./{NEW_MODEL_NAME}"  # Replace with your model's directory path

# Clean reload of the final version of the model
model = AutoModelForCausalLM.from_pretrained(model_path)

# Clean reload of the final version of the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# %%
# Push the model and tokenizer to the hub
tokenizer.push_to_hub(NEW_MODEL_NAME)
model.push_to_hub(NEW_MODEL_NAME)

print('!!!DONE!!!')
