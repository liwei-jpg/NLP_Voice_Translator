import torch
from datasets import load_dataset
from transformers import (
    MBartForConditionalGeneration, 
    MBart50TokenizerFast, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType

print("⚙️ Loading model, tokenizer, and dataset...")

MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

# Set source and target languages
tokenizer.src_lang = "en_XX"
tokenizer.tgt_lang = "es_XX"

# Load the dataset and select a small, manageable subset (5,000 sentences) for training
raw_dataset = load_dataset("Helsinki-NLP/opus-100", "en-es")
train_subset = raw_dataset["train"].shuffle(seed=42).select(range(5000))
val_subset = raw_dataset["validation"].shuffle(seed=42).select(range(500))

# --- STEP 3: PREPARE THE DATA FOR AI ---
# --- STEP 3: PREPARE THE DATA FOR AI ---
def preprocess_function(examples):
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["es"] for ex in examples["translation"]]
    
    # The modern way: Pass both inputs and targets (text_target) in one single step!
    model_inputs = tokenizer(
        inputs, 
        text_target=targets, 
        max_length=128, 
        truncation=True
    )
    
    return model_inputs

print("🧹 Preprocessing and cleaning dataset text...")
tokenized_train = train_subset.map(preprocess_function, batched=True, remove_columns=["translation"])
tokenized_val = val_subset.map(preprocess_function, batched=True, remove_columns=["translation"])

# --- STEP 4: SET UP LORA (PARAMETER-EFFICIENT FINE-TUNING) ---
print("🧬 Applying LoRA layers to the model...")
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=8,           # Dimensionality reduction factor
    lora_alpha=16, # Scaling factor
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"] # Target the attention weights
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Prints how much memory we saved!

# --- STEP 5: CONFIGURING THE TRAINER ---
training_args = Seq2SeqTrainingArguments(
    output_dir="./mbart_es_results",
    eval_strategy="epoch",
    learning_rate=5e-4,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=1,          # We will start with 1 quick test epoch
    predict_with_generate=True,
    fp16=torch.cuda.is_available(), # Use GPU acceleration if available
    logging_steps=50,
    report_to="none"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# --- STEP 6: START TRAINING ---
print("🚀 Starting fine-tuning training process...")
trainer.train()

# Save the specialized adapter layers
print("💾 Saving your fine-tuned LoRA weights...")
model.save_pretrained("./saved_lora_adapter")
print("🎉 Done! Your adapter weights are saved in './saved_lora_adapter'")