# --- LIBRARY IMPORTS & EXPLANATIONS ---

# Imports PyTorch, the core machine learning engine that handles all the heavy GPU/CPU mathematical calculations.
import torch 

# Imports the dataset downloader from Hugging Face so we can grab our training data.
from datasets import load_dataset 

# Imports the specific tools we need from Hugging Face's Transformers library:
# - MBartForConditionalGeneration: The blueprint for the mBART translation AI.
# - MBart50TokenizerFast: The tool that chops human words into mathematical numbers (tokens) the AI understands.
# - Seq2SeqTrainingArguments / Seq2SeqTrainer: The engine that automatically handles the training loop (so we don't have to write the complex math loops manually).
# - DataCollatorForSeq2Seq: A tool that dynamically adds "padding" (blank spaces) to sentences so they are all the exact same length in a batch.
from transformers import (
    MBartForConditionalGeneration, 
    MBart50TokenizerFast, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# Imports PEFT (Parameter-Efficient Fine-Tuning). This is the library that contains the LoRA architecture.
# TaskType helps tell LoRA what kind of task we are doing (Seq2Seq = Sequence to Sequence translation).
from peft import LoraConfig, get_peft_model, TaskType 

# Print a status message to the terminal so we know the script has started successfully.
print("⚙️ Loading model, tokenizer, and dataset...")

# --- STEP 1: LOAD THE BASE MODEL ---
# Define the exact name of the Facebook translation model hosted on the Hugging Face cloud.
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

# Download and load the Tokenizer (the dictionary that translates words into numbers).
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)

# Download and load the actual Neural Network (the "Brain").
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

# Tell the tokenizer what language the input text is (English).
tokenizer.src_lang = "en_XX"
# Tell the tokenizer what language the output/target text should be (Spanish).
tokenizer.tgt_lang = "es_XX"


# --- STEP 2: LOAD & SHRINK THE DATASET ---
# Download the massive opus-100 dataset containing millions of English-Spanish translations.
raw_dataset = load_dataset("Helsinki-NLP/opus-100", "en-es")

# To save time and compute power for this project, we don't need millions of sentences.
# We shuffle the data randomly (seed=42 ensures the randomization is the same every time we run it),
# and we select only the first 5,000 sentences to teach the AI.
train_subset = raw_dataset["train"].shuffle(seed=42).select(range(5000))

# We select 500 different sentences to test the AI during training to see if it is learning properly.
val_subset = raw_dataset["validation"].shuffle(seed=42).select(range(500))


# --- STEP 3: PREPARE THE DATA FOR AI (TOKENIZATION) ---
# AI models cannot read text; they only read numbers (tensors). 
# This function converts our text datasets into mathematical arrays.
def preprocess_function(examples):
    # Extract the English sentences and put them in a list.
    inputs = [ex["en"] for ex in examples["translation"]]
    # Extract the Spanish sentences and put them in a list.
    targets = [ex["es"] for ex in examples["translation"]]
    
    # Pass the English inputs and Spanish targets to the Tokenizer.
    model_inputs = tokenizer(
        inputs, 
        text_target=targets, # Automatically aligns the answers (targets) with the questions (inputs).
        max_length=128,      # Cap the sentence length at 128 tokens. This prevents long paragraphs from crashing the computer's memory.
        truncation=True      # If a sentence is longer than 128 tokens, chop off the extra parts.
    )
    
    # Return the newly created numbers (model_inputs).
    return model_inputs

print("🧹 Preprocessing and cleaning dataset text...")
# Use the .map() function to apply our preprocess_function to every single sentence in our 5000-sentence training set.
# batched=True makes it process in chunks, which is much faster.
# remove_columns=["translation"] deletes the original text. We only want to keep the numbers to save RAM!
tokenized_train = train_subset.map(preprocess_function, batched=True, remove_columns=["translation"])
tokenized_val = val_subset.map(preprocess_function, batched=True, remove_columns=["translation"])


# --- STEP 4: SET UP LORA (PARAMETER-EFFICIENT FINE-TUNING) ---
print("🧬 Applying LoRA layers to the model...")

# LORA EXPLANATION FOR PRESENTATION:
# Standard fine-tuning modifies all 600+ million parameters inside mBART, which requires massive, expensive GPUs.
# LoRA (Low-Rank Adaptation) freezes the original model completely. Instead, it injects tiny, 
# new matrices (layers) into the model's attention mechanism. We only train these tiny new layers.
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM, # Tells LoRA we are doing a Sequence-to-Sequence task (Translation).
    r=8,                             # 'Rank': The size of the tiny matrix we are injecting. r=8 is very small and fast.
    lora_alpha=16,                   # 'Alpha': A scaling multiplier. It tells the model how much to "listen" to our new LoRA weights vs the original base weights.
    lora_dropout=0.1,                # Randomly turns off 10% of the neurons during training to prevent the model from memorizing the data (Overfitting).
    target_modules=["q_proj", "v_proj"] # Specifically injects our LoRA matrices into the "Query" and "Value" parts of the Transformer's Attention mechanism.
)

# Wrap our massive base mBART model in this lightweight LoRA configuration.
model = get_peft_model(model, peft_config)

# Prints out a terminal message showing how many parameters are trainable. 
# (Usually shows we are only training ~0.5% of the model, saving 99.5% of memory!)
model.print_trainable_parameters() 


# --- STEP 5: CONFIGURING THE TRAINER ---
# These are the hyper-parameters that govern the training loop.
training_args = Seq2SeqTrainingArguments(
    output_dir="./mbart_es_results", # The folder where intermediate checkpoints will be saved in case of a crash.
    eval_strategy="epoch",           # Test the model's accuracy at the end of every epoch (1 full pass of the data).
    learning_rate=5e-4,              # How big of a mathematical "step" the optimizer takes when adjusting weights. 5e-4 is standard for LoRA.
    per_device_train_batch_size=4,   # Feeds 4 sentences into the GPU at a time during training.
    per_device_eval_batch_size=4,    # Feeds 4 sentences into the GPU at a time during evaluation.
    weight_decay=0.01,               # A regularization penalty to prevent weights from getting too large (prevents overfitting).
    save_total_limit=1,              # Only keep the 1 most recent checkpoint to save hard drive space.
    num_train_epochs=1,              # Train by looking at the 5,000 sentences exactly 1 time.
    predict_with_generate=True,      # Required for translation: forces the model to generate full Spanish sentences during evaluation.
    fp16=torch.cuda.is_available(),  # If a GPU is present, use 16-bit math instead of 32-bit. This doubles training speed and cuts memory usage in half!
    logging_steps=50,                # Print a progress update to the terminal every 50 batches.
    report_to="none"                 # Disables sending analytics to external trackers like Weights & Biases.
)

# The data collator acts as a dynamic packager. Since sentences are different lengths, 
# it adds "padding" (0s) to the end of short sentences so every batch forms a perfect rectangle matrix for the GPU.
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Construct the Trainer object by feeding it the model, the settings, the data, and the collator.
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
# Execute the automated training loop. The AI begins learning here!
trainer.train()

# --- STEP 7: SAVE THE FINAL ADAPTER ---
print("💾 Saving your fine-tuned LoRA weights...")

# Save ONLY the newly trained LoRA adapter weights (a tiny file), NOT the whole mBART model.
# This makes it highly portable and easy to upload to GitHub.
model.save_pretrained("./saved_lora_adapter")

print("🎉 Done! Your adapter weights are saved in './saved_lora_adapter'")