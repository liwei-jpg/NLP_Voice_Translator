# Import PyTorch, the core machine learning library that handles all the heavy 
# mathematical tensor calculations on the CPU or GPU.
import torch

# Import the load_dataset function from Hugging Face to fetch our data from the cloud.
from datasets import load_dataset

# Import specific tools from the transformers library:
# - MBartForConditionalGeneration: The neural network architecture (the "Brain").
# - MBart50TokenizerFast: The tool that converts human words into numbers.
# - Seq2SeqTrainingArguments / Seq2SeqTrainer: The automated engines that manage the complex training loop.
# - DataCollatorForSeq2Seq: A tool to dynamically add padding to our data batches.
from transformers import (
    MBartForConditionalGeneration, 
    MBart50TokenizerFast, 
    Seq2SeqTrainingArguments, 
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# Import Parameter-Efficient Fine-Tuning (PEFT) tools. 
# LoRA (Low-Rank Adaptation) allows us to train a massive AI on a standard computer 
# by freezing the main model and only training tiny, injected matrices.
from peft import LoraConfig, get_peft_model, TaskType

# Print a status message to the terminal so we know the program is starting.
print("⚙️ Loading model, tokenizer, and dataset...")

# Define the exact name of the Facebook mBART translation model hosted on Hugging Face.
MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

# Download and initialize the Tokenizer. This loads the vocabulary dictionary 
# that the AI uses to map English and Spanish words to mathematical numbers.
tokenizer = MBart50TokenizerFast.from_pretrained(MODEL_NAME)

# Download and initialize the massive base neural network (the 2.4GB mBART model).
model = MBartForConditionalGeneration.from_pretrained(MODEL_NAME)

# Set source and target languages
# We explicitly tell the tokenizer that the input language is English (en_XX).
tokenizer.src_lang = "en_XX"

# We explicitly tell the tokenizer that the target output language is Spanish (es_XX).
tokenizer.tgt_lang = "es_XX"

# Load the dataset and select a small, manageable subset (5,000 sentences) for training
# We download the opus-100 dataset, specifically the English-to-Spanish translation subset.
raw_dataset = load_dataset("Helsinki-NLP/opus-100", "en-es")

# To save massive amounts of compute time, we don't train on all millions of rows.
# .shuffle(seed=42) randomly mixes the data (seed=42 ensures the mix is the exact same every time).
# .select(range(5000)) takes only the first 5,000 random sentences to act as our training material.
train_subset = raw_dataset["train"].shuffle(seed=42).select(range(5000))

# We take 500 different random sentences to act as our "validation" (testing) material 
# to check if the AI is actually learning during the training process.
val_subset = raw_dataset["validation"].shuffle(seed=42).select(range(500))

# --- STEP 3: PREPARE THE DATA FOR AI ---
# --- STEP 3: PREPARE THE DATA FOR AI ---
# Define a function to convert human text into mathematical arrays (tensors).
def preprocess_function(examples):
    # Loop through the batch of data and extract all the English sentences into a list.
    inputs = [ex["en"] for ex in examples["translation"]]
    
    # Loop through the batch of data and extract all the Spanish sentences into a list.
    targets = [ex["es"] for ex in examples["translation"]]
    
    # The modern way: Pass both inputs and targets (text_target) in one single step!
    # We pass both English and Spanish lists into the tokenizer simultaneously.
    model_inputs = tokenizer(
        # The English inputs go here.
        inputs, 
        # The Spanish answers (targets) go here. The tokenizer automatically aligns them.
        text_target=targets, 
        # We cap the maximum sentence length at 128 tokens. This prevents very long 
        # paragraphs from causing Out-Of-Memory (OOM) crashes on our hardware.
        max_length=128, 
        # If a sentence is longer than 128 tokens, we tell the tokenizer to chop off the extra parts.
        truncation=True
    )
    
    # Return the newly created mathematical numbers (tokens).
    return model_inputs

# Print a status update to the terminal.
print("🧹 Preprocessing and cleaning dataset text...")

# Use the .map() function to apply our preprocess_function to every row in the training dataset.
# batched=True processes the data in chunks, which is much faster.
# remove_columns deletes the original human text strings, keeping only the numbers to save RAM!
tokenized_train = train_subset.map(preprocess_function, batched=True, remove_columns=["translation"])

# Apply the exact same preprocessing to our validation (testing) dataset.
tokenized_val = val_subset.map(preprocess_function, batched=True, remove_columns=["translation"])

# --- STEP 4: SET UP LORA (PARAMETER-EFFICIENT FINE-TUNING) ---
# Print a status update to the terminal.
print("🧬 Applying LoRA layers to the model...")

# Define the configuration for Low-Rank Adaptation (LoRA).
# Why we use this: Instead of updating all 600+ million parameters of mBART, 
# LoRA freezes the main model and injects tiny, trainable matrices.
peft_config = LoraConfig(
    # Tells LoRA that we are doing Sequence-to-Sequence translation.
    task_type=TaskType.SEQ_2_SEQ_LM,
    
    # r=8: The "Rank" or dimensionality reduction factor. 
    # This dictates how small our injected matrix is. 8 is very small and fast to train.
    r=8,           # Dimensionality reduction factor
    
    # lora_alpha=16: The scaling factor. This tells the neural network how much it should 
    # "listen" to our new LoRA weights compared to the original base weights.
    lora_alpha=16, # Scaling factor
    
    # Dropout=0.1 means we randomly turn off 10% of the neurons during training.
    # This prevents "overfitting" (preventing the AI from just memorizing the data).
    lora_dropout=0.1,
    
    # Target Modules: The Attention mechanism inside the Transformer has Query (Q) and Value (V) weights.
    # We specifically tell LoRA to inject our tiny matrices only into these specific mathematical areas.
    target_modules=["q_proj", "v_proj"] # Target the attention weights
)

# We wrap our massive base mBART model in this lightweight LoRA configuration.
model = get_peft_model(model, peft_config)

# This prints a summary to the terminal showing how many parameters are actually trainable.
# It proves to the user that we are only training a fraction of a percent of the model!
model.print_trainable_parameters() # Prints how much memory we saved!

# --- STEP 5: CONFIGURING THE TRAINER ---
# Set up the hyper-parameters that dictate how the machine learning loop behaves.
training_args = Seq2SeqTrainingArguments(
    # The folder where our checkpoint files will be saved in case the computer crashes.
    output_dir="./mbart_es_results",
    
    # Tells the trainer to test the model's accuracy at the end of every full pass (epoch).
    eval_strategy="epoch",
    
    # Learning rate: How big of a mathematical "step" the optimizer takes when fixing errors.
    # 5e-4 (0.0005) is a standard, stable learning rate for LoRA adapters.
    learning_rate=5e-4,
    
    # We process 4 sentences at a time during training to ensure we don't run out of memory.
    per_device_train_batch_size=4,
    
    # We process 4 sentences at a time during the validation testing phase.
    per_device_eval_batch_size=4,
    
    # Weight decay adds a tiny penalty to prevent weights from growing infinitely large.
    weight_decay=0.01,
    
    # Only keep the 1 most recent checkpoint on the hard drive to save disk space.
    save_total_limit=1,
    
    # How many times the AI will read through the entire 5,000 sentence dataset.
    num_train_epochs=1,          # We will start with 1 quick test epoch
    
    # Forces the model to actually generate full Spanish sentences during the evaluation phase.
    predict_with_generate=True,
    
    # Checks if a GPU (CUDA) is available. If yes, it uses 16-bit math instead of 32-bit.
    # This doubles the training speed and cuts memory consumption in half!
    fp16=torch.cuda.is_available(), # Use GPU acceleration if available
    
    # Print a progress update to the terminal screen every 50 steps.
    logging_steps=50,
    
    # Disables sending our training analytics to external third-party servers.
    report_to="none"
)

# The Data Collator acts as an automated packager. 
# Because sentences are different lengths, GPUs cannot process them in neat rectangles.
# The collator dynamically adds blank "padding" to the end of short sentences in a batch so they match.
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Construct the main Trainer object. We feed it our model, our arguments, 
# our training data, our testing data, the tokenizer, and the padding collator.
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# --- STEP 6: START TRAINING ---
# Print a status update to the terminal.
print("🚀 Starting fine-tuning training process...")

# Execute the automated training loop. The AI begins the math process to learn here!
trainer.train()

# Save the specialized adapter layers
# Print a status update to the terminal.
print("💾 Saving your fine-tuned LoRA weights...")

# Save ONLY the newly trained LoRA adapter weights (which is a tiny ~5MB file), 
# rather than saving the entire 2.4GB model. This makes the system highly portable!
model.save_pretrained("./saved_lora_adapter")

# Print the final success message to the terminal.
print("🎉 Done! Your adapter weights are saved in './saved_lora_adapter'")