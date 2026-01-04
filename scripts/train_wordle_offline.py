"""
Offline Training Script for Wordle SFT + GRPO
Uses locally saved model and dataset with advanced reward functions
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

import torch
import re
import ast
import math
import pandas as pd
from datasets import load_from_disk, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, GRPOConfig, GRPOTrainer
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================

class Config:
    # Paths (for offline mode)
    model_name = "/workspace/models/Qwen2.5-0.5B"
    dataset_path = "/workspace/data/wordle-grpo"
    word_list_path = "/workspace/data/wordle-words.csv"  # Will be created if not exists
    
    # Output paths
    sft_output_dir = "/workspace/outputs/wordle-grpo"
    grpo_output_dir = "/workspace/outputs/wordle-grpo"
    
    # LoRA Configuration
    lora_r = 16
    lora_alpha = 32
    lora_dropout = 0.05
    lora_target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
    
    # SFT Training Configuration
    sft_num_epochs = 3
    sft_batch_size = 4
    sft_gradient_accumulation_steps = 4
    sft_learning_rate = 2e-4
    sft_max_seq_length = 512
    sft_warmup_ratio = 0.1
    
    # GRPO Training Configuration
    grpo_num_epochs = 2
    grpo_batch_size = 2
    grpo_gradient_accumulation_steps = 4
    grpo_learning_rate = 1e-5
    grpo_num_generations = 4
    grpo_temperature = 0.7
    grpo_max_new_tokens = 128
    
    # Reward weights
    reward_weight_format = 1.0
    reward_weight_feedback = 1.0
    reward_weight_info_gain = 0.5
    
    # Hardware
    use_flash_attention = True
    use_4bit = False
    bf16 = True
    gradient_checkpointing = True
    
    # Misc
    seed = 42

config = Config()

# ============================================
# Reward Functions
# ============================================

class WordleRewardFunctions:
    """Advanced reward functions for Wordle GRPO training"""
    
    @staticmethod
    def output_format_check(prompt: str, completion: str, example: dict) -> float:
        """Check if output follows the correct format: <think>...</think><guess>...</guess>"""
        reward = 0.0
        try:
            # Add synthetic <think> tag
            completion = "<think>" + completion
            
            # Check format pattern
            regex = (
                r"^<think>\s*([^<]*(?:<(?!/?think>)[^<]*)*)\s*<\/think>\n"
                r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
            )
            
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 2:
                return 0.0
            
            guess = match.groups()[1].strip()
            
            # Basic validation: must be 5 characters
            if len(guess) != 5:
                return 0.1
            
            # Check if it's a valid word
            if "word_list" in example:
                try:
                    word_list = pd.read_csv(str(example["word_list"]))
                    if guess.upper() not in word_list["Word"].str.upper().values:
                        return 0.5
                except Exception:
                    pass
            
            reward = 1.0
        except Exception as e:
            logger.debug(f"Format check error: {e}")
            pass
        
        return reward
    
    @staticmethod
    def uses_previous_feedback(prompt: str, completion: str, example: dict) -> float:
        """Reward for using previous feedback correctly"""
        reward = 0.0
        try:
            # Add synthetic <think> tag
            completion = "<think>" + completion
            
            # Extract guess
            regex = r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 1:
                return 0.0
            
            guess = match.groups()[0].strip()
            if len(guess) != 5:
                return 0.0
            
            # Parse past guess history
            past_guess_history = ast.literal_eval(example.get("past_guess_history", "[]"))
            if len(past_guess_history) == 0:
                return 0.1  # Small reward for first guess
            
            # Build feedback constraints
            correct_letter_to_position = {}
            valid_letter_to_position = {}
            wrong_letter_to_position = {}
            
            for past_guess, past_feedback in past_guess_history:
                past_feedback_parts = past_feedback.split(" ")
                for i, fb in enumerate(past_feedback_parts):
                    if '✓' in fb:
                        letter = fb[0]
                        if letter not in correct_letter_to_position:
                            correct_letter_to_position[letter] = set()
                        correct_letter_to_position[letter].add(i)
                    elif '-' in fb:
                        letter = fb[0]
                        if letter not in valid_letter_to_position:
                            valid_letter_to_position[letter] = set()
                        valid_letter_to_position[letter].add(i)
                    else:
                        letter = fb[0]
                        if letter not in wrong_letter_to_position:
                            wrong_letter_to_position[letter] = set()
                        wrong_letter_to_position[letter].add(i)
            
            # Evaluate current guess
            for idx, letter in enumerate(guess):
                # Correct position reused
                if (letter in correct_letter_to_position and 
                    idx in correct_letter_to_position[letter]):
                    reward += 0.2
                # Valid letter in new position (exploration)
                elif (letter in valid_letter_to_position and 
                      idx not in valid_letter_to_position[letter]):
                    reward += 0.1
                # Valid letter in same wrong position (bad)
                elif (letter in valid_letter_to_position and 
                      idx in valid_letter_to_position[letter]):
                    reward -= 0.2
                # Using known-wrong letter (penalty)
                elif letter in wrong_letter_to_position:
                    reward -= 0.5
                else:
                    # Unknown letter (exploration bonus)
                    reward += 0.05
        
        except Exception as e:
            logger.debug(f"Feedback check error: {e}")
            return 0.0
        
        return reward
    
    @staticmethod
    def guess_value(prompt: str, completion: str, example: dict) -> float:
        """Compute normalized information gain of the guess"""
        
        def validate_guess(secret: str, guess: str, raw_feedback: bool = False):
            """Generate feedback for a guess"""
            feedback = []
            secret_list = list(secret)
            
            # Check correct positions
            for i, (g_char, s_char) in enumerate(zip(guess, secret)):
                if g_char == s_char:
                    feedback.append(f"{g_char}(✓) ")
                    secret_list[i] = None
                else:
                    feedback.append(None)
            
            # Check misplaced letters
            for i, g_char in enumerate(guess):
                if feedback[i] is None:
                    if g_char in secret_list:
                        feedback[i] = f"{g_char}(-) "
                        secret_list[secret_list.index(g_char)] = None
                    else:
                        feedback[i] = f"{g_char}(x) "
            
            if raw_feedback:
                return feedback
            return "".join(feedback).strip()
        
        def filter_candidates(all_words, past_guesses):
            """Filter candidate words based on past feedback"""
            filtered = []
            for word in all_words:
                valid = True
                for past_guess, past_feedback in past_guesses:
                    candidate_feedback = validate_guess(word, past_guess)
                    if candidate_feedback != past_feedback:
                        valid = False
                        break
                if valid:
                    filtered.append(word)
            return filtered
        
        def compute_normalized_information_gain(all_words, past_guesses, guess):
            """Compute information gain of a guess"""
            candidates = filter_candidates(all_words, past_guesses)
            total_candidates = len(candidates)
            
            if total_candidates == 0:
                return 0.0
            
            current_entropy = math.log2(total_candidates)
            
            # Group by feedback patterns
            feedback_groups = {}
            for word in candidates:
                feedback = validate_guess(word, guess, raw_feedback=True)
                pattern = "".join(
                    '1' if "✓" in fb else ('0' if "-" in fb else 'x')
                    for fb in feedback
                )
                feedback_groups.setdefault(pattern, []).append(word)
            
            expected_entropy = 0.0
            for group in feedback_groups.values():
                group_size = len(group)
                p = group_size / total_candidates
                group_entropy = math.log2(group_size) if group_size > 0 else 0
                expected_entropy += p * group_entropy
            
            expected_gain = current_entropy - expected_entropy
            normalized_gain = expected_gain / current_entropy if current_entropy > 0 else 0
            
            return normalized_gain
        
        reward = 0.0
        try:
            # Add synthetic <think> tag
            completion = "<think>" + completion
            
            # Extract guess
            regex = r"<guess>\s*([\s\S]*?)\s*<\/guess>$"
            match = re.search(regex, completion, re.DOTALL)
            if match is None or len(match.groups()) != 1:
                return 0.0
            
            guess = match.groups()[0].strip()
            if len(guess) != 5:
                return 0.0
            
            # Load word list
            if "word_list" not in example:
                return 0.0
            
            word_list = pd.read_csv(str(example["word_list"]))
            if guess.upper() not in word_list["Word"].str.upper().values:
                return 0.0
            
            # Get past guesses
            past_guess_history = ast.literal_eval(example.get("past_guess_history", "[]"))
            
            # Compute information gain
            normalized_gain = compute_normalized_information_gain(
                word_list["Word"].values,
                past_guess_history,
                guess.upper()
            )
            
            reward = normalized_gain
        
        except Exception as e:
            logger.debug(f"Info gain error: {e}")
            return 0.0
        
        return reward

# ============================================
# Helper Functions
# ============================================

def setup_model_and_tokenizer(model_path, use_4bit=False, use_flash_attention=True):
    """Load model and tokenizer"""
    logger.info(f"Loading tokenizer from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side="right"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    logger.info(f"Loading model from {model_path}")
    
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if config.bf16 else torch.float16,
        "device_map": "auto",
    }
    
    if use_flash_attention:
        model_kwargs["attn_implementation"] = "flash_attention_2"
    
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
    
    return model, tokenizer

def format_wordle_prompt(example):
    """Format example into prompt-response pair"""
    if "text" in example:
        return {"text": example["text"]}
    elif "prompt" in example and "completion" in example:
        return {"text": f"{example['prompt']}\n\n{example['completion']}"}
    else:
        # Combine all text fields
        text = " ".join([str(v) for v in example.values() if isinstance(v, str)])
        return {"text": text}

def create_word_list():
    """Create word list CSV if it doesn't exist"""
    if not os.path.exists(config.word_list_path):
        logger.info("Creating default word list...")
        # Create a basic word list (you can expand this)
        words = [
            "CRANE", "SLATE", "AUDIO", "RAISE", "STARE",
            "HOUSE", "LODGE", "PIECE", "JUDGE", "APPLE",
            "BEACH", "BREAD", "CHAIR", "DREAM", "EARTH",
            # Add more common 5-letter words
        ]
        df = pd.DataFrame({"Word": words})
        os.makedirs(os.path.dirname(config.word_list_path), exist_ok=True)
        df.to_csv(config.word_list_path, index=False)
        logger.info(f"Word list created at {config.word_list_path}")

# ============================================
# Stage 1: SFT Training
# ============================================

def train_sft():
    """Stage 1: Supervised Fine-Tuning"""
    logger.info("=" * 60)
    logger.info("STAGE 1: Supervised Fine-Tuning (SFT)")
    logger.info("=" * 60)
    
    # Load model and tokenizer
    model, tokenizer = setup_model_and_tokenizer(
        config.model_name,
        use_4bit=config.use_4bit,
        use_flash_attention=config.use_flash_attention
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    logger.info(f"Loading dataset from {config.dataset_path}")
    dataset = load_from_disk(config.dataset_path)
    
    # Format dataset
    dataset = dataset.map(format_wordle_prompt, remove_columns=dataset.column_names)
    
    # Split dataset
    dataset = dataset.train_test_split(test_size=0.2, seed=config.seed)
    
    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Eval samples: {len(dataset['test'])}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.sft_output_dir,
        num_train_epochs=config.sft_num_epochs,
        per_device_train_batch_size=config.sft_batch_size,
        per_device_eval_batch_size=config.sft_batch_size,
        gradient_accumulation_steps=config.sft_gradient_accumulation_steps,
        learning_rate=config.sft_learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=config.sft_warmup_ratio,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=25,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=3,
        fp16=False,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim="adamw_torch",
        seed=config.seed,
        report_to=["tensorboard"],
        remove_unused_columns=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )
    
    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        max_seq_length=config.sft_max_seq_length,
        dataset_text_field="text",
        packing=False,
    )
    
    # Train
    logger.info("Starting SFT training...")
    trainer.train()
    
    # Save
    logger.info(f"Saving model to {config.sft_output_dir}")
    trainer.save_model(config.sft_output_dir)
    tokenizer.save_pretrained(config.sft_output_dir)
    
    logger.info("SFT training completed!")
    return config.sft_output_dir

# ============================================
# Stage 2: GRPO Training
# ============================================

def train_grpo(sft_model_path):
    """Stage 2: Group Relative Policy Optimization"""
    logger.info("=" * 60)
    logger.info("STAGE 2: GRPO Training")
    logger.info("=" * 60)
    
    # Load SFT model
    logger.info(f"Loading SFT model from {sft_model_path}")
    model, tokenizer = setup_model_and_tokenizer(
        sft_model_path,
        use_4bit=config.use_4bit,
        use_flash_attention=config.use_flash_attention
    )
    
    # Load dataset
    dataset = load_from_disk(config.dataset_path)
    
    # Extract queries and add metadata
    def prepare_grpo_example(example):
        # Extract prompt
        if "prompt" in example:
            query = example["prompt"]
        elif "text" in example:
            query = example["text"].split("\n")[0]
        else:
            query = str(list(example.values())[0])
        
        return {
            "query": query,
            "word_list": config.word_list_path,
            "past_guess_history": example.get("past_guess_history", "[]"),
        }
    
    dataset = dataset.map(prepare_grpo_example)
    
    # Combined reward function
    reward_functions = WordleRewardFunctions()
    
    def combined_reward_function(prompts, completions, examples):
        """Combine multiple reward signals"""
        rewards = []
        
        for prompt, completion, example in zip(prompts, completions, examples):
            total_reward = 0.0
            
            # Format check (must pass)
            format_reward = reward_functions.output_format_check(prompt, completion, example)
            total_reward += config.reward_weight_format * format_reward
            
            # Only continue if format is correct
            if format_reward > 0.5:
                # Feedback usage
                feedback_reward = reward_functions.uses_previous_feedback(
                    prompt, completion, example
                )
                total_reward += config.reward_weight_feedback * feedback_reward
                
                # Information gain
                info_gain_reward = reward_functions.guess_value(
                    prompt, completion, example
                )
                total_reward += config.reward_weight_info_gain * info_gain_reward
            
            rewards.append(total_reward)
        
        return rewards
    
    # GRPO Configuration
    grpo_config = GRPOConfig(
        output_dir=config.grpo_output_dir,
        num_train_epochs=config.grpo_num_epochs,
        per_device_train_batch_size=config.grpo_batch_size,
        gradient_accumulation_steps=config.grpo_gradient_accumulation_steps,
        learning_rate=config.grpo_learning_rate,
        logging_steps=5,
        save_strategy="steps",
        save_steps=25,
        save_total_limit=2,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        report_to=["tensorboard"],
        seed=config.seed,
        # GRPO specific
        num_generations=config.grpo_num_generations,
        temperature=config.grpo_temperature,
        max_new_tokens=config.grpo_max_new_tokens,
    )
    
    # Initialize trainer
    trainer = GRPOTrainer(
        model=model,
        config=grpo_config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_function=combined_reward_function,
    )
    
    # Train
    logger.info("Starting GRPO training...")
    trainer.train()
    
    # Save
    logger.info(f"Saving model to {config.grpo_output_dir}")
    trainer.save_model(config.grpo_output_dir)
    tokenizer.save_pretrained(config.grpo_output_dir)
    
    logger.info("GRPO training completed!")
    return config.grpo_output_dir

# ============================================
# Main
# ============================================

def main():
    logger.info("=" * 60)
    logger.info("Wordle SFT + GRPO Training Pipeline (Offline Mode)")
    logger.info("=" * 60)
    logger.info(f"Model: {config.model_name}")
    logger.info(f"Dataset: {config.dataset_path}")
    logger.info(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"GPUs: {torch.cuda.device_count()}")
    logger.info("=" * 60)
    
    # Create word list if needed
    create_word_list()
    
    # Stage 1: SFT
    sft_model_path = train_sft()
    
    # Stage 2: GRPO
    grpo_model_path = train_grpo(sft_model_path)
    
    logger.info("=" * 60)
    logger.info("Training Pipeline Completed!")
    logger.info(f"SFT Model: {sft_model_path}")
    logger.info(f"GRPO Model: {grpo_model_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
