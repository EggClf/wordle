"""
Inference Script for Wordle Models
Supports both SFT and GRPO trained models
"""

import os
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

def load_model(base_model_path, peft_path=None, device="auto"):
    """
    Load model for inference
    
    Args:
        base_model_path: Path to base model
        peft_path: Optional path to PEFT adapter
        device: Device to load model on
    """
    print(f"Loading tokenizer from {base_model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"Loading model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )
    
    if peft_path:
        print(f"Loading PEFT adapter from {peft_path}...")
        model = PeftModel.from_pretrained(model, peft_path)
        print("Merging adapter weights...")
        model = model.merge_and_unload()
    
    model.eval()
    print("Model loaded successfully!")
    return model, tokenizer

def parse_wordle_response(text):
    """
    Parse Wordle response to extract thinking and guess
    
    Returns:
        tuple: (thinking, guess) or (None, None) if parsing fails
    """
    # Try to extract <think> and <guess> tags
    think_pattern = r"<think>(.*?)</think>"
    guess_pattern = r"<guess>(.*?)</guess>"
    
    think_match = re.search(think_pattern, text, re.DOTALL)
    guess_match = re.search(guess_pattern, text, re.DOTALL)
    
    thinking = think_match.group(1).strip() if think_match else None
    guess = guess_match.group(1).strip() if guess_match else None
    
    return thinking, guess

def generate_wordle_guess(
    model, 
    tokenizer, 
    prompt, 
    max_new_tokens=150,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
):
    """
    Generate Wordle guess from prompt
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter
        do_sample: Whether to use sampling
        num_return_sequences: Number of sequences to generate
    
    Returns:
        list: Generated responses
    """
    # Prepare prompt
    if not prompt.endswith("\n<think>"):
        prompt = prompt + "\n<think>"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode responses
    responses = []
    for output in outputs:
        # Remove input from output
        response = tokenizer.decode(
            output[inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        responses.append(response)
    
    return responses

def interactive_mode(model, tokenizer, args):
    """Interactive Wordle assistant mode"""
    print("\n" + "=" * 70)
    print("Wordle Assistant - Interactive Mode")
    print("=" * 70)
    print("\nCommands:")
    print("  - Type your Wordle prompt")
    print("  - 'multi' to see multiple guesses")
    print("  - 'temp <value>' to change temperature")
    print("  - 'clear' to start new game")
    print("  - 'quit' to exit")
    print("=" * 70 + "\n")
    
    temperature = args.temperature
    game_history = []
    
    while True:
        try:
            user_input = input("Prompt: ").strip()
            
            if not user_input:
                continue
            
            # Handle commands
            if user_input.lower() == 'quit':
                print("Goodbye!")
                break
            
            if user_input.lower() == 'clear':
                game_history = []
                print("Game history cleared!")
                continue
            
            if user_input.lower().startswith('temp '):
                try:
                    temperature = float(user_input.split()[1])
                    print(f"Temperature set to {temperature}")
                except:
                    print("Invalid temperature value")
                continue
            
            if user_input.lower() == 'multi':
                num_guesses = 3
            else:
                num_guesses = 1
            
            # Generate responses
            print("\nGenerating...\n")
            responses = generate_wordle_guess(
                model, 
                tokenizer, 
                user_input,
                max_new_tokens=args.max_new_tokens,
                temperature=temperature,
                top_p=args.top_p,
                num_return_sequences=num_guesses
            )
            
            # Display responses
            for i, response in enumerate(responses, 1):
                if num_guesses > 1:
                    print(f"--- Guess {i} ---")
                
                thinking, guess = parse_wordle_response(response)
                
                if thinking:
                    print(f"Thinking: {thinking[:200]}{'...' if len(thinking) > 200 else ''}")
                if guess:
                    print(f"Guess: {guess}")
                else:
                    print(f"Raw response: {response}")
                
                if num_guesses > 1:
                    print()
            
            print()
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit.")
        except Exception as e:
            print(f"Error: {e}")

def batch_mode(model, tokenizer, args):
    """Batch inference mode"""
    print(f"\nProcessing prompts from {args.input_file}...")
    
    with open(args.input_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(prompts)} prompts")
    
    results = []
    for i, prompt in enumerate(prompts, 1):
        print(f"\nProcessing {i}/{len(prompts)}: {prompt[:50]}...")
        
        responses = generate_wordle_guess(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            num_return_sequences=1
        )
        
        thinking, guess = parse_wordle_response(responses[0])
        results.append({
            "prompt": prompt,
            "thinking": thinking,
            "guess": guess,
            "raw_response": responses[0]
        })
    
    # Save results
    if args.output_file:
        import json
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output_file}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("Summary:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Guess: {result['guess'] or 'N/A'}")
    print("=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Wordle Model Inference")
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        default="/workspace/outputs/wordle-grpo",
        help="Path to trained model (SFT or GRPO)"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        default="/workspace/models/Qwen2.5-0.5B",
        help="Path to base model"
    )
    parser.add_argument(
        "--use_peft",
        action="store_true",
        default=True,
        help="Whether model is a PEFT adapter"
    )
    
    # Generation arguments
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=150,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter"
    )
    
    # Mode arguments
    parser.add_argument(
        "--mode",
        type=str,
        choices=["interactive", "batch"],
        default="interactive",
        help="Inference mode"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input file for batch mode (one prompt per line)"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file for batch mode (JSON)"
    )
    
    args = parser.parse_args()
    
    # Load model
    if args.use_peft:
        model, tokenizer = load_model(
            args.base_model_path,
            peft_path=args.model_path
        )
    else:
        model, tokenizer = load_model(args.model_path)
    
    # Run inference
    if args.mode == "interactive":
        interactive_mode(model, tokenizer, args)
    else:
        if not args.input_file:
            print("Error: --input_file required for batch mode")
            return
        batch_mode(model, tokenizer, args)

if __name__ == "__main__":
    main()
