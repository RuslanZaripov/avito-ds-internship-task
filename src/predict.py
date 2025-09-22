import torch
import argparse
import sys
from typing import List, Union
from logger import Logger
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    LogitsProcessor, LogitsProcessorList
)


class EDConstrainedLogitsProcessor(LogitsProcessor):
    def __init__(self, encoder_inputs: List[List[int]], pad_token_id: int, space_token_id: int, eos_token_id: int):
        """
        encoder_inputs: list of lists (for each batch element) containing encoder input ids 
        (without special padding or with padding - we'll trim)
        """
        self.encoder_inputs = [self._trim(inp, pad_token_id) for inp in encoder_inputs]
        self.pad = pad_token_id
        self.space_id = space_token_id
        self.eos_id = eos_token_id

    def _trim(self, arr, pad_id):
        # remove trailing pad tokens if present
        trimmed = [int(x) for x in arr if int(x) != pad_id]
        return trimmed

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        input_ids: (batch_size, cur_len)
        scores: (batch_size, vocab_size) (logits for next token)
        Return masked scores.
        """
        bs = input_ids.size(0)
        vocab_size = scores.size(-1)
        device = scores.device
        big_neg = -1e9

        masked = scores.clone()

        for i in range(bs):
            enc = self.encoder_inputs[i]
            gen = input_ids[i].tolist()
            
            # compute position in encoder: count how many times we consumed a source token
            pos = 0
            for tok in gen:
                if pos < len(enc) and tok == enc[pos]:
                    pos += 1

            allowed = {self.space_id}
            if pos < len(enc):
                allowed.add(enc[pos])
            if pos >= len(enc):
                # Allow EOS after consuming all input
                allowed.add(self.eos_id)
            
            # Create mask
            mask = torch.full((vocab_size,), big_neg, device=device)
            allowed_list = list(allowed)
            mask[allowed_list] = 0.0
            masked[i] = scores[i] + mask

        return masked


class SpaceCorrectionPredictor:
    def __init__(self, model_name: str = "zarus03/byt5-wsc", device: str = None, logger: Logger = None):
        """
        Initialize the space correction predictor.
        
        Args:
            model_name: HuggingFace model name or path
            device: Device to run on ('cuda', 'cpu', or None for auto-detection)
            logger: Logger instance for logging
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.logger = logger if logger else Logger(show=True).get_logger("SpaceCorrectionPredictor")
        
        self.logger.info(f"Loading model: {model_name}")
        self.logger.info(f"Using device: {self.device}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
            
            self.pad_id = self.tokenizer.pad_token_id
            self.space_id = self.tokenizer.encode(" ", add_special_tokens=False)[0]
            self.eos_id = self.tokenizer.eos_token_id
            
            self.logger.info("Model loaded successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def _build_logits_processor_for_batch(self, batch_input_ids):
        """Build logits processor for constrained decoding"""
        encoder_inputs = [row.tolist() for row in batch_input_ids]
        ed = EDConstrainedLogitsProcessor(
            encoder_inputs=encoder_inputs,
            pad_token_id=self.pad_id,
            space_token_id=self.space_id,
            eos_token_id=self.eos_id
        )
        return LogitsProcessorList([ed])
    
    def _generate_constrained(self, batch, max_length: int):
        """
        Generate text with constrained decoding.
        
        Args:
            batch: dict with 'input_ids' and 'attention_mask'
            max_length: maximum generation length
            
        Returns:
            Generated sequences
        """
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        logits_processor = self._build_logits_processor_for_batch(
            input_ids.detach().cpu()
        )

        self.logger.debug(f"Generating text with max_length: {max_length}")
        
        generated = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_length,
            logits_processor=logits_processor,
            return_dict_in_generate=True,
            output_scores=False,
            do_sample=False,
        )

        return generated.sequences
    
    def predict(self, text: Union[str, List[str]], max_length: int = 128) -> Union[str, List[str]]:
        """
        Correct spaces in input text(s).
        
        Args:
            text: Single string or list of strings
            max_length: Maximum sequence length
            
        Returns:
            Corrected string or list of corrected strings
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        self.logger.info(f"Processing {len(texts)} text(s) with max_length: {max_length}")
        
        if len(texts) > 1:
            self.logger.debug(f"Batch size: {len(texts)}")
        
        try:
            inputs = self.tokenizer(
                texts, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=max_length
            ).to(self.device)
            
            generated_sequences = self._generate_constrained(inputs, max_length)
            
            results = [
                self.tokenizer.decode(seq, skip_special_tokens=True).strip()
                for seq in generated_sequences
            ]
            
            self.logger.info("Prediction completed successfully")
            
            return results[0] if is_single else results
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Space Correction Predictor")
    parser.add_argument("--model", default="zarus03/byt5-wsc", 
                       help="Model name or path (default: zarus03/byt5-wsc)")
    parser.add_argument("--device", choices=['cuda', 'cpu'], 
                       help="Device to use (default: auto-detect)")
    parser.add_argument("--max_length", type=int, default=128,
                       help="Maximum sequence length (default: 128)")
    parser.add_argument("--log_level", default="INFO", 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help="Logging level (default: INFO)")
    parser.add_argument("--no_console_log", action="store_true",
                       help="Disable console logging")
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--text", help="Input text to correct")
    
    args = parser.parse_args()
    
    logger_obj = Logger(show=not args.no_console_log)
    logger =  logger_obj.get_logger("SpaceCorrectionPredictor")
    
    logger.info("Starting Space Correction Predictor")
    logger.debug(f"Command line arguments: {vars(args)}")
    
    try:
        predictor = SpaceCorrectionPredictor(
            model_name=args.model,
            device=args.device,
            logger=logger
        )
        
        if args.text:
            logger.info("Processing single text input")
            result = predictor.predict(args.text, args.max_length)
            logger.info(f"Input:  {args.text}")
            logger.info(f"Output: {result}")
            print(f"Input:  {args.text}")
            print(f"Output: {result}")
        
        logger.info("Space Correction Predictor completed successfully")
        
    except Exception as e:
        logger.error(f"Application failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
