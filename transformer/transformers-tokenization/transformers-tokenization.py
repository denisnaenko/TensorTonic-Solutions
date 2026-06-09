import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        for i, token in enumerate(special_tokens):
            self.word_to_id[token] = i
            self.id_to_word[i] = token
            self.vocab_size += 1
            
        unique = set()
        for text in texts:
            [unique.add(word.lower()) for word in text.split(" ")]

        offset = 4
        for i, word in enumerate(sorted(unique)):
            if word not in special_tokens:
                self.word_to_id[word] = i + offset
                self.id_to_word[i + offset] = word
                self.vocab_size += 1

                
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        normalized_words = [word.lower() for word in text.split(" ") if word]
        encoded = []
        for word in normalized_words:
            encoded.append(self.word_to_id.get(word, 1))

        return encoded
        
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        decoded = [self.id_to_word.get(id, self.unk_token) for id in ids]
        return " ".join(decoded)