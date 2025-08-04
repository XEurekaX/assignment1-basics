import json
from typing import Iterator
import regex as re
class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        初始化Tokenizer实例
        
        Args:
            vocab: 词汇表，将整数ID映射到对应的字节序列
            merges: 合并规则列表，每个元素是包含两个字节序列的元组
            special_tokens: 特殊标记列表，按长度降序排列，默认为None
        """
        self.vocab = vocab
        # 创建反向词汇表，将字节序列映射到整数ID
        self.vocab_reversed = {v: k for k, v in self.vocab.items()}  # bytes: int
        self.merges = merges
        # 将特殊标记按长度降序排列，以便优先匹配较长的特殊标记
        self.special_tokens = sorted(special_tokens or [], key=lambda x: -len(x))
        
        # 创建合并映射，将合并对映射到其在merges列表中的索引位置
        self.merge_map = {pair: i for i, pair in enumerate(merges)}

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        """
        从词汇表文件和合并规则文件创建Tokenizer实例
        
        Args:
            vocab_filepath: 词汇表文件路径，文件应包含JSON格式的词汇表数据
            merges_filepath: 合并规则文件路径，每行包含两个用空格分隔的token
            special_tokens: 特殊标记列表，默认为None
            
        Returns:
            Tokenizer: 新创建的Tokenizer实例
        """
        vocab: dict[int, bytes] = {}
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            data = json.load(vocab_filepath)
            vocab = {k.encode('utf-8'): int(v) for k, v in data.items()}

        
        merges :list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    merges.append(parts[0].encode('utf-8'), parts[1].encode('utf-8'))


        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        将文本编码为token ID序列
        
        Args:
            text: 需要编码的文本字符串
            
        Returns:
            list[int]: 文本对应的token ID列表
        """
        token_ids = []
        pre_tokens_list = process_chunk((text, self.special_tokens, True))
        
        for tokens in pre_tokens_list:
            # 应用BPE合并算法
            tokens = self._bpe_merge(tokens)
            
            for token in tokens:
                token_ids.append(self.vocab_reversed.get(token))
        
        return token_ids

    def _bpe_merge(self, tokens: list[bytes]) -> list[bytes]:
        """
        使用BPE算法合并token序列
        
        Args:
            tokens: 字节序列列表
            
        Returns:
            list[bytes]: 合并后的字节序列列表
        """
        if len(tokens) <= 1:
            return tokens
        
        current_tokens = list(tokens)
        
        while len(current_tokens) > 1:
            best_merge_index = None
            best_merge_rank = float('inf')

            for i in range(len(current_tokens) - 1):
                pair = (current_tokens[i], current_tokens[i+1])
                if pair in self.merge_map:
                    rank = self.merge_map[pair]
                    if rank < best_merge_rank:
                        best_merge_rank = rank
                        best_merge_index = i
            
            if best_merge_index is None:
                break
                
            first_token = current_tokens[best_merge_index]
            second_token = current_tokens[best_merge_index + 1]
            merged_token = first_token + second_token
            
            current_tokens[best_merge_index] = merged_token
            current_tokens.pop(best_merge_index + 1)
            
        return current_tokens

    def encode_iterable(self, iterable: list[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings into a stream of token IDs.

        Useful for memory-efficient tokenization of large datasets.

        Args:
            iterable (list[str]): An iterable of strings (e.g., lines from a file).

        Returns:
            iter: A generator that yields token IDs one at a time.
        """
        for line in iterable:
            token_ids = self.encode(line)
            for token_id in token_ids:
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        tokens = bytes()
        vocab_size = len(self.vocab)
        replacement_char = "\uFFFD"

        for token_id in ids:
            if token_id < vocab_size:
                token = self.vocab[token_id]  
            else:
                token = bytes(replacement_char, encoding='utf-8')   # Replace tokens with Unicode replacement characters if index out of bounds

            tokens += token
        decoded = tokens.decode(encoding='utf-8', errors='replace')

        return decoded 

def process_chunk(args: tuple[str, list[str], bool]) -> list[list[bytes]]:
    """
    处理文本块，将其分割为预token序列
    
    Args:
        args: 包含三个元素的元组：
            - chunk: 需要处理的文本块
            - special_tokens: 特殊标记列表
            - keep_special_tokens: 是否保留特殊标记的布尔值
            
    Returns:
        list[list[bytes]]: 预token字节序列的列表
    """
    chunk, special_tokens, keep_special_tokens = args
    pattern = "|".join(re.escape(tok) for tok in special_tokens) if special_tokens else ""
    if keep_special_tokens and pattern:
        pattern = f"({pattern})"
    segments = re.split(pattern, chunk) if pattern else [chunk]

    pre_tokens_bytes: list[list[bytes]] = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    for segment in segments:
        if keep_special_tokens and special_tokens and segment in special_tokens:
            token_bytes = [segment.encode("utf-8")]
            pre_tokens_bytes.append(token_bytes)
        else:
            tokens = [match.group(0).encode("utf-8") for match in re.finditer(PAT, segment)]
            for token in tokens:
                token_bytes = [bytes([b]) for b in token]
                pre_tokens_bytes.append(token_bytes)
    return pre_tokens_bytes