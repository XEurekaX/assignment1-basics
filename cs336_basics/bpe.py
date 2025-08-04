import os
import regex as re
from typing import BinaryIO
from multiprocessing import Pool
from collections import defaultdict

def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int = 8
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    #  1.词汇表初始化 
    #  分词器词汇表是从字节串 token 到整数 ID 的一对一映射。
    #  由于训练的是字节级 BPE 分词器，初始词汇表就是所有字节的集合。
    #  由于存在 256 种可能的字节值，初始词汇表大小为 256。
    vocab = {i: bytes([i]) for i in range(256)}
    for tok in special_tokens:
        vocab[len(vocab)] = tok.encode("utf-8")


    #  2. 预分词处理 
    # 一旦建立了词汇表，理论上你可以统计文本中字节相邻出现的频率，并从出现频率最高的字节对开始合并。
    # 然而，这种方法计算成本很高，因为每次合并都需要对整个语料库进行一次完整扫描。此外，直接在语料库中合并字节
    # 可能会导致仅因标点符号不同而产生差异的 token（例如 dog!与 dog.）。这些 token 会被分配完全不同的 token ID，尽管
    # 它们在语义上可能高度相似（仅标点符号不同）。
    # 为避免这种情况，我们需要对语料库进行预分词处理。你可以将其视为一种粗粒度的分词方法，帮助我们统计字符对出
    # 现的频率。例如，单词"text"可能作为一个预分词单元出现了 10 次。这种情况下，当我们统计字符't'和'e'相邻出现的次数
    # 时，会发现单词"text"中包含相邻的't'和'e'，此时我们可以直接将它们的计数增加 10，而无需遍历整个语料库。由于我们
    # 训练的是字节级 BPE 模型，每个预分词单元都表示为 UTF-8 字节序列。
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, "<|endoftext|>".encode("utf-8"))

    task_args = [(input_path, start, end, special_tokens) for start, end in zip(boundaries[:-1], boundaries[1:])]
    with Pool(processes=num_processes) as pool:
        chunk_results = pool.map(process_chunk, task_args)
    
    # 3. 计算 BPE 合并 
    # 现在，我们已将输入文本转换为预标记，并将每个预标记表示为 UTF-8 字节序列，接下来可以计算 BPE
    # 合并（即训练 BPE 分词器）。简而言之，BPE 算法会迭代统计每个字节对，并找出频率最高的组合（"A", "B"）。随后将
    # 这个最高频的字节对（"A", "B"）的所有出现实例合并，即替换为新标记"AB"。这个新合并的标记会被加入我们的词汇
    # 表；因此，BPE 训练后的最终词汇表大小等于初始词汇表（本例中为 256）加上训练期间执行的 BPE 合并操作次数。为
    # 提高 BPE 训练效率，我们不考虑跨越预标记边界的字节对。在计算合并时，若遇到频率相同的字节对，则按照字典序
    # 优先合并较大的组合。例如，若字节对（"A", "B"）、（"A", "C"）、（"B", "ZZ"）和（"BA", "A"）都具有最高频率，我们
    # 将优先合并（"BA", "A"）。
    merges : list[tuple[bytes, bytes]] = []
    pre_tokens_bytes: list[list[bytes]] = [token for chunk in chunk_results for token in chunk]
    counts = defaultdict(int)
    pair_to_indices = defaultdict(set)
    for idx, token in enumerate(pre_tokens_bytes):
        for i in range(len(token) - 1):
            pair = (token[i], token[i + 1])
            counts[pair] += 1
            pair_to_indices[pair].add(idx)

    idx = len(vocab)
    while idx < vocab_size:
        if not counts:
            break
            
        max_pair: tuple[bytes, bytes] = None
        max_cnt= -1
        for pair, cnt in counts.items():
            if cnt > max_cnt:
                max_pair = pair
                max_cnt = cnt
            elif cnt == max_cnt:
                if max_pair is None or pair > max_pair:
                    max_pair = pair

        merges.append(max_pair)
        a, b = max_pair
        new_token = a + b
        vocab[idx] = new_token
        idx += 1

        affected_indices = pair_to_indices[max_pair].copy()
        for j in affected_indices:
            token = pre_tokens_bytes[j]
            for i in range(len(token) - 1):
                old_pair = (token[i], token[i+1])
                pair_to_indices[old_pair].discard(j)
                counts[old_pair] -= 1
                if counts[old_pair] == 0:
                    counts.pop(old_pair)
                    pair_to_indices.pop(old_pair, None)

            merged = []
            i = 0
            while i < len(token):
                if i < len(token) - 1 and token[i] == a and token[i+1]==b:
                    merged.append(new_token)
                    i += 2
                else:
                    merged.append(token[i])
                    i += 1
            pre_tokens_bytes[j]=merged

            token = pre_tokens_bytes[j]
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                counts[pair] += 1
                pair_to_indices[pair].add(j)

    return vocab, merges

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                true_position = initial_position + found_at
                chunk_boundaries[bi] = true_position
                break

            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))



def process_chunk(args: tuple[str, int, int, list[str]]) -> list[list[bytes]]:
    input_path, start, end, special_tokens = args
    """
    Processes a chunk of the input file and returns byte pair frequency counts.

    Args:
        input_path (str): The path of input file.
        start (int): Start byte offset of the chunk.
        end (int): End byte offset of the chunk.
        special_tokens (list[str]): List of special tokens that should not be merged across.

    Returns:
        pre_token_bytes (list[list[bytes]]): list of tokens, where each token is a list of bytes
    """

    with open(input_path, "rb") as file:
        file.seek(start)
        chunk = file.read(end - start).decode("utf-8", errors="ignore")

    # 1. Remove special tokens by splitting the chunk at those tokens
    pattern = "|".join(re.escape(tok) for tok in special_tokens)
    documents = re.split(pattern, chunk)

    # 2. Pre-tokenize and count byte pair frequencies
    pre_tokens_bytes: list[list[bytes]] = []
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    for doc in documents:
        tokens = [match.group(0).encode("utf-8") for match in re.finditer(PAT, doc)]
        for token in tokens:
            token_bytes = [bytes([b]) for b in token]
            pre_tokens_bytes.append(token_bytes)

    return pre_tokens_bytes

