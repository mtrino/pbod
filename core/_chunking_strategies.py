def fixed_size_chunking(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    """Returns a list of text chunks according to the chunk size and overlap sizes given"""
    chunks = []
    start, end = 0, 0

    while end < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)

    return chunks
