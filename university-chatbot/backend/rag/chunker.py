"""Text chunking with sliding-window overlap."""


def chunk_text(
    text: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[str]:
    """
    Split text into chunks of approximately *chunk_size* characters
    with *chunk_overlap* characters of overlap between consecutive chunks.

    Splits preferentially on paragraph boundaries (double newlines),
    then single newlines, then spaces.
    """
    if not text or not text.strip():
        return []

    segments = _recursive_split(text, ["\n\n", "\n", " "], chunk_size)

    chunks: list[str] = []
    current = ""

    for segment in segments:
        if len(current) + len(segment) <= chunk_size:
            current += segment
        else:
            if current.strip():
                chunks.append(current.strip())
            # Start new chunk with overlap from end of previous
            if chunk_overlap > 0 and current:
                overlap_text = current[-chunk_overlap:]
                current = overlap_text + segment
            else:
                current = segment

    if current.strip():
        chunks.append(current.strip())

    return chunks


def _recursive_split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Split text using the first separator that produces segments <= chunk_size."""
    if len(text) <= chunk_size:
        return [text]

    for sep in separators:
        parts = text.split(sep)
        if all(len(p) <= chunk_size for p in parts):
            result = []
            for i, part in enumerate(parts):
                result.append(part + (sep if i < len(parts) - 1 else ""))
            return result

    # Fallback: hard split at chunk_size boundaries
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
