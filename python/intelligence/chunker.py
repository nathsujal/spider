import hashlib
import re

import tiktoken

from python.models import Chunk, Document, Section


class Chunker:
    """
    Splits a Document into contextual Chunks for storage in Spider.

    Strategy (recursive boundary-aware splitting):
      1. If document is short (< chunk_size tokens) → single chunk, no sections
      2. Otherwise, iterate sections → split each section on paragraph
         boundaries → fall back to sentence boundaries → hard split

    Chunks carry their section title so the graph builder can create
    DOCUMENT → SECTION → CHUNK hierarchy.
    """

    def __init__(
        self,
        *,
        chunk_size: int = 512,
        min_chunk_size: int = 50,
        tiktoken_model: str = "cl100k_base",
    ):
        self.chunk_size = chunk_size
        self.min_chunk_size = min_chunk_size
        self._enc = tiktoken.get_encoding(tiktoken_model)

    def token_count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def chunk(self, doc: Document) -> list[Chunk]:
        """
        Main entry point. Takes a Document and returns a list of Chunks.

        Short documents (< chunk_size tokens) get a single chunk with
        no section hierarchy (flat fallback).
        """
        total_tokens = self.token_count(doc.raw_text)

        # flat fallback for short documents
        if total_tokens <= self.chunk_size:
            return [self._make_chunk(
                text=doc.raw_text,
                index=0,
                section_title="Full Document",
                page_num=1,
            )]

        # if the loader didn't detect sections, treat entire text as one section
        sections = doc.sections
        if not sections:
            sections = [Section(
                title="Full Document",
                level=1,
                text=doc.raw_text,
                page_start=1,
            )]

        chunks: list[Chunk] = []
        global_index = 0

        for section in sections:
            section_chunks = self._chunk_section(section, global_index)
            chunks.extend(section_chunks)
            global_index += len(section_chunks)

        return chunks

    def _chunk_section(self, section: Section, start_index: int) -> list[Chunk]:
        """Split a single section's text into chunks."""

        text = section.text.strip()
        if not text:
            return []

        tokens = self.token_count(text)

        # section fits in one chunk
        if tokens <= self.chunk_size:
            if tokens < self.min_chunk_size:
                return []  # too small to be useful
            return [self._make_chunk(
                text=text,
                index=start_index,
                section_title=section.title,
                page_num=section.page_start,
            )]

        # split on paragraph boundaries first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        chunk_texts = self._group_by_size(paragraphs)

        # if any resulting chunk is still too large, split on sentences
        final_texts: list[str] = []
        for ct in chunk_texts:
            if self.token_count(ct) > self.chunk_size:
                sentences = self._split_sentences(ct)
                final_texts.extend(self._group_by_size(sentences))
            else:
                final_texts.append(ct)

        # build Chunk objects
        chunks: list[Chunk] = []
        for i, text_block in enumerate(final_texts):
            text_block = text_block.strip()
            if not text_block:
                continue
            tc = self.token_count(text_block)
            if tc < self.min_chunk_size:
                # merge into previous chunk if possible
                if chunks:
                    prev = chunks[-1]
                    merged = f"{prev.text}\n\n{text_block}"
                    chunks[-1] = self._make_chunk(
                        text=merged,
                        index=prev.index,
                        section_title=section.title,
                        page_num=section.page_start,
                    )
                    continue

            chunks.append(self._make_chunk(
                text=text_block,
                index=start_index + len(chunks),
                section_title=section.title,
                page_num=section.page_start,
            ))

        return chunks

    def _group_by_size(self, segments: list[str]) -> list[str]:
        """
        Group text segments (paragraphs or sentences) into chunks
        that don't exceed chunk_size tokens.
        """
        groups: list[str] = []
        current: list[str] = []
        current_tokens = 0

        for seg in segments:
            seg_tokens = self.token_count(seg)

            # single segment exceeds chunk_size → add as-is (will be
            # further split at sentence level by the caller)
            if seg_tokens > self.chunk_size:
                if current:
                    groups.append("\n\n".join(current))
                    current = []
                    current_tokens = 0
                groups.append(seg)
                continue

            # would adding this segment exceed the limit?
            if current_tokens + seg_tokens > self.chunk_size:
                groups.append("\n\n".join(current))
                current = [seg]
                current_tokens = seg_tokens
            else:
                current.append(seg)
                current_tokens += seg_tokens

        if current:
            groups.append("\n\n".join(current))

        return groups

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        """Split text into sentences using regex."""
        # split on . ! ? followed by whitespace and uppercase letter
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        return [p.strip() for p in parts if p.strip()]

    def _make_chunk(
        self,
        text: str,
        index: int,
        section_title: str,
        page_num: int,
    ) -> Chunk:
        return Chunk(
            text=text,
            index=index,
            section_title=section_title,
            page_num=page_num,
            token_count=self.token_count(text),
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
        )