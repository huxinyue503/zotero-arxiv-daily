from functools import cached_property
from typing import Optional
from loguru import logger
from llm import get_llm
import tiktoken
import re


class OpenAlexPaper:
    def __init__(self, raw: dict):
        self._raw = raw
        self.score = None

    @property
    def title(self) -> str:
        return self._raw.get("title", "")

    @property
    def summary(self) -> str:
        abs_inv = self._raw.get("abstract_inverted_index")
        if not abs_inv:
            return ""
        words = []
        for k, idxs in abs_inv.items():
            for i in idxs:
                if i >= len(words):
                    words.extend([""] * (i - len(words) + 1))
                words[i] = k
        return " ".join(words)

    @property
    def authors(self) -> list[str]:
        return [
            a["author"]["display_name"]
            for a in self._raw.get("authorships", [])
        ]

    @property
    def pdf_url(self) -> Optional[str]:
        return self._raw.get("primary_location", {}).get("pdf_url")

    @cached_property
    def code_url(self) -> Optional[str]:
        return None   # OpenAlex 不提供 code repo

    @cached_property
    def tldr(self) -> str:
        llm = get_llm()
        prompt = f"""Given the title and abstract of a paper, generate a one-sentence TLDR summary in {llm.lang}:

Title: {self.title}
Abstract: {self.summary}
"""
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)[:4000]
        prompt = enc.decode(prompt_tokens)

        tldr = llm.generate(
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        return tldr

    @cached_property
    def affiliations(self) -> Optional[list[str]]:
        try:
            affs = []
            for a in self._raw.get("authorships", []):
                insts = a.get("institutions", [])
                for i in insts:
                    if "display_name" in i:
                        affs.append(i["display_name"])
            return list(set(affs)) if affs else None
        except Exception as e:
            logger.debug(f"Failed to extract affiliations: {e}")
            return None

