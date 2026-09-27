#!/usr/bin/env python3
"""
Mini-replication of the symbolic-reasoning experiments in
Wei et al. (2022), "Chain-of-Thought Prompting Elicits Reasoning in LLMs" (Sec. 5, Fig. 8).

Tasks (generated programmatically, no dataset needed):
  * coinflip : "A coin is heads up. Ann flips the coin. Bob does not flip ... still heads up?" -> yes/no
  * letters  : "Take the last letters of the words in "Amy Brown" and concatenate them." -> "yn"

Conditions:
  * standard : few-shot exemplars contain only the answer   ("A: The answer is yes.")
  * cot      : few-shot exemplars contain a chain of thought before the answer
Exemplars always use 2 steps (2 people / 2 words); tests use 2 (in-domain) and 3, 4 (out-of-domain).
Exemplar prompts are taken from the paper's Appendix G, Tables 22-23.

Usage examples:
  python chain_of_thoughts.py --backend mock                                   # pipeline test, no model needed
  python chain_of_thoughts.py --backend hf --model Qwen/Qwen2.5-1.5B --n 100
  ANTHROPIC_API_KEY=... python chain_of_thoughts.py --backend anthropic --model claude-haiku-4-5-20251001 --n 50
"""
import argparse, json, math, random, re, sys, time

  python cot_toy.py --backend mock                                   # pipeline test, no model needed
  python cot_toy.py --backend hf --model Qwen/Qwen2.5-1.5B --n 100
  ANTHROPIC_API_KEY=... python cot_toy.py --backend anthropic --model claude-haiku-4-5-20251001 --n 50
"""

if __name__ == "__main__":
    main()
