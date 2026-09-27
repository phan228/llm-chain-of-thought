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

# ------------------------------------------------------------------ exemplars (Tables 22, 23)
def _coin_chain(flippers):
    k = len(flippers)
    if k == 0:
        return ("The coin was flipped by no one. So the coin was flipped 0 times. The coin started heads up, "
                "and it was not flipped, so it is still heads up. So the answer is yes.")
    who = " and ".join(flippers)
    if k % 2 == 0:
        return (f"The coin was flipped by {who}. So the coin was flipped {k} times, which is an even number. "
                "The coin started heads up, so after an even number of flips, it will still be heads up. So the answer is yes.")
    return (f"The coin was flipped by {who}. So the coin was flipped {k} time, which is an odd number. "
            "The coin started heads up, so after an odd number of flips, it will be tails up. So the answer is no.")

def _coin_ex(a, a_flips, b, b_flips):
    parts = [f"{n} flips the coin." if f else f"{n} does not flip the coin." for n, f in ((a, a_flips), (b, b_flips))]
    q = "A coin is heads up. " + " ".join(parts) + " Is the coin still heads up?"
    flippers = [n for n, f in ((a, a_flips), (b, b_flips)) if f]
    ans = "yes" if len(flippers) % 2 == 0 else "no"
    return (q, _coin_chain(flippers), ans)

COIN_EX = [
    _coin_ex("Ka", True, "Sherrie", True),
    _coin_ex("Jamey", True, "Teressa", True),
    _coin_ex("Maybelle", True, "Shalonda", False),
    _coin_ex("Millicent", False, "Conception", True),
    _coin_ex("Sal", True, "Raymond", False),
    _coin_ex("Conception", True, "Kristian", False),
    _coin_ex("Inga", False, "Elanor", False),
    _coin_ex("Ryan", True, "Shaunda", True),
]

# ------------------------------------------------------------------ task generators
def make_coin(k, rng):
    names = rng.sample(COIN_NAMES, k)
    flips = [rng.random() < 0.5 for _ in names]
    parts = [f"{n} flips the coin." if f else f"{n} does not flip the coin." for n, f in zip(names, flips)]
    q = "A coin is heads up. " + " ".join(parts) + " Is the coin still heads up?"
    return q, ("yes" if sum(flips) % 2 == 0 else "no")


if __name__ == "__main__":
    main()
