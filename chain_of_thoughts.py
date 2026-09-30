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

# Names used for TEST questions. None of them appear in the exemplars.
FIRST = ["Amy", "Daniel", "Waldo", "Phoebe", "Osvaldo", "Andree", "Audrie", "Dallas", "Marcus",
         "Priya", "Tomas", "Yusuf", "Helga", "Lucia", "Nikhil", "Farah", "Gregor", "Imani"]
LAST = ["Brown", "Schmidt", "Friedman", "Okafor", "Lindqvist", "Moreau", "Tanaka", "Delgado",
        "Petrov", "Hassan", "Novak", "Fischer", "Costa", "Kowalski", "Bianchi", "Nakamura"]
COIN_NAMES = FIRST + ["Reuben", "Thandi", "Mikael", "Soraya", "Bertrand", "Ottilie"]

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

def _letters_chain(words):
    last = [w[-1].lower() for w in words]
    s = " ".join(f'The last letter of "{w}" is "{c}".' for w, c in zip(words, last))
    return f'{s} Concatenating them is "{"".join(last)}". The answer is {"".join(last)}.'

def _letters_ex(name):
    words = name.split()
    return (f'Take the last letters of the words in "{name}" and concatenate them.',
            _letters_chain(words), "".join(w[-1].lower() for w in words))

LETTERS_EX = [_letters_ex(n) for n in ["Elon Musk", "Larry Page", "Sergey Brin", "Bill Gates"]]

# ------------------------------------------------------------------ task generators
def make_coin(k, rng):
    names = rng.sample(COIN_NAMES, k)
    flips = [rng.random() < 0.5 for _ in names]
    parts = [f"{n} flips the coin." if f else f"{n} does not flip the coin." for n, f in zip(names, flips)]
    q = "A coin is heads up. " + " ".join(parts) + " Is the coin still heads up?"
    return q, ("yes" if sum(flips) % 2 == 0 else "no")

def make_letters(k, rng):
    words = [rng.choice(FIRST if i % 2 == 0 else LAST) for i in range(k)]
    q = f'Take the last letters of the words in "{" ".join(words)}" and concatenate them.'
    return q, "".join(w[-1].lower() for w in words)

TASKS = {"coinflip": (make_coin, COIN_EX), "letters": (make_letters, LETTERS_EX)}

def build_prompt(exemplars, question, mode):
    blocks = [f"Q: {q}\nA: " + (cot if mode == "cot" else f"The answer is {ans}.") for q, cot, ans in exemplars]
    blocks.append(f"Q: {question}\nA:")
    return "\n\n".join(blocks)

def make_hf(model_name):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    def gen(prompt, max_new_tokens, mode):
        inputs = tok(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return gen


if __name__ == "__main__":
    main()
