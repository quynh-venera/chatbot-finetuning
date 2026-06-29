"""
mcq_benchmark.py

Stage 4: MCQ (multiple-choice question) benchmark evaluation.

Prerequisites:
- Fine-tuned GGUF model loaded in Unsloth Studio Chat
- OPENAI_API_KEY set in .env (for distractor generation only)
- UNSLOTH_STUDIO_API_KEY set in .env (sk-unsloth-... key from Studio Settings)
- Run from the chatbot-finetuning project root

Usage:
    python -m src.eval.mcq_benchmark
    python -m src.eval.mcq_benchmark --test-file data/processed/test.jsonl
    python -m src.eval.mcq_benchmark --model-url http://127.0.0.1:8888/v1
"""

import argparse
import datetime
import json
import os
import random
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

TEST_FILE = Path("data/processed/test.jsonl")
EVAL_OUTPUT_DIR = Path("data/eval")

STUDIO_API_BASE = "http://127.0.0.1:8888/v1"
STUDIO_MODEL_NAME = "/Users/quynhdang/.unsloth/studio/outputs/unsloth_Qwen2.5-1.5B-Instruct_1782772883"

NUM_DISTRACTORS = 3
RANDOM_SEED = 42

SYSTEM_PROMPT = (
    "You are a helpful assistant for Venera AI, a health technology company. "
    "Answer questions accurately and concisely based on Venera AI's products, "
    "features, and documentation. If you don't know the answer, say so clearly. "
    "You do not provide medical advice or diagnoses — only information about how "
    "Venera AI's platform works."
)

DISTRACTOR_PROMPT = """\
You are building a multiple-choice quiz. Given a question and the correct answer,
generate exactly {n} plausible but incorrect answer choices. The wrong answers
should be believable — similar in length and style to the correct answer, but
factually wrong or subtly different in a meaningful way. Do NOT include the
correct answer. Return ONLY a JSON array of {n} strings, no preamble or fences.

Question: {question}
Correct answer: {correct_answer}
"""

MCQ_EVAL_PROMPT = """\
Answer the following multiple-choice question by replying with ONLY the letter
of the correct answer (A, B, C, or D). Do not explain your choice.

Question: {question}

A) {a}
B) {b}
C) {c}
D) {d}
"""

CHOICE_LETTERS = ["A", "B", "C", "D"]

studio_client = OpenAI(
    api_key=os.environ.get("UNSLOTH_STUDIO_API_KEY", ""),
    base_url=STUDIO_API_BASE,
)


def load_test_pairs(path: Path) -> list[dict]:
    pairs = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            messages = record.get("messages", [])
            question = next(
                (m["content"] for m in messages if m["role"] == "user"), None
            )
            answer = next(
                (m["content"] for m in messages if m["role"] == "assistant"), None
            )
            if question and answer:
                pairs.append({"question": question, "answer": answer})
    return pairs


def generate_distractors(question: str, correct_answer: str) -> list[str]:
    prompt = DISTRACTOR_PROMPT.format(
        n=NUM_DISTRACTORS,
        question=question,
        correct_answer=correct_answer,
    )
    try:
        response = studio_client.chat.completions.create(
            model=STUDIO_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500,
        )
        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        # Extract just the JSON array — the local model sometimes appends
        # extra text or newlines after the closing bracket.
        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if not match:
            raise ValueError("No JSON array found in response")
        distractors = json.loads(match.group(0))
        if not isinstance(distractors, list):
            raise ValueError(f"Expected list, got {type(distractors)}")
        return [str(d) for d in distractors[:NUM_DISTRACTORS]]
    except Exception as e:
        print(f"[mcq] Distractor generation failed: {e}. Using fallback distractors.")
        return [
            "This feature is not available in Venera AI.",
            "Venera AI does not support this functionality.",
            "This is handled by a third-party provider, not Venera AI directly.",
        ]


def build_mcq(question: str, correct_answer: str, distractors: list[str], rng: random.Random) -> dict:
    # Defensive padding: if the model returned fewer than NUM_DISTRACTORS
    # distractors (e.g. it stopped early, or returned a short list that
    # still parsed as valid JSON), pad with generic fallbacks so we always
    # end up with exactly 4 total choices. Without this, dict(zip(...))
    # silently truncates to the shorter list and choices like "C" or "D"
    # go missing, crashing ask_model() with a KeyError.
    generic_fallbacks = [
        "This feature is not available in Venera AI.",
        "Venera AI does not support this functionality.",
        "This is handled by a third-party provider, not Venera AI directly.",
    ]
    distractors = list(distractors[:NUM_DISTRACTORS])
    i = 0
    while len(distractors) < NUM_DISTRACTORS and i < len(generic_fallbacks):
        if generic_fallbacks[i] not in distractors:
            distractors.append(generic_fallbacks[i])
        i += 1

    choices = [correct_answer] + distractors[:NUM_DISTRACTORS]
    rng.shuffle(choices)
    correct_letter = CHOICE_LETTERS[choices.index(correct_answer)]
    return {
        "choices": dict(zip(CHOICE_LETTERS, choices)),
        "correct_letter": correct_letter,
    }


def ask_model(question: str, choices: dict) -> str | None:
    prompt = MCQ_EVAL_PROMPT.format(
        question=question,
        a=choices["A"],
        b=choices["B"],
        c=choices["C"],
        d=choices["D"],
    )
    try:
        response = studio_client.chat.completions.create(
            model=STUDIO_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=10,
        )
        raw = (response.choices[0].message.content or "").strip().upper()
        match = re.search(r"\b([ABCD])\b", raw)
        return match.group(1) if match else None
    except Exception as e:
        print(f"[mcq] Model call failed: {e}")
        return None


def run_benchmark(test_file: Path, model_url: str) -> dict:
    global studio_client
    studio_client = OpenAI(
        api_key=os.environ.get("UNSLOTH_STUDIO_API_KEY", ""),
        base_url=model_url,
    )

    EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(RANDOM_SEED)

    pairs = load_test_pairs(test_file)
    print(f"[mcq] Loaded {len(pairs)} test pairs from {test_file}")

    results = []
    correct_count = 0

    for i, pair in enumerate(pairs):
        question = pair["question"]
        correct_answer = pair["answer"]

        print(f"[mcq] Q{i+1}/{len(pairs)}: generating distractors...")
        distractors = generate_distractors(question, correct_answer)

        mcq = build_mcq(question, correct_answer, distractors, rng)
        model_letter = ask_model(question, mcq["choices"])
        is_correct = model_letter == mcq["correct_letter"]

        if is_correct:
            correct_count += 1

        result = {
            "question_index": i + 1,
            "question": question,
            "correct_answer": correct_answer,
            "correct_letter": mcq["correct_letter"],
            "choices": mcq["choices"],
            "model_answer_letter": model_letter,
            "model_answer_text": mcq["choices"].get(model_letter, None) if model_letter else None,
            "is_correct": is_correct,
        }
        results.append(result)

        status = "✓" if is_correct else "✗"
        print(f"  {status} Model picked {model_letter}, correct was {mcq['correct_letter']}")

    total = len(pairs)
    score_pct = round(correct_count / total * 100, 1) if total > 0 else 0.0

    summary = {
        "date": datetime.date.today().isoformat(),
        "test_file": str(test_file),
        "model_url": model_url,
        "total_questions": total,
        "correct": correct_count,
        "incorrect": total - correct_count,
        "score_pct": score_pct,
        "results": results,
    }

    today = datetime.date.today().isoformat()
    out_path = EVAL_OUTPUT_DIR / f"mcq_results_{today}.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[mcq] Score: {correct_count}/{total} ({score_pct}%)")
    print(f"[mcq] Results written to {out_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 4 MCQ benchmark.")
    parser.add_argument(
        "--test-file",
        type=Path,
        default=TEST_FILE,
        help=f"Path to test JSONL file (default: {TEST_FILE})",
    )
    parser.add_argument(
        "--model-url",
        type=str,
        default=STUDIO_API_BASE,
        help=f"Unsloth Studio API base URL (default: {STUDIO_API_BASE})",
    )
    args = parser.parse_args()
    run_benchmark(args.test_file, args.model_url)


if __name__ == "__main__":
    main()
