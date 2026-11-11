"""
IITGN RAG - Answer Evaluation (section 4, step 3 of the design document)

Runs every question in evaluation/questions.jsonl through the full pipeline
(retrieval + LLM answer) and grades the answers:
  correct   : does it give the key facts of the expected answer?        (LLM judge)
  faithful  : is every claim stated in the sources the LLM was given?   (LLM judge)
  citations : does each [n] point to a source that supports the claim?  (LLM judge)
  declined  : for the questions with no answer in the data, did it say so? (text check)

The judge is an LLM too (GEMINI_JUDGE_MODEL), so treat the scores as a guide
and read the listed problems yourself.

The free Gemini tier allows only about 20 requests per model per day, and each question
needs 2. So finished questions are saved after each one, and a new run skips them.
The run stops early when several questions fail in a row (usually the daily limit).
To start over, for example after changing the prompt, delete output/answer_evaluation.json.

Input : evaluation/questions.jsonl
Output: a printed report and output/answer_evaluation.json
Run   : python main.py evaluate-answers
"""

import json
import re
from pathlib import Path

from config import GEMINI_JUDGE_MODEL, LLM_PROVIDER
from evaluate import load_questions
from generate import LAST_MODEL, NOT_FOUND, Assistant, create_llm

RESULTS_FILE = Path("output/answer_evaluation.json")
LOG_FILE = Path("output/answer_evaluation_log.jsonl")   # Kept apart from the chat log
MAX_FAILURES_IN_A_ROW = 3

POINTS = {"yes": 1.0, "partly": 0.5, "no": 0.0}   # Score for each "correct" grade

JUDGE_PROMPT = """You grade answers written by an IIT Gandhinagar question-answering assistant.
You get the question, the expected answer written by a person, the numbered sources the assistant was given, and the assistant's answer.

Reply with JSON only, in this form:
{"correct": "yes" | "partly" | "no", "faithful": "yes" | "no", "citations": "yes" | "no", "reason": "<one short sentence>"}

- correct: does the answer give the key facts of the expected answer? Extra correct detail is fine.
  "partly" if some key facts are missing or one is wrong. "no" if it misses the point or says it cannot find the answer.
- faithful: is every claim in the answer stated in the sources? Judge only against the sources, not your own knowledge.
- citations: does each [n] point to a source that supports the sentence it follows? "no" if any citation is wrong or the answer cites nothing.
- reason: name the main problem, or "fine"."""


# ---------- Grading ----------

def parse_verdict(text):
    """Read the judge's JSON reply. Returns None if it cannot be read."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        verdict = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return None
    grades = {key: str(verdict.get(key, "")).strip().lower() for key in ("correct", "faithful", "citations")}
    if grades["correct"] not in POINTS or grades["faithful"] not in ("yes", "no") \
            or grades["citations"] not in ("yes", "no"):
        return None
    return {**grades, "reason": str(verdict.get("reason", ""))}


def judge(judge_llm, question, reply):
    user = (f"Question: {question['question']}\n\n"
            f"Expected answer: {question['answer']}\n\n"
            f"Sources given to the assistant:\n{reply['context']}\n\n"
            f"Assistant's answer:\n{reply['answer']}")
    text = "".join(judge_llm(JUDGE_PROMPT, user))
    return parse_verdict(text) or {"judge_error": text[:300]}


def load_finished():
    """Questions answered and graded in earlier runs."""
    if not RESULTS_FILE.exists():
        return []
    records = json.loads(RESULTS_FILE.read_text(encoding="utf-8"))
    return [record for record in records if "error" not in record and "judge_error" not in record]


def save(records):
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    finished = [record for record in records if "error" not in record]
    RESULTS_FILE.write_text(json.dumps(finished, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------- Report ----------

def share(count, total):
    return f"{count}/{total} ({count / total:.0%})" if total else "0/0"


def print_report(records):
    failed = [record for record in records if "error" in record]
    done = [record for record in records if "error" not in record]
    answerable = [record for record in done if record["answerable"]]
    judged = [record for record in answerable if "judge_error" not in record]
    no_answer = [record for record in done if not record["answerable"]]

    print(f"\n===== Answers: {len(answerable)} answerable questions ({len(judged)} graded) =====")
    if judged:
        grades = [record["correct"] for record in judged]
        score = sum(POINTS[grade] for grade in grades) / len(judged)
        print(f"Correct             : {grades.count('yes')} yes, {grades.count('partly')} partly, "
              f"{grades.count('no')} no  (score {score:.2f})")
        print(f"Faithful to sources : {share(sum(r['faithful'] == 'yes' for r in judged), len(judged))}")
        print(f"Citations correct   : {share(sum(r['citations'] == 'yes' for r in judged), len(judged))}")
    print(f"Wrongly 'not found' : {sum(record['declined'] for record in answerable)}")
    print(f"Made-up citations   : removed from {sum(bool(r['removed_citations']) for r in done)} answers")
    times = [record["generate_seconds"] for record in done if record["generate_seconds"]]
    if times:
        print(f"Answer time         : {sum(times) / len(times):.1f} s on average")
    models = sorted({record["model"] for record in done})
    print(f"Models that answered: {', '.join(models)}")

    print("\nCorrect score by category:")
    for category in sorted({record["category"] for record in judged}):
        scores = [POINTS[record["correct"]] for record in judged if record["category"] == category]
        print(f"  {category:<15} {sum(scores) / len(scores):.2f}  ({len(scores)} questions)")

    print(f"\n===== Declining: {len(no_answer)} questions with no answer in the data =====")
    for record in no_answer:
        result = "declined" if record["declined"] else "ANSWERED"
        print(f"  {result:<9} {record['question']}")
        if not record["declined"]:
            print(f"            {' '.join(record['answer'].split())[:200]}")
    print(f"Correctly declined: {share(sum(record['declined'] for record in no_answer), len(no_answer))}")

    print("\nProblems to read (not fully correct, not faithful, or wrong citations):")
    for record in judged:
        if record["correct"] != "yes" or record["faithful"] == "no" or record["citations"] == "no":
            print(f"  {record['id']} correct={record['correct']} faithful={record['faithful']} "
                  f"citations={record['citations']} | {record['question']}\n      {record['reason']}")
    for record in answerable:
        if "judge_error" in record:
            print(f"  {record['id']} the judge's reply could not be read: {record['judge_error'][:120]}")
    for record in failed:
        print(f"  {record['id']} no answer, error: {record['error'][:150]}")


# ---------- Run the evaluation ----------

def run_answer_evaluation():
    print("starting the answer evaluation")
    questions = load_questions()
    records = load_finished()
    finished_ids = {record["id"] for record in records}
    remaining = [question for question in questions if question["id"] not in finished_ids]
    print(f"{len(finished_ids)} questions finished in earlier runs, {len(remaining)} to go")
    if not remaining:
        print_report(records)
        return

    assistant = Assistant(log_file=LOG_FILE)
    # The judge uses its own model; with Ollama it is the same local model
    judge_llm = create_llm([GEMINI_JUDGE_MODEL]) if LLM_PROVIDER == "gemini" else assistant.llm

    failures_in_a_row = 0
    try:
        for question in remaining:
            print(f"\n{question['id']}: {question['question']}")
            assistant.history = []   # Each question is asked on its own, not as a follow-up
            try:
                reply = assistant.answer(question["question"])
                record = {
                    "id": question["id"],
                    "category": question["category"],
                    "question": question["question"],
                    "expected": question["answer"],
                    "answerable": bool(question["sources"]),
                    "answer": reply["answer"],
                    "declined": NOT_FOUND in reply["answer"],
                    "cited": [f"[{source['number']}] {source['title']} | {source['url']}"
                              for source in reply["sources"]],
                    "removed_citations": reply["removed_citations"],
                    "low_confidence": reply["low_confidence"],
                    "model": LAST_MODEL["name"],
                    "generate_seconds": reply["timings"].get("generate"),
                }
                if record["answerable"]:
                    record.update(judge(judge_llm, question, reply))
                print(f"  -> {'declined' if record['declined'] else 'answered'}, "
                      f"correct={record.get('correct', '-')}")
                failures_in_a_row = 0
            except Exception as error:   # e.g. the daily API limit
                print(f"  failed: {str(error)[:300]}")
                record = {"id": question["id"], "question": question["question"], "error": str(error)}
                failures_in_a_row += 1
            records.append(record)
            save(records)   # Keep progress, so the next run can continue from here
            if failures_in_a_row == MAX_FAILURES_IN_A_ROW:
                print(f"\nStopping: {MAX_FAILURES_IN_A_ROW} questions failed in a row "
                      "(probably the daily API limit). Run again later to continue.")
                break
    finally:
        assistant.close()

    print_report(records)
    finished = len([record for record in records if "error" not in record])
    print(f"\n{finished}/{len(questions)} questions finished; details saved to {RESULTS_FILE}")
    print("answer evaluation is completed")
