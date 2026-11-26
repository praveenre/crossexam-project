"""STEP 2: three small fixes. Still one script. Still no queue, no database server, no web screen.

What step 1 got wrong, and the plain fix for each:
  1. Every paper was forced into the invoice form  ->  ask first: what kind of paper is this? Then use the matching form.
  2. Running it twice paid twice                   ->  fingerprint each file (SHA-256) and skip files we have seen.
  3. A crash lost everything                       ->  save each result in a file as soon as it exists.

Run:  python ladder/step2_three_small_fixes.py data/generated/files/0002_*.pdf
"""
import hashlib
import json
import os
import sys
from typing import Literal

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from step1_one_agent import Invoice, read_file, _get_model  # step 1 stays as it was: we only add to it

MODEL = _get_model()
SAVED = os.environ.get("STEP2_RESULTS", "ladder/results.jsonl")   # fix 3: results live in a file, one line per paper


# Fix 1: one small form per kind of paper.
class Kind(BaseModel):
    kind: Literal["invoice", "shipping", "certificate", "other"]   # shipping = packing list, lorry receipt or e-way bill


class Shipping(BaseModel):
    invoice_number: str = ""
    vehicle_number: str = ""


class Certificate(BaseModel):
    batch_number: str = ""
    expiry_date: str = ""
    result: str = ""


FORMS = {"invoice": Invoice, "shipping": Shipping, "certificate": Certificate}


def ask(form, instruction: str, content: list) -> BaseModel:
    agent = create_agent(MODEL, tools=[], system_prompt=instruction, response_format=ToolStrategy(form))
    return agent.invoke({"messages": [HumanMessage(content=content)]})["structured_response"]


def already_done() -> dict:
    if not os.path.exists(SAVED):
        return {}
    return {row["sha256"]: row for row in map(json.loads, open(SAVED))}


def read_paper(path: str) -> dict:
    data = open(path, "rb").read()
    fingerprint = hashlib.sha256(data).hexdigest()          # fix 2: the same bytes are never read twice
    done = already_done()
    if fingerprint in done:
        return {**done[fingerprint], "skipped": True}
    content = read_file(path)
    kind = ask(Kind, "What kind of paper is this? Look at the whole file.", content).kind   # fix 1
    fields = ask(FORMS[kind], "Fill the form from the paper. Copy values exactly. Leave a field empty if it is not there.", content).model_dump() if kind in FORMS else {}
    row = {"file": os.path.basename(path), "sha256": fingerprint, "kind": kind, "fields": fields}
    with open(SAVED, "a") as out:                            # fix 3: saved now, not at the end
        out.write(json.dumps(row) + "\n")
    return row


if __name__ == "__main__":
    for path in sys.argv[1:]:
        row = read_paper(path)
        print(f"{row['file']:24} {row['kind']:12} {row['fields']}" + ("   (skipped: seen before)" if row.get("skipped") else ""))
