"""STEP 1: the simplest thing that could work.

One LangChain agent. You give it one file, it gives you the invoice fields. Nothing else:
no queue, no database, no checks, no screen. It is the first version anyone would write.

Run:  python ladder/step1_one_agent.py data/generated/files/0002_bundle_scan.pdf
"""
import base64
import io
import os
import sys
import time

import pypdfium2 as pdfium
from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
from pypdf import PdfReader

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

def _get_model():
    provider = os.environ.get("LLM_PROVIDER", "gemini")
    model_name = os.environ.get("MODEL_SMALL")
    if provider == "openai" or os.environ.get("OPENAI_API_KEY"):
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model_name or "google/gemini-2.5-flash-lite",
            api_key=os.environ.get("OPENAI_API_KEY") or None,
            base_url=os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE") or None,
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=model_name or "gemini-3.5-flash-lite")


# The "form" the agent must fill: only what a person would read off an invoice.
class Invoice(BaseModel):
    supplier_name: str = ""
    invoice_number: str = ""
    po_number: str = ""
    total: float = 0


# The agent: a model, no tools, and a form to fill. That is all "an agent" is at this step.
agent = create_agent(
    _get_model(),
    tools=[],
    system_prompt="You read business papers. Fill the invoice form from the paper. Copy numbers exactly.",
    response_format=ToolStrategy(Invoice),
)


def read_file(path: str) -> list:
    """Turn a file into something the model can read: text if the PDF has text, a picture if it does not."""
    if path.endswith(".json"):
        return [{"type": "text", "text": open(path).read()}]
    if path.endswith((".jpg", ".png")):
        return [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(open(path, "rb").read()).decode()}}]
    text = "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    if len(text.strip()) > 40:
        return [{"type": "text", "text": text}]
    parts = []  # a scan: send every page as a picture
    for page in pdfium.PdfDocument(path):
        buf = io.BytesIO()
        page.render(scale=2).to_pil().convert("L").save(buf, format="PNG")
        parts.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()}})
    return parts


def read_invoice(path: str) -> Invoice:
    result = agent.invoke({"messages": [HumanMessage(content=read_file(path))]})
    return result["structured_response"]


if __name__ == "__main__":
    for path in sys.argv[1:]:
        started = time.time()
        print(f"{path.split('/')[-1]:24} {read_invoice(path)}   ({time.time() - started:.1f} s)")
