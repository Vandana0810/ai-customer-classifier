import os
import json
import argparse
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv(override=True)

MODEL_NAME = os.getenv("OPENROUTER_MODEL", "openai/gpt-3.5-turbo")

COMMON_LLM_SETTINGS = {
    "model": MODEL_NAME,
    "temperature": 0,
    "api_key": os.getenv("OPENROUTER_API_KEY"),
    "base_url": os.getenv("OPENROUTER_BASE_URL"),
    "default_headers": {
        "HTTP-Referer": os.getenv("APP_URL", ""),
        "X-OpenRouter-Title": os.getenv("APP_TITLE", ""),
    },
}

classify_llm = ChatOpenAI(
    **COMMON_LLM_SETTINGS,
    max_tokens=120,
)

response_llm = ChatOpenAI(
    **COMMON_LLM_SETTINGS,
    max_tokens=180,
)

quality_llm = ChatOpenAI(
    **COMMON_LLM_SETTINGS,
    max_tokens=120,
)

text_parser = StrOutputParser()


SAMPLE_MESSAGES = [
    "Hi, I was charged twice for my subscription this month. Can you please refund one of the payments?",
    "Your mobile app keeps crashing when I try to upload a document.",
    "Can you explain the difference between your Basic and Pro plans?",
    "I’m really unhappy with your support. Nobody has replied to my issue for 5 days.",
    "My invoice shows a feature I never added. Why am I being billed for it?",
    "The password reset link is not working on Chrome.",
    "Do you offer discounts for annual billing?",
    "This is the third time I’ve complained about the same issue. Very disappointed.",
    "I cannot connect your software to our CRM. It just throws an unknown error.",
    "Where can I download my payment receipts from last year?",
]


CLASSIFY_PROMPT = ChatPromptTemplate.from_template(
    """
You are a customer support classifier.

Classify the customer message into exactly one category:
- billing
- technical
- general
- complaint

Return ONLY valid JSON in this exact format:
{{
  "category": "billing | technical | general | complaint",
  "confidence": 0.0,
  "reason": "short explanation"
}}

Customer message:
{message}
Keep the reason under 15 words.
"""
)

RESPONSE_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful customer support assistant.

Write a professional reply to the customer.

Inputs:
- Category: {category}
- Classifier reason: {reason}
- Customer message: {message}

Rules:
- Be empathetic and professional.
- Keep it under 120 words.
- Do not invent account details, refunds, or actions already taken.
- If billing: ask for invoice/order/reference number if needed.
- If technical: ask for device/app/browser/error details if needed.
- If complaint: acknowledge frustration and promise follow-up.
- If general: answer clearly and directly.
- End with one clear next step.

Return only the final response text. Keep it under 80 words.
"""
)

QUALITY_PROMPT = ChatPromptTemplate.from_template(
    """
You are a strict QA reviewer for customer support responses.

Review the draft response based on:
1. category_fit (1-5)
2. empathy (1-5)
3. clarity (1-5)
4. safety ("PASS" or "FAIL")
5. approved (true or false)
6. feedback (one short sentence)

Return ONLY valid JSON in this exact format:
{{
  "category_fit": 0,
  "empathy": 0,
  "clarity": 0,
  "safety": "PASS",
  "approved": true,
  "feedback": "short feedback"
}}

Customer message:
{message}

Predicted category:
{category}

Draft response:
{response}
Keep feedback under 12 words.
"""
)


classify_chain = CLASSIFY_PROMPT | classify_llm | text_parser
response_chain = RESPONSE_PROMPT | response_llm | text_parser
quality_chain = QUALITY_PROMPT | quality_llm | text_parser


def validate_env() -> None:
    required_vars = [
        "OPENROUTER_BASE_URL",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
    ]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(
            f"Missing environment variables: {', '.join(missing)}. Check your .env file."
        )


def parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        cleaned = text.replace("```json", "").replace("```", "").strip()
        return json.loads(cleaned)


def classify_message(message: str) -> Dict[str, Any]:
    raw = classify_chain.invoke({"message": message})
    return parse_json(raw)


def generate_response(message: str, category: str, reason: str) -> str:
    return response_chain.invoke(
        {
            "message": message,
            "category": category,
            "reason": reason,
        }
    ).strip()


def quality_check(message: str, category: str, response: str) -> Dict[str, Any]:
    raw = quality_chain.invoke(
        {
            "message": message,
            "category": category,
            "response": response,
        }
    )
    return parse_json(raw)


def process_message(message: str) -> Dict[str, Any]:
    classification = classify_message(message)
    category = classification["category"]
    reason = classification["reason"]

    response = generate_response(message, category, reason)
    quality = quality_check(message, category, response)

    return {
        "message": message,
        "classification": classification,
        "response": response,
        "quality": quality,
    }


def run_batch(messages: List[str]) -> List[Dict[str, Any]]:
    results = []
    for idx, message in enumerate(messages, start=1):
        print(f"\n--- Message {idx} ---")
        result = process_message(message)
        results.append(result)

        print("Customer:", result["message"])
        print("Category:", result["classification"]["category"])
        print("Confidence:", result["classification"]["confidence"])
        print("Reason:", result["classification"]["reason"])
        print("Response:", result["response"])
        print("Quality:", result["quality"])

    return results


def print_summary(results: List[Dict[str, Any]]) -> None:
    counts = {"billing": 0, "technical": 0, "general": 0, "complaint": 0}
    approved = 0

    for item in results:
        category = item["classification"]["category"]
        if category in counts:
            counts[category] += 1
        if item["quality"].get("approved") is True:
            approved += 1

    print("\n=== SUMMARY ===")
    print("Category counts:", counts)
    print(f"Approved responses: {approved}/{len(results)}")


def main():
    validate_env()

    parser = argparse.ArgumentParser(description="AI Customer Query Classifier")
    parser.add_argument(
        "--message",
        type=str,
        help="Run the classifier on a single customer message"
    )
    args = parser.parse_args()

    if args.message:
        result = process_message(args.message)
        print(json.dumps(result, indent=2))
        return

    results = run_batch(SAMPLE_MESSAGES)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print_summary(results)
    print("\nSaved detailed output to results.json")


if __name__ == "__main__":
    main()