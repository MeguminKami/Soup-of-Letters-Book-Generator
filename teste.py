import os, json, re
from typing import Dict, Any, List
from jsonschema import Draft7Validator, ValidationError
from openai import OpenAI

# Use Groq via its OpenAI-compatible endpoint
client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

# JSON Schema: exactly 14 Christmas-themed words:
# 4x4 letters, 4x5 letters, 3x5 letters, 3x6 letters
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "four_of_4_letters": {
            "type": "array",
            "items": {"type": "string", "pattern": "^[A-Za-z]{4}$"},
            "minItems": 4, "maxItems": 4, "uniqueItems": True
        },
        "four_of_5_letters": {
            "type": "array",
            "items": {"type": "string", "pattern": "^[A-Za-z]{5}$"},
            "minItems": 4, "maxItems": 4, "uniqueItems": True
        },
        "three_more_of_5_letters": {
            "type": "array",
            "items": {"type": "string", "pattern": "^[A-Za-z]{5}$"},
            "minItems": 3, "maxItems": 3, "uniqueItems": True
        },
        "three_of_6_letters": {
            "type": "array",
            "items": {"type": "string", "pattern": "^[A-Za-z]{6}$"},
            "minItems": 3, "maxItems": 3, "uniqueItems": True
        }
    },
    "required": [
        "four_of_4_letters",
        "four_of_5_letters",
        "three_more_of_5_letters",
        "three_of_6_letters"
    ],
    "additionalProperties": False
}
VALIDATOR = Draft7Validator(SCHEMA)

SYSTEM_RULES = (
    "You are a careful data generator. "
    "Output ONLY valid JSON that matches the given JSON Schema. "
    "No prose, no markdown, no code fences. "
    "All words must be Christmas-themed English words; letters only (A–Z), no spaces or hyphens. "
    "Prefer singular unless the plural is the common term. Keep words widely recognizable."
)

USER_TASK = (
    "Generate Christmas words as per the schema: "
    "4 words of 4 letters, 4 words of 5 letters, 3 more words of 5 letters, and 3 words of 6 letters."
)

def _strip_code_fences(text: str) -> str:
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()

def ask_groq_for_json(model: str = "llama-3.1-8b-instant", max_retries: int = 3) -> Dict[str, Any]:
    """
    Calls Groq Chat Completions, asks for strict JSON, validates via jsonschema,
    and retries with corrective feedback if needed.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Missing GROQ_API_KEY environment variable.")

    schema_str = json.dumps(SCHEMA)
    messages = [
        {"role": "system", "content": f"{SYSTEM_RULES}\nJSON Schema:\n{schema_str}"},
        {"role": "user", "content": USER_TASK},
    ]

    for attempt in range(1, max_retries + 1):
        resp = client.chat.completions.create(
            model=model,
            temperature=0.2,
            messages=messages,
        )
        raw = resp.choices[0].message.content
        text = _strip_code_fences(raw)

        try:
            data = json.loads(text)
            # Validate against JSON Schema
            VALIDATOR.validate(data)

            # Extra hard checks (regex) for safety
            def ok(arr: List[str], n: int) -> bool:
                pat = re.compile(fr"^[A-Za-z]{{{n}}}$")
                return all(isinstance(w, str) and bool(pat.fullmatch(w)) for w in arr)

            assert ok(data["four_of_4_letters"], 4)
            assert ok(data["four_of_5_letters"], 5)
            assert ok(data["three_more_of_5_letters"], 5)
            assert ok(data["three_of_6_letters"], 6)

            return data

        except (json.JSONDecodeError, ValidationError, AssertionError) as e:
            # Feed the error back to the model and try again
            messages.append({"role": "assistant", "content": raw[:4000]})
            messages.append({
                "role": "user",
                "content": (
                    "The JSON above failed to parse/validate. Error:\n"
                    f"{str(e)}\n"
                    "Return corrected JSON ONLY, matching the schema exactly. No extra text."
                ),
            })

    raise RuntimeError("Failed to get valid JSON after max retries.")

def flatten_to_string(data: Dict[str, Any]) -> str:
    # Normalize and combine to a single comma-separated string
    def norm(arr: List[str]) -> List[str]:
        return [w.strip() for w in arr]

    words = (
        norm(data["four_of_4_letters"]) +
        norm(data["four_of_5_letters"]) +
        norm(data["three_more_of_5_letters"]) +
        norm(data["three_of_6_letters"])
    )
    return ", ".join(words)

if __name__ == "__main__":
    result_json = ask_groq_for_json()          # get validated JSON
    result_string = flatten_to_string(result_json)
    print(result_string)                        # <- your single string
    # If you also want to see the JSON, uncomment:
    # print(json.dumps(result_json, indent=2))
