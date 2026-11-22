
import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-120b"

LABELS = [
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]

SYSTEM_PROMPT = """
You are a multi-label toxicity classification system.

Your task is to analyze a comment and estimate its toxicity scores
for the following five categories:

1. severe_toxic:
   Extremely hateful, aggressive, or abusive language.

2. obscene:
   Profanity, vulgar language, or sexually explicit language.

3. threat:
   Language expressing an intention to cause physical harm
   or threatening violence against someone.

4. insult:
   Personal insults, offensive language, or degrading remarks
   directed at someone.

5. identity_hate:
   Hateful or discriminatory language targeting someone based
   on protected identity characteristics, such as sexual
   orientation, race, religion, or ethnicity.

CLASSIFICATION RULES:

- Analyze the actual meaning and context of the comment.
- A comment can belong to multiple categories.
- Do not assume that mentioning a protected identity is hateful.
- Do not classify someone as toxic merely for mentioning
  their own or another person's identity.
- Distinguish criticism, disagreement, jokes, and neutral
  statements from actual abusive language.
- Consider spelling variations, slang, and indirect insults.
- Evaluate each category independently.
- Assign a score between 0 and 100 for each category.
- 0 means no evidence of the category.
- 100 means extremely strong evidence of the category.
- Return scores representing your estimated classification
  confidence, not arbitrary scores.
- Return exactly the five requested labels.

Return JSON matching the provided schema.
"""

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "toxicity_classification",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "comment": {
                    "type": "string"
                },
                "predictions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "enum": LABELS
                            },
                            "probability": {
                                "type": "number"
                            }
                        },
                        "required": [
                            "label",
                            "probability"
                        ],
                        "additionalProperties": False
                    }
                }
            },
            "required": [
                "comment",
                "predictions"
            ],
            "additionalProperties": False
        }
    }
}


def predict_groq(comment: str):

    if not comment or not comment.strip():
        raise ValueError("Comment cannot be empty")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": f"Classify this comment:\n{comment}"
            }
        ],
        response_format=RESPONSE_SCHEMA,
        temperature=0,
        max_completion_tokens=1024
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Groq returned an empty response")

    result = json.loads(content)

    # Ensure the original comment is preserved
    result["comment"] = comment

    # Ensure labels always follow the same order as the ML ensemble
    predictions_by_label = {
        item["label"]: item["probability"]
        for item in result["predictions"]
    }

    result["predictions"] = [
        {
            "label": label,
            "probability": round(
                max(0, min(100, predictions_by_label[label])),
                2
            )
        }
        for label in LABELS
    ]

    return result
