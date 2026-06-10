import json
import os
from flask import Flask, request, Response, stream_with_context
import boto3

app = Flask(__name__)

bedrock = boto3.client("bedrock-runtime", region_name="ap-southeast-1")
MODEL_ID = "global.anthropic.claude-sonnet-4-6-20260217-v1:0"


def cors_headers():
    return {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type",
        "Access-Control-Allow-Methods": "POST,OPTIONS",
    }


@app.route("/", methods=["OPTIONS"])
def options():
    return Response("", status=200, headers=cors_headers())


@app.route("/", methods=["POST"])
def generate_meal_plan():
    data = request.get_json(force=True)
    dietary_preferences = data.get("dietary_preferences", "")

    prompt = (
        f"You are a professional nutritionist and meal planner. "
        f"Create a single day's meal plan for someone with these dietary preferences: {dietary_preferences}.\n\n"
        f"Provide a structured meal plan with:\n"
        f"- **Breakfast**: Meal name and a brief description\n"
        f"- **Lunch**: Meal name and a brief description\n"
        f"- **Dinner**: Meal name and a brief description\n\n"
        f"Also include a quick snack suggestion. "
        f"Make the meals practical, delicious, and nutritionally balanced. "
        f"Use markdown formatting with headers and bullet points."
    )

    # Build messages
    messages = [{"role": "user", "content": []}]

    # Handle file uploads (multimodal)
    file_data = data.get("file_data")
    file_mime = data.get("file_mime")
    if file_data and file_mime:
        if file_mime.startswith("image/"):
            messages[0]["content"].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": file_mime,
                    "data": file_data,
                },
            })
        else:
            messages[0]["content"].append({
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": file_mime,
                    "data": file_data,
                },
            })

    messages[0]["content"].append({"type": "text", "text": prompt})

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": messages,
    })

    def generate():
        response = bedrock.invoke_model_with_response_stream(
            modelId=MODEL_ID,
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        stream = response.get("body")
        for event in stream:
            chunk = event.get("chunk")
            if chunk:
                payload = json.loads(chunk.get("bytes").decode())
                if payload.get("type") == "content_block_delta":
                    delta = payload.get("delta", {})
                    text = delta.get("text", "")
                    if text:
                        yield text

    return Response(
        stream_with_context(generate()),
        content_type="text/plain; charset=utf-8",
        headers=cors_headers(),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
