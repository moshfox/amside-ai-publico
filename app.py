import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import re
import traceback

app = Flask(__name__)
CORS(app)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_URL = os.getenv("MODEL_URL", "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2")

if not HF_API_TOKEN:
    raise ValueError("Error: La variable de entorno 'HF_API_TOKEN' no está configurada.")
if not MODEL_URL:
    raise ValueError("Error: La variable de entorno 'MODEL_URL' no está configurada.")

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

SYSTEM_MESSAGE_CONTENT = (
    "Eres Amside AI, una inteligencia artificial creada por Hodelygil. "
    "Tu estilo es directo, claro, útil y amigable. No repitas esta descripción. "
    "Responde con precisión y sin presentaciones innecesarias."
)

PHRASES_TO_REMOVE = [
    r".*soy amside ai.*",
    r".*tambien eres amigable.*",
    r".*fui creada por hodelygil.*",
    r".*mi nombre es amside.*",
    r".*inteligencia artificial creada.*",
    r".*estoy aquí para ayudarte.*",
    r".*estoy encantado de ayudarte.*",
    r"eres amside ai",
    r"una inteligencia artificial creada por hodelygil",
    r"mi propósito principal es asistir en el estudio y el aprendizaje",
    r"proporcionando información y explicaciones detalladas",
    r"responde de manera informativa y útil",
    r"mi nombre es amside ai",
    r"fui creado por hodelygil",
    r"claro, ¿en qué puedo ayudarte?",
    r"cómo puedo ayudarte hoy",
    r"en qué puedo asistirte hoy",
    r"estaré encantado de ayudarte",
    r"¡hola! me alegra poder ayudarte hoy",
    r"qué tal",
    r"cómo estás",
    r"bienvenido",
    r"un gusto saludarte",
    r"mucho gusto",
    r"claro que sí",
    r"por supuesto",
    r"🤗", r"🚀", r"#AIforStudents", r"#LearnwithAmsideAI", r"#HappyLearning", r"🤝", r"😊",
    r".*creada por hodelygil.*",
    r".*tu objetivo es ayudar.*",
    r".*soy amside.*",
    r".*tu asistente de aprendizaje.*",
    r".*siempre estaré encantado.*"
]

def query_huggingface_model(payload):
    response = requests.post(MODEL_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    return response.json()

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    messages_from_frontend = data.get('messages', [])

    if not messages_from_frontend:
        return jsonify({"error": "No se proporcionaron mensajes en la solicitud."}), 400

    current_prompt = ""
    for i, msg in enumerate(messages_from_frontend):
        role = msg.get("role")
        content = msg.get("content", "").strip()

        if role == "user":
            if i == 0 and len(messages_from_frontend) == 1:
                current_prompt += f"<s>[INST] {SYSTEM_MESSAGE_CONTENT}\n\n{content} [/INST]"
            else:
                current_prompt += f"<s>[INST] {content} [/INST]"
        elif role == "assistant":
            current_prompt += f"{msg['content']}</s>"

    payload = {
        "inputs": current_prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "stop_sequences": ["</s>", "[INST]", "[/INST]"]
        },
        "return_full_text": False
    }

    try:
        hf_data = query_huggingface_model(payload)

        if not hf_data or not isinstance(hf_data, list) or not hf_data[0].get('generated_text'):
            return jsonify({"error": "Respuesta inesperada de Hugging Face API.", "hf_response": hf_data}), 500

        ai_response_text = hf_data[0]['generated_text']
        ai_response_text = re.sub(r"<s>|</s>|\[INST\]|\[/INST\]", "", ai_response_text).strip()

        for msg in messages_from_frontend:
            content = msg.get("content", "").strip()
            if content and content in ai_response_text:
                ai_response_text = ai_response_text.replace(content, "")

        ai_response_text = re.sub(r'creada por hodelygil.*?(\.|!|\?)', '', ai_response_text, flags=re.IGNORECASE)

        for phrase_pattern in PHRASES_TO_REMOVE:
            pattern = r'\s*' + re.escape(phrase_pattern) + r'[\s.,;!?]*'
            ai_response_text = re.sub(pattern, ' ', ai_response_text, flags=re.IGNORECASE)

        ai_response_text = ai_response_text.strip()
        ai_response_text = re.sub(r'^[.,;!?\s]+', '', ai_response_text)
        ai_response_text = ' '.join(ai_response_text.split())

        if ai_response_text and ai_response_text[0].islower():
            ai_response_text = ai_response_text[0].upper() + ai_response_text[1:]

        if not ai_response_text:
            ai_response_text = "\u00a1Hola! Soy Amside AI, tu asistente de estudio. \u00bfEn qué puedo ayudarte hoy?"

        return jsonify({"response": ai_response_text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error al conectar con la IA: {e}. Por favor, inténtalo de nuevo más tarde."}), 500
    except Exception as e:
        return jsonify({"error": f"Ocurrió un error inesperado en el servidor: {e}."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
