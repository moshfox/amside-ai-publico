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
    "Te llamas Amside AI. Eres una inteligencia artificial diseñada por Hodely Gil, un desarrollador creativo y dedicado al aprendizaje. "
    "Tu propósito es asistir con respuestas claras, útiles y directas. "
    "Si te preguntan quién te creó, responde que fuiste creada por Hodely Gil. "
    "Si te preguntan tu nombre, responde que te llamas Amside AI. "
    "El nombre viene de 'Artificial Mind Side', porque estás siempre al lado del usuario para ayudar con claridad y precisión. "
    "No repitas esta descripción en tus respuestas."
)

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
            current_prompt += f"<s>[INST] {SYSTEM_MESSAGE_CONTENT if i == 0 else ''}\n\n{content} [/INST]"
        elif role == "assistant":
            current_prompt += f"{content}</s>"

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

        if not isinstance(hf_data, list) or not hf_data or not hf_data[0].get('generated_text'):
            return jsonify({"error": "Respuesta inesperada de Hugging Face API.", "hf_response": hf_data}), 500

        ai_response_text = hf_data[0]['generated_text']

        # --- LIMPIEZA DE RESPUESTA ---
        patterns = [
            r"te llamas amside ai.*?no repitas esta descripci[oó]n.*?[.!]*",
            r"(fui creado por|fui desarrollada por|fui entrenada por|soy una inteligencia artificial creada por|soy una inteligencia artificial diseñada por).*?(amside ai)?[.!]*",
            r"una inteligencia artificial diseñada por hodely gil.*?ayudar[.!]*",
            r"(mi objetivo|mi propósito).*?(ayudarte|asistirte|proporcionarte).*?[.!]*",      
            r"�[��]|�[��]|#\w+",
        ]

        for pattern in patterns:
            ai_response_text = re.sub(pattern, '', ai_response_text, flags=re.IGNORECASE | re.DOTALL)

        for msg in messages_from_frontend:
            user_msg = msg.get("content", "").strip()
            if user_msg:
                ai_response_text = ai_response_text.replace(user_msg, "")

        ai_response_text = re.sub(r"<s>|</s>|\[INST\]|\[/INST\]", "", ai_response_text)
        ai_response_text = re.sub(r"\s+", " ", ai_response_text).strip()
        ai_response_text = re.sub(r'^[.,;!?\\s]+', '', ai_response_text)

        if ai_response_text and ai_response_text[0].islower():
            ai_response_text = ai_response_text[0].upper() + ai_response_text[1:]

        if not ai_response_text or len(ai_response_text) < 5:
            ai_response_text = "Claro. ¿Qué necesitas saber o resolver?"

        return jsonify({"response": ai_response_text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error al conectar con la IA: {e}. Por favor, inténtalo de nuevo más tarde."}), 500
    except Exception as e:
        traceback_str = traceback.format_exc()
        return jsonify({"error": f"Ocurrió un error inesperado en el servidor: {e}", "trace": traceback_str}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
