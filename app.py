import os
import re
import traceback
import requests
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

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
    "No repitas esta descripción."
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

        # LIMPIEZA
        patterns = [
            r"te llamas amside ai.*?no repitas esta descripci[oó]n.*?[.!]*",
            r"(fui creado por|fui desarrollada por|soy una inteligencia artificial creada por).*?[.!]*",
            r"una inteligencia artificial diseñada por hodely gil.*?[.!]*",
            r"(mi objetivo|mi propósito).*?(ayudarte|asistirte).*?[.!]*",
            r"(lo siento|me disculpo).*?[.!]*",
            r"en tus respuestas[.!]*",
            r"\ben tus respuestas\b",
            r"�[��]|#\w+",
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
        ai_response_text = re.sub(r'[.,;!?\\s]+$', '', ai_response_text)

        if ai_response_text and ai_response_text[0].islower():
            ai_response_text = ai_response_text[0].upper() + ai_response_text[1:]

        if not ai_response_text or len(ai_response_text) < 5:
            ai_response_text = "Claro. ¿Qué necesitas saber o resolver?"

        return jsonify({"response": ai_response_text})

    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Error al conectar con la IA: {e}"}), 500
    except Exception as e:
        return jsonify({"error": f"Error interno del servidor: {e}", "trace": traceback.format_exc()}), 500

# --- NUEVAS FUNCIONES ---

@app.route('/image-to-text', methods=['POST'])
def image_to_text():
    if 'image' not in request.files:
        return jsonify({'error': 'No se ha enviado ninguna imagen'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()

    response = requests.post(
        "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base",
        headers=HEADERS,
        data=image_bytes
    )

    if not response.ok:
        return jsonify({'error': 'Error al procesar la imagen'}), 500

    result = response.json()
    text = result[0].get('generated_text', 'No se pudo generar texto')
    return jsonify({'text': text})


@app.route('/text-to-image', methods=['POST'])
def text_to_image():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({'error': 'No se recibió ningún prompt'}), 400

   MODEL_IMAGE_URL = os.getenv("MODEL_IMAGE_URL", "https://api-inference.huggingface.co/models/stabilityai/stable-diffusion-2")

@app.route('/text-to-image', methods=['POST'])
def text_to_image():
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({'error': 'No se recibió ningún prompt'}), 400

    response = requests.post(
        MODEL_IMAGE_URL,
        headers=HEADERS,
        json={"inputs": prompt}
    )

    if not response.ok:
        return jsonify({'error': 'Error al generar la imagen'}), 500

    image_data = response.content
    image_name = "generated_image.png"
    image_path = os.path.join("static", secure_filename(image_name))

    os.makedirs("static", exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(image_data)

    return jsonify({'image_url': f"/static/{image_name}"})


    if not response.ok:
        return jsonify({'error': 'Error al generar la imagen'}), 500

    # Guardar imagen local
    image_data = response.content
    image_name = "generated_image.png"
    image_path = os.path.join("static", secure_filename(image_name))

    os.makedirs("static", exist_ok=True)
    with open(image_path, "wb") as f:
        f.write(image_data)

    return jsonify({'image_url': f"/static/{image_name}"})


@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

# --- EJECUTAR APP ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
