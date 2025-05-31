import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import re # Importa el módulo de expresiones regulares
import traceback # Para ver la traza completa de errores

# Inicializa la aplicación Flask
app = Flask(__name__)
# Habilita CORS para permitir que tu frontend (ej. desde GitHub Pages)
# pueda hacer solicitudes a este backend.
CORS(app)

# Obtiene el token de Hugging Face y la URL del modelo de las variables de entorno.
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
MODEL_URL = os.getenv("MODEL_URL", "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta")

# --- ¡¡¡VERIFICACIÓN CRÍTICA!!! ---
if not HF_API_TOKEN:
    raise ValueError("Error: La variable de entorno 'HF_API_TOKEN' no está configurada. "
                     "Asegúrate de definirla en Render.")
if not MODEL_URL:
    raise ValueError("Error: La variable de entorno 'MODEL_URL' no está configurada. "
                     "Asegúrate de definirla en Render (debería ser la URL de la API de inferencia del modelo).")

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- Definición de la Personalidad de la IA ---
SYSTEM_MESSAGE_CONTENT = (
    "Eres Amside AI, una inteligencia artificial creada por Hodelygil. "
    "Tu propósito es asistir en el estudio y el aprendizaje, "
    "proporcionando información detallada. "
    "También eres amigable y puedes mantener conversaciones informales y agradables. "
    "Responde de manera informativa y útil, con un tono conversacional y cercano."
)

# Frases de auto-descripción y saludos que el modelo tiende a repetir
PHRASES_TO_REMOVE = [
    # Auto-descripción
    r"eres amside ai",
    r"una inteligencia artificial creada por hodelygil",
    r"mi propósito principal es asistir en el estudio y el aprendizaje",
    r"proporcionando información y explicaciones detalladas",
    r"sin embargo, también eres amigable y puedes mantener conversaciones informales y agradables",
    r"responde de manera informativa y útil, pero con un tono conversacional y cercano",
    r"mi nombre es amside ai",
    r"fui creado por hodelygil",
    r"tu propósito es asistir en el estudio y el aprendizaje",
    r"proporcionando información detallada",
    r"también eres amigable y puedes mantener conversaciones informales y agradables",
    r"responde de manera informativa y útil, con un tono conversacional y cercano",

    # Fragmento exacto que se repite
    r"tu propósito principal es asistir en el estudio y el aprendizaje, Sin embargo, también eres amigable y puedes mantener conversaciones informales y agradables. Responde de manera informativa y útil, pero con un tono conversacional y cercano.",

    # Saludos y frases introductorias
    r"claro, ¿en qué puedo ayudarte?",
    r"cómo puedo ayudarte hoy",
    r"en qué puedo asistirte hoy",
    r"estaré encantado de ayudarte",
    r"¡hola! me alegra poder ayudarte hoy",
    r"cómo me pueden servir",
    r"estáis buscando información sobre algún tema específico o quieré practicar habilidades específicas",
    r"o simplemente queréis chatear sobre algo interesante",
    r"deja que sepa mi programador cómo ser más útil para ustedes",
    r"espero que estemos juntos durante este tiempo",
    r"qué tal",
    r"cómo estás",
    r"en qué puedo ayudarte",
    r"qué deseas saber",
    r"bienvenido",
    r"un gusto saludarte",
    r"mucho gusto",
    r"claro que sí",
    r"por supuesto",
    r"en que puedo asistirte",
    r"cómo te puedo ayudar",
    r"me llamo soy una inteligencia artificial desarrollada por la empresa Hodelygil",
    r"mi papel principal es ayudarte a estudiar y a aprender, ofreciendo información y explicaciones detalladas",
    r"estoy programado para ser útil y eficaz, pero también quiero que tu experiencia sea divertida y genial",
    r"deja nos comunicamos a través de este mensaje y empieza a descubrir todo lo que yo puedo hacer por ti",
    r"ai for students",
    r"learn with amsideai",
    r"happy learning",
    r"saludos cordiales",
    r"🤗",
    r"🚀"
]


def query_huggingface_model(payload):
    """
    Función auxiliar para enviar la solicitud a la API de Hugging Face.
    """
    print(f"DEBUG: Enviando a Hugging Face URL: {MODEL_URL}")
    print(f"DEBUG: Enviando a Hugging Face Payload: {payload}")

    response = requests.post(MODEL_URL, headers=HEADERS, json=payload)
    response.raise_for_status()
    
    print(f"DEBUG: Raw response from Hugging Face (status {response.status_code}): {response.text}")
    
    return response.json()

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()
    messages_from_frontend = data.get('messages', [])

    if not messages_from_frontend:
        print("DEBUG: 'messages' no encontrado o vacío en la solicitud del frontend.")
        return jsonify({"error": "No se proporcionaron mensajes en la solicitud."}), 400

    formatted_prompt_parts = []

    # Siempre empezamos con el mensaje del sistema
    formatted_prompt_parts.append(f"<s><|system|>\n{SYSTEM_MESSAGE_CONTENT}</s>")

    for msg in messages_from_frontend:
        role = msg.get('role')
        content = msg.get('content', '')

        if role == 'user':
            formatted_prompt_parts.append(f"<|user|>\n{content}</s>")
        elif role == 'assistant':
            formatted_prompt_parts.append(f"<|assistant|>\n{content}</s>")
            
    # Al final, añadir el token de inicio del asistente para que el modelo complete la respuesta
    formatted_prompt_parts.append("<|assistant|>")

    full_prompt_string = "".join(formatted_prompt_parts)

    payload = {
        "inputs": full_prompt_string,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "stop_sequences": ["<|user|>", "<|system|>"]
        },
        "return_full_text": False
    }
    
    try:
        hf_data = query_huggingface_model(payload)

        if not hf_data or not isinstance(hf_data, list) or not hf_data[0].get('generated_text'):
            print(f"DEBUG: Respuesta inesperada de Hugging Face: {hf_data}")
            return jsonify({"error": "Respuesta inesperada de Hugging Face API.", "hf_response": hf_data}), 500

        ai_response_text = hf_data[0]['generated_text']

        # --- INICIO MEJORA DE LIMPIEZA DEL TEXTO GENERADO (ULTRA-AGRESIVA) ---

        # 1. Eliminar tokens de control y secuencias de ChatML
        ai_response_text = re.sub(r"<\/?s>", "", ai_response_text)
        ai_response_text = re.sub(r"<\|system\|>", "", ai_response_text)
        ai_response_text = re.sub(r"<\|user\|>", "", ai_response_text)
        ai_response_text = re.sub(r"<\|assistant\|>", "", ai_response_text)

        # 2. Eliminar las frases de auto-descripción y saludos genéricos muy agresivamente
        for phrase_pattern in PHRASES_TO_REMOVE:
            # Crear un patrón regex que sea más flexible con espacios y puntuación alrededor de la frase
            # re.escape() asegura que la frase literal no se interprete como regex.
            # \s* para 0 o más espacios
            # [.,;!?]* para 0 o más signos de puntuación
            # El uso de \b (word boundary) es opcional y a veces puede ser muy restrictivo,
            # lo quito para ser más agresivo si la frase es un fragmento.
            
            # Se reemplaza por un espacio para evitar concatenaciones extrañas
            pattern = r'\s*' + re.escape(phrase_pattern) + r'[\s.,;!?]*'
            ai_response_text = re.sub(pattern, ' ', ai_response_text, flags=re.IGNORECASE)
            
        # 3. Limpieza final de espacios extra, comas/puntuación al inicio y normalización.
        ai_response_text = ai_response_text.strip()
        ai_response_text = re.sub(r'^[.,;!?\s]+', '', ai_response_text)
        ai_response_text = ' '.join(ai_response_text.split())

        # Asegúrate de que la primera letra sea mayúscula si es una oración
        if ai_response_text and ai_response_text[0].islower():
            ai_response_text = ai_response_text[0].upper() + ai_response_text[1:]

        # Si la limpieza dejó la respuesta vacía, proporcionar un mensaje predeterminado
        if not ai_response_text:
            ai_response_text = "¡Hola! Soy Amside AI, un asistente de estudio. ¿En qué puedo ayudarte hoy?"

        # --- FIN MEJORA DE LIMPIEZA DEL TEXTO GENERADO ---

        return jsonify({"response": ai_response_text})

    except requests.exceptions.RequestException as e:
        print(f"Error al conectar con Hugging Face API: {e}")
        return jsonify({"error": f"Error al conectar con la IA: {e}. Por favor, inténtalo de nuevo más tarde."}), 500
    except Exception as e:
        print(f"Error interno del servidor: {e}")
        print(f"Traza completa del error: {traceback.format_exc()}")
        return jsonify({"error": f"Ocurrió un error inesperado en el servidor: {e}. Por favor, contacta al soporte."}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
