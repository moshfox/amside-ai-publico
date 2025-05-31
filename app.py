import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import re
import traceback

app = Flask(__name__)
CORS(app)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# ¡CAMBIO A MISTRAL-7B-INSTRUCT-V0.2!
MODEL_URL = os.getenv("MODEL_URL", "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2")

if not HF_API_TOKEN:
    raise ValueError("Error: La variable de entorno 'HF_API_TOKEN' no está configurada. "
                     "Asegúrate de definirla en Render.")
if not MODEL_URL:
    raise ValueError("Error: La variable de entorno 'MODEL_URL' no está configurada. "
                     "Asegúrate de definirla en Render (debería ser la URL de la API de inferencia del modelo).")

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- Definición de la Personalidad de la IA ---
# Para Mistral, el SYSTEM_MESSAGE_CONTENT suele ir al inicio del primer mensaje de usuario
# o como parte del mensaje del usuario con un formato específico.
# Mantendremos el mismo system message por ahora, pero lo integraremos de forma diferente.
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

    # Fragmentos específicos de Zephyr (puede que no aparezcan con Mistral, pero los dejamos por si acaso)
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
    r"🚀",
    r"#AIforStudents", # El Mistral también puede generar hashtags
    r"#LearnwithAmsideAI",
    r"#HappyLearning",
    r"🤝", # Otros emoticonos que podrían aparecer
    r"😊"
]


def query_huggingface_model(payload):
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

    # --- CAMBIO IMPORTANTE EN LA CONSTRUCCIÓN DEL PROMPT PARA MISTRAL ---
    # Mistral-7B-Instruct-v0.2 usa un formato de chat específico:
    # <s>[INST] {user_message} [/INST] {assistant_response}</s>[INST] {next_user_message} [/INST]
    # El system message se suele integrar en el primer [INST]
    
    formatted_prompt_parts = []
    
    # El primer mensaje (que es del usuario) debe incluir el SYSTEM_MESSAGE_CONTENT
    # Esto es crucial para Mistral. Lo ponemos como una "pregunta" del usuario.
    # Si el historial está vacío, o solo tiene el último mensaje del usuario, construimos el inicio.
    if not messages_from_frontend or messages_from_frontend[0].get('role') == 'user':
        # Asume que el primer mensaje que procesaremos es el del usuario.
        # Si ya hay un historial, lo reconstruimos.
        
        # El primer turno debe empezar con <s>
        # Si el primer mensaje del historial es de usuario, le añadimos el system message
        # Si no hay historial, o el primer es assistant, el primer user message es el que empieza la conversación.
        
        # Construye el historial para Mistral
        # <s>[INST] System message + user_message_1 [/INST] assistant_response_1 </s> [INST] user_message_2 [/INST]
        
        # Empezamos con el "inst" del primer usuario y su mensaje.
        # El system message lo ponemos como parte de la instrucción inicial.
        
        # Formato de Mistral
        # <s>[INST] {prompt} [/INST]
        # {completion}
        # Para chats, es: <s>[INST] User Message [/INST] Assistant Response </s> [INST] User Message 2 [/INST]
        
        # Si es el inicio de la conversación y el primer mensaje es del usuario
        # O si queremos que el system message SIEMPRE esté al inicio de la conversación completa para el modelo.
        # Lo más común es ponerlo en el primer [INST] del usuario.
        
        # Reconstruimos el historial completo para Mistral
        current_prompt = "<s>"
        for i, msg in enumerate(messages_from_frontend):
            role = msg.get('role')
            content = msg.get('content', '').strip() # Strip para limpiar espacios iniciales/finales
            
            if role == 'user':
                if i == 0: # Si es el primer mensaje de usuario, añadir el system message
                    current_prompt += f"[INST] {SYSTEM_MESSAGE_CONTENT}\n\n{content} [/INST]"
                else:
                    current_prompt += f"[INST] {content} [/INST]"
            elif role == 'assistant':
                current_prompt += f" {content}</s>" # La respuesta del asistente no tiene tags, y termina con </s>
                
        # El prompt final no termina con </s> porque esperamos la respuesta del asistente.
        full_prompt_string = current_prompt

    # --- FIN CAMBIO IMPORTANTE EN LA CONSTRUCCIÓN DEL PROMPT PARA MISTRAL ---

    payload = {
        "inputs": full_prompt_string,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            # Stop sequences para Mistral. Importante: [/INST] para evitar que el modelo se meta en el turno del usuario.
            # </s> para detenerlo si termina su frase.
            "stop_sequences": ["</s>", "[INST]", "[/INST]"] 
        },
        "return_full_text": False
    }
    
    try:
        hf_data = query_huggingface_model(payload)

        if not hf_data or not isinstance(hf_data, list) or not hf_data[0].get('generated_text'):
            print(f"DEBUG: Respuesta inesperada de Hugging Face: {hf_data}")
            return jsonify({"error": "Respuesta inesperada de Hugging Face API.", "hf_response": hf_data}), 500

        ai_response_text = hf_data[0]['generated_text']

        # --- INICIO MEJORA DE LIMPIEZA DEL TEXTO GENERADO (ADAPTADO A MISTRAL) ---

        # 1. Eliminar tokens de control específicos de Mistral si aparecen (y otros que no deberían)
        ai_response_text = re.sub(r"<s>", "", ai_response_text) # Inicio de secuencia
        ai_response_text = re.sub(r"</s>", "", ai_response_text) # Fin de secuencia
        ai_response_text = re.sub(r"\[INST\]", "", ai_response_text) # Tokens de instrucción
        ai_response_text = re.sub(r"\[/INST\]", "", ai_response_text) # Tokens de fin de instrucción
        ai_response_text = ai_response_text.strip() # Eliminar espacios en blanco al inicio/fin después de los tokens

        # 2. Eliminar las frases de auto-descripción y saludos genéricos muy agresivamente
        for phrase_pattern in PHRASES_TO_REMOVE:
            # Crear un patrón regex que sea más flexible con espacios y puntuación alrededor de la frase
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
            ai_response_text = "¡Hola! Soy Amside AI, tu asistente de estudio. ¿En qué puedo ayudarte hoy?"

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
