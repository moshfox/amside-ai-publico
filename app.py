import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import re
import traceback

app = Flask(__name__)
CORS(app)

HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# Asumo que ahora apuntas a Mixtral-8x7B-Instruct-v0.1 en Render
MODEL_URL = os.getenv("MODEL_URL", "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1")

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
    r"soy amside ai", # Añadido
    r"la inteligencia artificial friendy y estudiosa", # Añadido
    r"me encanta ayudar en el proceso de aprender cosas nuevas y divertidas", # Añadido
    r"mi papel principal es ayudarte a estudiar y a aprender", # Añadido
    r"ofreciendo información y explicaciones detalladas", # Añadido
    r"estoy programado para ser útil y eficaz", # Añadido
    r"pero también quiero que tu experiencia sea divertida y genial", # Añadido

    # Fragmentos específicos de Zephyr/Mistral que pueden persistir
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
    r"#AIforStudents",
    r"#LearnwithAmsideAI",
    r"#HappyLearning",
    r"🤝",
    r"😊",
    r"holaa!", # Añadido
    r"estoy genial, muchísimas gracias por preguntarlo.", # Añadido
    r"encantad@ de estar aquí contigo compartiendo este momento y list@ para responder a cualquier consulta o curiosidad que se te presente.", # Añadido
    r"además, siempre procuro mantener un ambiente agradable y distendido durante nuestra charla.", # Añadido
    r"así que no dudes en plantearme temas formales o incluso más desenfadados; me adaptaré sin problemas para hacer de esta experiencia algo realmente entretenido y provechoso.", # Añadido
    r"y tú, ¿estás disfrutando del tiempo?", # Añadido
    r"hola", # Añadido para eliminar 'Hola Hola' inicial
    r"¿hoy?" # Añadido para eliminar fragmento "¿ hoy?"
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

    # --- Construcción del Prompt para Mistral/Mixtral ---
    # <s>[INST] {System Message} + User Message 1 [/INST] Assistant Response 1</s>[INST] User Message 2 [/INST]
    
    current_prompt = "<s>"
    last_assistant_response = "" # Para almacenar la última respuesta del asistente para la limpieza

    for i, msg in enumerate(messages_from_frontend):
        role = msg.get('role')
        content = msg.get('content', '').strip()
        
        if role == 'user':
            if i == 0: # Si es el primer mensaje de usuario, añadir el system message
                current_prompt += f"[INST] {SYSTEM_MESSAGE_CONTENT}\n\n{content} [/INST]"
            else:
                current_prompt += f"[INST] {content} [/INST]"
        elif role == 'assistant':
            current_prompt += f" {content}</s>"
            last_assistant_response = content # Almacenar la respuesta del asistente
            
    full_prompt_string = current_prompt

    payload = {
        "inputs": full_prompt_string,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "do_sample": True,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            # Stop sequences para Mistral/Mixtral. Importante: [/INST] para evitar que el modelo se meta en el turno del usuario.
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

        # --- INICIO MEJORA DE LIMPIEZA DEL TEXTO GENERADO (ULTRA-AGRESIVA y CONSCIENTE DEL HISTORIAL) ---

        # 1. Eliminar tokens de control y secuencias de ChatML
        ai_response_text = re.sub(r"<\/?s>", "", ai_response_text)
        ai_response_text = re.sub(r"<\|system\|>", "", ai_response_text)
        ai_response_text = re.sub(r"<\|user\|>", "", ai_response_text)
        ai_response_text = re.sub(r"<\|assistant\|>", "", ai_response_text)
        ai_response_text = re.sub(r"\[INST\]", "", ai_response_text)
        ai_response_text = re.sub(r"\[/INST\]", "", ai_response_text)
        ai_response_text = ai_response_text.strip() # Eliminar espacios en blanco al inicio/fin

        # 2. **CRÍTICO: Eliminar la repetición de la ÚLTIMA RESPUESTA del asistente si aparece al inicio**
        # Normalizar para comparación (quitar espacios extra, bajar a minúsculas)
        clean_last_assistant_response = ' '.join(last_assistant_response.lower().split())
        clean_ai_response_text = ' '.join(ai_response_text.lower().split())
        
        if clean_last_assistant_response and clean_ai_response_text.startswith(clean_last_assistant_response):
            # Si la nueva respuesta empieza con la anterior, la cortamos.
            # Usamos el len de la versión original (no la limpia) para el corte,
            # pero necesitamos encontrar la posición de la coincidencia.
            
            # Intentar encontrar la posición de la última respuesta en la nueva respuesta
            # Esto es tricky por las variaciones de espacios y puntuación.
            # Una forma simple es reemplazar la primera ocurrencia si se encuentra.
            
            # Buscar la última respuesta (insensible a mayúsculas/minúsculas y espacios extra)
            # y eliminarla del inicio de la nueva respuesta.
            pattern_last_response = r'^\s*' + re.escape(last_assistant_response) + r'[\s.,;!?]*'
            ai_response_text = re.sub(pattern_last_response, ' ', ai_response_text, flags=re.IGNORECASE).strip()


        # 3. Eliminar las frases de auto-descripción y saludos genéricos muy agresivamente
        for phrase_pattern in PHRASES_TO_REMOVE:
            pattern = r'\s*' + re.escape(phrase_pattern) + r'[\s.,;!?]*'
            ai_response_text = re.sub(pattern, ' ', ai_response_text, flags=re.IGNORECASE)
            
        # 4. Limpieza final de espacios extra, comas/puntuación al inicio y normalización.
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
