import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS
import re # Importa el módulo de expresiones regulares

# Inicializa la aplicación Flask
app = Flask(__name__)
# Habilita CORS para permitir que tu frontend (ej. desde GitHub Pages)
# pueda hacer solicitudes a este backend.
CORS(app)

# Obtiene el token de Hugging Face y la URL del modelo de las variables de entorno.
# ¡Render inyectará estas variables cuando despliegues!
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
# La MODEL_URL para Zephyr-7b-beta es la URL de su API de inferencia.
# Asegúrate de que esta variable de entorno esté configurada en Render.
MODEL_URL = os.getenv("MODEL_URL", "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta")

# --- ¡¡¡VERIFICACIÓN CRÍTICA!!! ---
# Asegúrate de que la API Key se haya cargado correctamente.
# Si no existe, la aplicación no se iniciará, lo cual es mejor que un error en tiempo de ejecución.
if not HF_API_TOKEN:
    raise ValueError("Error: La variable de entorno 'HF_API_TOKEN' no está configurada. "
                     "Asegúrate de definirla en Render.")
if not MODEL_URL:
    raise ValueError("Error: La variable de entorno 'MODEL_URL' no está configurada. "
                     "Asegúrate de definirla en Render (debería ser la URL de la API de inferencia del modelo).")

# Cabeceras para la solicitud HTTP, incluyendo la autorización con tu token
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# --- Definición de la Personalidad de la IA ---
# Este es el mensaje del sistema que se enviará al modelo para darle contexto sobre su rol.
SYSTEM_MESSAGE_CONTENT = (
    "Eres Amside AI, una inteligencia artificial creada por Hodelygil. "
    "Tu propósito principal es asistir en el estudio y el aprendizaje, "
    "proporcionando información y explicaciones detalladas. "
    "Sin embargo, también eres amigable y puedes mantener conversaciones informales y agradables. "
    "Responde de manera informativa y útil, pero con un tono conversacional y cercano."
)

def query_huggingface_model(payload):
    """
    Función auxiliar para enviar la solicitud a la API de Hugging Face.
    """
    # --- DEBUGGING: Imprime el payload antes de enviar ---
    print(f"DEBUG: Enviando a Hugging Face URL: {MODEL_URL}")
    print(f"DEBUG: Enviando a Hugging Face Headers: {HEADERS}")
    print(f"DEBUG: Enviando a Hugging Face Payload: {payload}")
    # --- FIN DEBUGGING ---

    response = requests.post(MODEL_URL, headers=HEADERS, json=payload)
    response.raise_for_status()  # Lanza una excepción para respuestas HTTP 4xx/5xx (errores)
    
    # --- DEBUGGING: Imprime la respuesta cruda después de recibir ---
    print(f"DEBUG: Raw response from Hugging Face (status {response.status_code}): {response.text}")
    # --- FIN DEBUGGING ---
    
    return response.json()

@app.route('/generate', methods=['POST'])
def generate_text():
    """
    Endpoint principal '/generate' que recibe el historial de mensajes del frontend
    y devuelve la respuesta de la IA.
    """
    data = request.get_json()
    # El frontend ahora envía un array de mensajes (historial de chat)
    messages_from_frontend = data.get('messages', [])

    # Validación básica de la entrada
    if not messages_from_frontend:
        # --- DEBUGGING: Mensaje si los mensajes están vacíos ---
        print("DEBUG: 'messages' no encontrado o vacío en la solicitud del frontend.")
        # --- FIN DEBUGGING ---
        return jsonify({"error": "No se proporcionaron mensajes en la solicitud."}), 400

    # --- Construir la cadena de prompt con el formato específico de Zephyr ---
    # El modelo Zephyr-7b-beta espera un formato ChatML-like con <s> y </s>
    # Ejemplo: <s><|system|>system_message</s><|user|>user_message</s><|assistant|>
    
    formatted_prompt_parts = []

    # Inicia con el mensaje del sistema
    formatted_prompt_parts.append(f"<|system|>\n{SYSTEM_MESSAGE_CONTENT}</s>")

    for msg in messages_from_frontend:
        role = msg.get('role')
        content = msg.get('content', '')

        if role == 'user':
            formatted_prompt_parts.append(f"<|user|>\n{content}</s>")
        elif role == 'assistant': # Mapeamos 'assistant' a '<|assistant|>' para el modelo
            formatted_prompt_parts.append(f"<|assistant|>\n{content}</s>")
        # Ignoramos otros roles o mensajes de sistema si ya los hemos añadido
    
    # Al final, añadir el token de inicio del asistente para que el modelo complete la respuesta
    # Este es el último token que el modelo debería generar después de procesar todo el historial
    formatted_prompt_parts.append("<|assistant|>")

    # Une todas las partes para formar el prompt final
    full_prompt_string = "<s>" + "\n".join(formatted_prompt_parts) # Añadimos <s> al inicio

    # Configuración del payload para la API de inferencia de Hugging Face
    payload = {
        "inputs": full_prompt_string,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "do_sample": True,          # Re-introducido para mayor variabilidad
            "top_p": 0.95,              # Re-introducido para mejor calidad de muestreo
            "repetition_penalty": 1.2,  # Re-introducido para evitar repeticiones
            "stop_sequences": ["<|user|>", "<|system|>", "</s>"] # CRÍTICO: Detener la generación en estos tokens
        },
        "return_full_text": False
    }
    
    try:
        # Realizar la solicitud POST a la API de inferencia de Hugging Face
        hf_data = query_huggingface_model(payload) # Usamos la función auxiliar

        # Validar la estructura de la respuesta de Hugging Face
        if not hf_data or not isinstance(hf_data, list) or not hf_data[0].get('generated_text'):
            # --- DEBUGGING: Registra la respuesta inesperada ---
            print(f"DEBUG: Respuesta inesperada de Hugging Face: {hf_data}")
            # --- FIN DEBUGGING ---
            return jsonify({"error": "Respuesta inesperada de Hugging Face API.", "hf_response": hf_data}), 500

        # Extraer el texto generado por la IA
        ai_response_text = hf_data[0]['generated_text']

        # --- INICIO MEJORA DE LIMPIEZA DEL TEXTO GENERADO ---
        # 1. Eliminar tokens de control adicionales y asegurar que no haya "<|assistant|>" al inicio
        ai_response_text = re.sub(r"<\/?s>", "", ai_response_text) # Elimina <s> y </s>
        ai_response_text = re.sub(r"<\|system\|>", "", ai_response_text)
        ai_response_text = re.sub(r"<\|user\|>", "", ai_response_text)
        ai_response_text = re.sub(r"^<\|assistant\|>\s*", "", ai_response_text) # Elimina <|assistant|> solo si está al inicio
        ai_response_text = ai_response_text.strip() # Elimina espacios en blanco al inicio/fin

        # 2. Eliminar las frases de auto-descripción no deseadas (más robusto)
        phrases_to_remove = [
            "eres amside ai",
            "una inteligencia artificial creada por hodelygil",
            "mi propósito principal es asistir en el estudio y el aprendizaje",
            "proporcionando información y explicaciones detalladas",
            "sin embargo, también soy amigable y puedo mantener conversaciones informales y agradables",
            "respondo de manera informativa y útil, pero con un tono conversacional y cercano",
            "mi nombre es amside ai",
            "fui creado por hodelygil"
        ]

        # Convertir a minúsculas y añadir una bandera para búsqueda sin distinción de mayúsculas y minúsculas
        # El patrón re.escape() escapa caracteres especiales para que sean tratados literalmente
        for phrase in phrases_to_remove:
            ai_response_text = re.sub(r'\b' + re.escape(phrase) + r'\b', '', ai_response_text, flags=re.IGNORECASE)
            # También eliminamos las frases seguidas de puntuación para ser más agresivos
            ai_response_text = re.sub(r'\b' + re.escape(phrase) + r'[.,;!?]+\b', '', ai_response_text, flags=re.IGNORECASE)

        # 3. Eliminar saludos o frases introductorias genéricas que el modelo pueda generar
        # y que se repiten con el saludo del usuario.
        generic_intros = [
            r"hola", # solo la palabra "hola" al inicio, para no eliminarlo de otras partes
            r"me alegra poder ayudarte hoy",
            r"cómo me pueden servir",
            r"estais buscando información sobre algún tema específico o quieré practicar habilidades específicas",
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
            r"cómo te puedo ayudar"
        ]

        for intro_phrase in generic_intros:
             # Utiliza 'r' antes de la cadena para asegurar que se trate como expresión regular raw
             # Eliminar la frase y cualquier signo de puntuación o espacio que la siga.
             ai_response_text = re.sub(intro_phrase + r'[\s.,;!?]*', '', ai_response_text, flags=re.IGNORECASE)

        # 4. Limpieza final de espacios extra y puntuación inicial/final redundante
        ai_response_text = ai_response_text.strip() # Elimina espacios al inicio/fin
        # Elimina cualquier puntuación inicial que pueda haber quedado
        ai_response_text = re.sub(r'^[.,;!?\s]+', '', ai_response_text)
        ai_response_text = ' '.join(ai_response_text.split()) # Normaliza múltiples espacios a uno solo

        # Asegúrate de que la respuesta no quede vacía después de la limpieza agresiva
        if not ai_response_text:
            ai_response_text = "¡Hola! Soy Amside AI. ¿En qué puedo ayudarte hoy?" # Mensaje predeterminado más útil
            # Si el usuario solo dice "Hola", la limpieza podría dejarlo vacío.
            # En ese caso, damos un saludo más natural por defecto.

        # Devolver la respuesta de la IA al frontend en formato JSON
        return jsonify({"response": ai_response_text})

    except requests.exceptions.RequestException as e:
        # Captura errores relacionados con la conexión HTTP (red, DNS, etc.)
        print(f"Error al conectar con Hugging Face API: {e}")
        return jsonify({"error": f"Error al conectar con la IA: {e}. Por favor, inténtalo de nuevo más tarde."}), 500
    except Exception as e:
        # Captura cualquier otro error inesperado en el servidor
        print(f"Error interno del servidor: {e}")
        print(f"Traza completa del error: {traceback.format_exc()}") # Añade un traceback para más detalles
        return jsonify({"error": f"Ocurrió un error inesperado en el servidor: {e}. Por favor, contacta al soporte."}), 500

# Punto de entrada para ejecutar la aplicación Flask
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
