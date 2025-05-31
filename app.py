import os
from flask import Flask, request, jsonify
import requests
from flask_cors import CORS

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

    # --- NUEVO: Construir la cadena de prompt con el formato específico de Zephyr ---
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
    full_prompt_string = "\n".join(formatted_prompt_parts)
    # Algunos modelos pueden requerir un token de inicio de secuencia al principio de la cadena
    # Zephyr a menudo espera <s> al inicio de una nueva "conversación" o "interacción".
    # Lo añadimos aquí al inicio de la cadena completa.
    full_prompt_string = "<s>" + full_prompt_string 
    # --- FIN NUEVO ---


    # Configuración del payload para la API de inferencia de Hugging Face
    # Ahora, 'inputs' es la cadena de prompt formateada
    payload = {
        "inputs": full_prompt_string, # <--- CAMBIO AQUÍ
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            # Se han eliminado temporalmente "do_sample", "top_p" y "repetition_penalty"
            # Si con estos cambios funciona, puedes intentar añadirlos de nuevo uno a uno
            # para identificar cuál podría estar causando el problema.
        },
        # Cuando envías una cadena pre-formateada, a veces return_full_text puede ser True
        # o incluso omitirse. Si el modelo devuelve todo, ajustaremos la limpieza.
        # Por ahora, vamos a intentar con False, esperando que la API de HF haga la limpieza.
        "return_full_text": False 
    }
    
    try:
        # Realizar la solicitud POST a la API de inferencia de Hugging Face
        hf_data = query_huggingface_model(payload) # Usamos la función auxiliar

        # Validar la estructura de la respuesta de Hugging Face
        # Para modelos de texto a texto, la respuesta suele ser un array con un diccionario que contiene 'generated_text'
        if not hf_data or not isinstance(hf_data, list) or not hf_data[0].get('generated_text'):
            # --- DEBUGGING: Registra la respuesta inesperada ---
            print(f"DEBUG: Respuesta inesperada de Hugging Face: {hf_data}")
            # --- FIN DEBUGGING ---
            return jsonify({"error": "Respuesta inesperada de Hugging Face API.", "hf_response": hf_data}), 500

        # Extraer el texto generado por la IA
        ai_response_text = hf_data[0]['generated_text']

        # --- Limpieza del texto generado ---
        # Si return_full_text es False, la API de HF ya debería haber limpiado el prompt.
        # Pero a veces los modelos aún pueden repetir partes o añadir tokens.
        # Quitamos cualquier token de finalización o de turno que el modelo pueda haber generado.
        ai_response_text = ai_response_text.split('</s>')[0].strip() 
        ai_response_text = ai_response_text.replace('<|assistant|>', '').strip()
        ai_response_text = ai_response_text.replace('<|user|>', '').strip()
        ai_response_text = ai_response_text.replace('<|system|>', '').strip()
        ai_response_text = ai_response_text.replace('<s>', '').strip() # Eliminar token de inicio de secuencia
        ai_response_text = ai_response_text.replace('</s>', '').strip() # Eliminar token de fin de secuencia
        # --- FIN Limpieza ---

        # Devolver la respuesta de la IA al frontend en formato JSON
        return jsonify({"response": ai_response_text})

    except requests.exceptions.RequestException as e:
        # Captura errores relacionados con la conexión HTTP (red, DNS, etc.)
        # El raise_for_status() ya manejará los 4xx/5xx específicos de HF
        print(f"Error al conectar con Hugging Face API: {e}")
        return jsonify({"error": f"Error al conectar con la IA: {e}. Por favor, inténtalo de nuevo más tarde."}), 500
    except Exception as e:
        # Captura cualquier otro error inesperado en el servidor
        print(f"Error interno del servidor: {e}")
        return jsonify({"error": f"Ocurrió un error inesperado en el servidor: {e}. Por favor, contacta al soporte."}), 500

# Punto de entrada para ejecutar la aplicación Flask
if __name__ == '__main__':
    # Cuando ejecutas `python app.py`, esto se ejecuta.
    # `debug=True` es útil para el desarrollo local (recarga el servidor automáticamente, muestra errores detallados).
    # Para producción (como en Render), un servidor WSGI como Gunicorn se encargará de esto.
    # Render proporcionará el puerto a través de la variable de entorno 'PORT'.
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
