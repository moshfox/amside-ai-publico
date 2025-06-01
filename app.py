
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
import os
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
            r"(lo siento|lamento|me disculpo|mis disculpas).*?[.!]*",
            r"(podemos empezar de nuevo|puedo ayudarte de nuevo).*?[.!]*",
            r"no fue mi intención.*?[.!]*",
            r"en tus respuestas[.!]*",
            r"\ben tus respuestas\b",
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
        ai_response_text = re.sub(r'[.,;!?\\s]+$', '', ai_response_text)

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
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000))
