# intent_classifier.py


def classify_intent(text):

    text = text.lower().strip()

    # =====================
    # COMMAND
    # =====================

    commands = {

        "status":
            "status",

        "cuaca":
            "cuaca",

        "pompa":
            "pompa",

        "tanaman":
            "tanaman",

    }

    if text in commands:

        return {

            "intent":
                "command",

            "command":
                commands[text]
        }

    # =====================
    # ANALYSIS
    # =====================

    analysis_keywords = [

        "analisis",

        "analisa",

        "evaluasi",

        "cek kondisi",

        "kondisi lahan",

        "kondisi tanaman",

        "bagaimana kondisi",

        "status lahan"
    ]

    if any(
        k in text
        for k in analysis_keywords
    ):

        return {

            "intent":
                "analysis"
        }

    
    # =====================
    # DEFAULT CHAT
    # =====================

    return {

        "intent":
            "chat"
    }