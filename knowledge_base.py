# knowledge_base.py

"""
Knowledge Base adapter untuk Firebase RTDB dan file lokal.

Prinsip:
- Firebase RTDB /knowledge_base menjadi sumber utama.
- knowledge_base.json menjadi fallback jika Firebase tidak tersedia.
- Mendukung struktur JSON:
    {
        "knowledge_base": {
            "metadata": {...},
            "environment": {...},
            ...
        }
    }
- Retrieval dilakukan berdasarkan tujuan/topik.
- Tidak ada Gemini, LSTM, atau kontrol aktuator di file ini.
"""

import json
from pathlib import Path

from firebase_service import get_data


class KnowledgeBaseError(RuntimeError):
    pass


# ============================================================
# PEMETAAN TOPIK
# ============================================================

TOPIC_MAP = {
    "environment": [
        "environment",
        "soil",
    ],

    "soil": [
        "soil",
        "environment",
    ],

    "irrigation": [
        "irrigation",
        "environment",
        "soil",
    ],

    "fertilization": [
        "fertilization",
        "soil",
        "planting",
    ],

    "planting": [
        "planting",
        "seed_and_nursery",
        "land_preparation",
        "location_and_planting",
        "soil",
    ],

    "maintenance": [
        "maintenance",
        "irrigation",
        "fertilization",
        "environment",
    ],

    "harvest": [
        "harvest",
        "planting",
        "maintenance",
        "environment",
    ],

    "post_harvest": [
        "post_harvest",
        "harvest",
    ],

    "general": [
        "environment",
        "fertilization",
        "harvest",
        "irrigation",
        "land_preparation",
        "location_and_planting",
        "maintenance",
        "planting",
        "post_harvest",
        "seed_and_nursery",
        "soil",
    ],
}


class KnowledgeBase:

    def __init__(
        self,
        firebase_path="knowledge_base",
        local_file=None,
        firebase_loader=get_data,
    ):
        self.firebase_path = firebase_path

        # Default:
        # knowledge_base.json berada satu folder
        # dengan knowledge_base.py
        if local_file is None:
            local_file = (
                Path(__file__).resolve().parent
                / "knowledge_base.json"
            )

        self.local_file = Path(local_file)

        self.firebase_loader = firebase_loader

        # Cache
        self._data = None

        # Sumber KB yang sedang digunakan
        self._source = None

    # ========================================================
    # NORMALISASI STRUKTUR KB
    # ========================================================

    @staticmethod
    def _normalize_data(data):
        """
        Menormalisasi struktur Knowledge Base.

        Mendukung dua bentuk:

        Bentuk 1:
        {
            "knowledge_base": {
                "metadata": {...},
                "environment": {...}
            }
        }

        Bentuk 2:
        {
            "metadata": {...},
            "environment": {...}
        }
        """

        if not isinstance(data, dict):
            return None

        # Jika memiliki wrapper "knowledge_base"
        if (
            "knowledge_base" in data
            and isinstance(
                data["knowledge_base"],
                dict,
            )
        ):
            return data["knowledge_base"]

        # Jika langsung berisi topic
        return data

    # ========================================================
    # LOAD KNOWLEDGE BASE
    # ========================================================

    def load(self, force_refresh=False):

        # Gunakan cache
        if (
            self._data is not None
            and not force_refresh
        ):
            return self._data

        data = None

        # ----------------------------------------------------
        # 1. COBA FIREBASE
        # ----------------------------------------------------

        try:

            data = self.firebase_loader(
                self.firebase_path
            )

        except Exception as exc:

            print(
                f"KB Firebase error: {exc}"
            )

        normalized_data = self._normalize_data(
            data
        )

        if (
            isinstance(
                normalized_data,
                dict,
            )
            and normalized_data
        ):

            self._data = normalized_data
            self._source = "firebase"

            print(
                "Knowledge Base loaded dari Firebase."
            )

            return self._data

        # ----------------------------------------------------
        # 2. FALLBACK FILE LOKAL
        # ----------------------------------------------------

        print(
            "Firebase KB tidak tersedia."
        )

        print(
            "Mencoba knowledge_base.json..."
        )

        if not self.local_file.exists():

            raise KnowledgeBaseError(
                "Knowledge Base tidak tersedia. "
                "Firebase gagal dan file lokal "
                f"tidak ditemukan:\n"
                f"{self.local_file}"
            )

        try:

            local_raw = json.loads(
                self.local_file.read_text(
                    encoding="utf-8"
                )
            )

        except json.JSONDecodeError as exc:

            raise KnowledgeBaseError(
                "Format knowledge_base.json "
                f"tidak valid:\n{exc}"
            ) from exc

        except OSError as exc:

            raise KnowledgeBaseError(
                "Gagal membaca knowledge_base.json:\n"
                f"{exc}"
            ) from exc

        local_data = self._normalize_data(
            local_raw
        )

        if (
            not isinstance(
                local_data,
                dict,
            )
            or not local_data
        ):

            raise KnowledgeBaseError(
                "knowledge_base.json "
                "tidak memiliki struktur Knowledge Base "
                "yang valid."
            )

        self._data = local_data
        self._source = "local"

        print(
            "Knowledge Base loaded dari file lokal:"
        )

        print(
            self.local_file
        )

        return self._data

    # ========================================================
    # REFRESH
    # ========================================================

    def refresh(self):

        return self.load(
            force_refresh=True
        )

    # ========================================================
    # GET TOPIC
    # ========================================================

    def get_topic(
        self,
        topic,
        force_refresh=False,
    ):

        data = self.load(
            force_refresh=force_refresh
        )

        return data.get(
            topic,
            {},
        )

    # ========================================================
    # GET MULTIPLE TOPICS
    # ========================================================

    def get_topics(
        self,
        topics,
        force_refresh=False,
    ):

        data = self.load(
            force_refresh=force_refresh
        )

        result = {}

        for topic in topics:

            if topic in data:

                result[topic] = data[topic]

        return result

    # ========================================================
    # RETRIEVAL BERDASARKAN TUJUAN
    # ========================================================

    def retrieve(
        self,
        purpose="general",
        force_refresh=False,
    ):

        purpose = (
            purpose or "general"
        ).lower()

        topics = TOPIC_MAP.get(
            purpose,
            TOPIC_MAP["general"],
        )

        return self.get_topics(
            topics,
            force_refresh=force_refresh,
        )

    # ========================================================
    # METADATA
    # ========================================================

    def get_metadata(self):

        return self.get_topic(
            "metadata"
        )

    # ========================================================
    # SUMBER DOKUMEN
    # ========================================================

    def get_source(self):

        metadata = self.get_metadata()

        if not isinstance(
            metadata,
            dict,
        ):
            return None

        return (
            metadata.get(
                "source_title"
            )
            or metadata.get(
                "source_document"
            )
        )

    # ========================================================
    # SUMBER KB YANG SEDANG DIGUNAKAN
    # ========================================================

    def get_loaded_source(self):

        return self._source

    # ========================================================
    # BUILD CONTEXT
    # ========================================================

    def build_context(
        self,
        purpose="general",
        force_refresh=False,
    ):

        return {
            "purpose": purpose,

            "source": self.get_source(),

            "loaded_from": self.get_loaded_source(),

            "knowledge": self.retrieve(
                purpose=purpose,
                force_refresh=force_refresh,
            ),
        }

    # ========================================================
    # EVALUASI KELEMBAPAN TANAH
    # ========================================================

    def evaluate_soil_moisture(
        self,
        soil_moisture,
    ):

        value = float(
            soil_moisture
        )

        # Default sistem
        ideal_min = 60.0
        ideal_max = 80.0

        # ----------------------------------------------------
        # Ambil dari irrigation
        # irrigation -> soil_moisture -> ideal
        # ----------------------------------------------------

        irrigation = self.get_topic(
            "irrigation"
        )

        if isinstance(
            irrigation,
            dict,
        ):

            soil_moisture_data = (
                irrigation.get(
                    "soil_moisture"
                )
            )

            if isinstance(
                soil_moisture_data,
                dict,
            ):

                ideal = (
                    soil_moisture_data.get(
                        "ideal"
                    )
                )

                if isinstance(
                    ideal,
                    dict,
                ):

                    if (
                        "min" in ideal
                        and "max" in ideal
                    ):

                        ideal_min = float(
                            ideal["min"]
                        )

                        ideal_max = float(
                            ideal["max"]
                        )

        # ----------------------------------------------------
        # Evaluasi
        # ----------------------------------------------------

        if value < ideal_min:

            status = "LOW"

        elif value > ideal_max:

            status = "HIGH"

        else:

            status = "IDEAL"

        return {
            "value": value,
            "ideal_min": ideal_min,
            "ideal_max": ideal_max,
            "status": status,
        }

    # ========================================================
    # EVALUASI pH
    # ========================================================

    def evaluate_ph(
        self,
        ph,
    ):

        value = float(ph)

        # Default
        min_ph = 5.5
        max_ph = 6.8

        # Struktur aktual:
        #
        # soil
        # └── ph
        #     └── tolerable
        #
        soil = self.get_topic(
            "soil"
        )

        if isinstance(
            soil,
            dict,
        ):

            ph_data = soil.get(
                "ph"
            )

            if isinstance(
                ph_data,
                dict,
            ):

                tolerable = (
                    ph_data.get(
                        "tolerable"
                    )
                )

                if isinstance(
                    tolerable,
                    dict,
                ):

                    if (
                        "min" in tolerable
                        and "max" in tolerable
                    ):

                        min_ph = float(
                            tolerable["min"]
                        )

                        max_ph = float(
                            tolerable["max"]
                        )

        # ----------------------------------------------------
        # Evaluasi
        # ----------------------------------------------------

        if value < min_ph:

            status = "LOW"

        elif value > max_ph:

            status = "HIGH"

        else:

            status = "IDEAL"

        return {
            "value": value,
            "min": min_ph,
            "max": max_ph,
            "status": status,
        }

    # ========================================================
    # KONVERSI KE PROMPT GEMINI
    # ========================================================

    @staticmethod
    def to_prompt_context(
        knowledge,
    ):

        return json.dumps(
            knowledge,
            ensure_ascii=False,
            indent=2,
        )


# ============================================================
# DEFAULT KNOWLEDGE BASE
# ============================================================

_default_kb = KnowledgeBase()


# ============================================================
# HELPER
# ============================================================

def get_knowledge_base(
    force_refresh=False,
):

    return _default_kb.load(
        force_refresh=force_refresh
    )
