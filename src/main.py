import os
import gradio as gr
from src.utils import query_api, initialize_llm, split_text, get_ligthcast_access_token
import logging

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    level=logging.INFO,
)


HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/papluca/xlm-roberta-base-language-detection"
LIGHTCAST_API_URL = "https://emsiservices.com/skills/versions/latest/extract/trace"
HUGGINGFACE_TOKEN = os.environ.get("HF_API_KEY")
LIGHTCAST_TOKEN = os.environ.get("LIGHTCAST_TOKEN")


class SkillMatch:

    def __init__(self):

        self.skills = None
        self.trace = None
        self.normalizedText = None

    def lang_detection(self, text):
        language = query_api(
            {"inputs": split_text(text)}, HUGGINGFACE_API_URL, HUGGINGFACE_TOKEN
        )[0][0].get("label")

        return language

    def skills_detection(self, text):
        skill_data = query_api(
            {"text": text, "confidenceThreshold": 0.6, "includeNormalizedText": True},
            LIGHTCAST_API_URL,
            get_ligthcast_access_token(),
            params={"language": "en"},
        )

        return skill_data

    def pipeline(self, text):

        language = self.lang_detection(text)
        if language != "en":
            llm = initialize_llm()
            text = llm.invoke(f"Translate this document into English: '{text}'")

        skill_data = self.skills_detection(text)
        logging.info(type(skill_data))

        self.skills = [
            skill.get("skill").get("name")
            for skill in skill_data.get("data").get("skills")
        ]
        self.trace = [
            x.get("surfaceForm").get("value")
            for x in skill_data.get("data").get("trace")
        ]
        self.normalizedText = skill_data.get("data").get("normalizedText")

    def ner(self, text):

        self.pipeline(text)

        return {
            "text": self.normalizedText,
            "entities": [
                {
                    "end": x["sourceEnd"],
                    "start": x["sourceStart"],
                    "word": x["value"],
                    "entity": "skill",
                }
                for x in [x.get("surfaceForm") for x in self.trace]
                if not x["value"] == "r"
            ],
        }
