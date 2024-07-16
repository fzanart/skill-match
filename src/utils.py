import os
import json
import requests
from langchain_google_genai import GoogleGenerativeAI
from langchain_text_splitters import TokenTextSplitter
from google.generativeai.types.safety_types import HarmBlockThreshold, HarmCategory
from transformers import XLMRobertaTokenizerFast


def query_api(payload, api_url, token, method="POST", params=None):
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    response = requests.request(
        method, api_url, json=payload, headers=headers, params=params
    )
    return response.json()


def get_ligthcast_access_token():

    url = "https://auth.emsicloud.com/connect/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    CLIENT_ID = os.environ.get("CLIENT_ID")
    SECRET = os.environ.get("SECRET")
    payload = f"client_id={CLIENT_ID}&client_secret={SECRET}&grant_type=client_credentials&scope=emsi_open"
    response = requests.request("POST", url, data=payload, headers=headers)

    return json.loads(response.text).get("access_token")


def initialize_llm():
    return GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        temperature=0,
        safety_settings={
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        },
    )


def split_text(text, chunk_size=int(512 * 0.9)):
    # return first 450 token of text for language detection
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(
        "papluca/xlm-roberta-base-language-detection"
    )
    text_splitter = TokenTextSplitter.from_huggingface_tokenizer(
        tokenizer, chunk_size=chunk_size, chunk_overlap=0
    )
    texts = text_splitter.split_text(text)
    return texts[0]
