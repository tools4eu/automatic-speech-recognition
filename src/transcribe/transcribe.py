import numpy as np
import torch
from faster_whisper import WhisperModel
import logging
import string


logging.basicConfig()
logging.getLogger("faster_whisper").setLevel(logging.DEBUG)

model_size = "large-v2"

languages = {
    "English": "en",
    "Chinese": "zh",
    "German": "de",
    "Spanish": "es",
    "Russian": "ru",
    "Korean": "ko",
    "French": "fr",
    "Japanese": "ja",
    "Portuguese": "pt",
    "Turkish": "tr",
    "Polish": "pl",
    "Catalan": "ca",
    "Dutch": "nl",
    "Arabic": "ar",
    "Swedish": "sv",
    "Italian": "it",
    "Indonesian": "id",
    "Hindi": "hi",
    "Finnish": "fi",
    "Vietnamese": "vi",
    "Hebrew": "iw",
    "Ukrainian": "uk",
    "Greek": "el",
    "Malay": "ms",
    "Czech": "cs",
    "Romanian": "ro",
    "Danish": "da",
    "Hungarian": "hu",
    "Tamil": "ta",
    "Norwegian": "no",
    "Thai": "th",
    "Urdu": "ur",
    "Croatian": "hr",
    "Bulgarian": "bg",
    "Lithuanian": "lt",
    "Latin": "la",
    "Maori": "mi",
    "Malayalam": "ml",
    "Welsh": "cy",
    "Slovak": "sk",
    "Telugu": "te",
    "Persian": "fa",
    "Latvian": "lv",
    "Bengali": "bn",
    "Serbian": "sr",
    "Azerbaijani": "az",
    "Slovenian": "sl",
    "Kannada": "kn",
    "Estonian": "et",
    "Macedonian": "mk",
    "Breton": "br",
    "Basque": "eu",
    "Icelandic": "is",
    "Armenian": "hy",
    "Nepali": "ne",
    "Mongolian": "mn",
    "Bosnian": "bs",
    "Kazakh": "kk",
    "Albanian": "sq",
    "Swahili": "sw",
    "Galician": "gl",
    "Marathi": "mr",
    "Punjabi": "pa",
    "Sinhala": "si",
    "Khmer": "km",
    "Shona": "sn",
    "Yoruba": "yo",
    "Somali": "so",
    "Afrikaans": "af",
    "Occitan": "oc",
    "Georgian": "ka",
    "Belarusian": "be",
    "Tajik": "tg",
    "Sindhi": "sd",
    "Gujarati": "gu",
    "Amharic": "am",
    "Yiddish": "yi",
    "Lao": "lo",
    "Uzbek": "uz",
    "Faroese": "fo",
    "Haitian creole": "ht",
    "Pashto": "ps",
    "Turkmen": "tk",
    "Nynorsk": "nn",
    "Maltese": "mt",
    "Sanskrit": "sa",
    "Luxembourgish": "lb",
    "Myanmar": "my",
    "Tibetan": "bo",
    "Tagalog": "tl",
    "Malagasy": "mg",
    "Assamese": "as",
    "Tatar": "tt",
    "Hawaiian": "haw",
    "Lingala": "ln",
    "Hausa": "ha",
    "Bashkir": "ba",
    "Javanese": "jw",
    "Sundanese": "su",
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# def transcriber(input: str, language: str, translate: bool, progress) -> dict:
def transcriber(input: str, language: str, translate: bool) -> dict:
    """Transcribes the audio using the OpenAI Whisper model.
    Args:
        input: file path to the audio file in any format
        language: name of the language in which the audio is recorded
        translate: boolean indicator to enable immediate translation
    Returns: transcription and segment-timestamps.
    """

    model = WhisperModel(model_size, device=device)

    

    language = languages.get(language, None)
    if translate:
        task = "translate"
    else:
        task = "transcribe"

    # segments, _ = model.transcribe(audio=input, vad_filter=True, task=task, progress=progress)
    segments, _ = model.transcribe(audio=input, vad_filter=True, task=task)
    segments = list(segments)  # The transcription will actually run here.

    # Process and modify the segment texts
    modified_texts = []
    for i, segment in enumerate(segments):
        text = segment.text.strip()
        if text[-1] not in string.punctuation:
                text += '.'
        modified_texts.append(text)

    # Create a list of dictionaries in the required format
    tr_segments = [
        {
            "start": segment.start,
            "end": segment.end,
            "text": modified_text
        }
        for segment, modified_text in zip(segments, modified_texts)
    ]

    concatenated_text = "\n".join(modified_texts)

    tr = {
        "text": concatenated_text,
        "segments": tr_segments
    }

    return tr
