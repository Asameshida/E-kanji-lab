from flask import Flask, request, jsonify, send_from_directory
import anthropic
import os
import re
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="static")

USE_MOCK = os.environ.get("MOCK_KANJI_API", "").lower() in ("1", "true", "yes")
_anthropic_client = None


def get_anthropic_client():
    global _anthropic_client
    if _anthropic_client is None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY is not set (or use MOCK_KANJI_API=1 for dummy mode)")
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


def mock_suggestions(name: str, traits: list, style: str | None):
    return [
        {
            "kanji": "蒼詩龍",
            "reading": "そうしりゅう",
            "overall_meaning": f"A bold spirit who wields poetry like a dragon commands the sky — bestowed upon {name}.",
            "kanji_details": [
                {"kanji": "蒼", "reading": "Sou", "gloss": "vast blue-green; the wild expanse of sky and sea"},
                {"kanji": "詩", "reading": "Shi", "gloss": "poetry; the craft of giving form to feeling"},
                {"kanji": "龍", "reading": "Ryu", "gloss": "dragon; commanding power and mythic strength"},
            ],
            "trait_resonance": "蒼 evokes free-spirited boldness, 詩 honours the artistic voice, 龍 commands the strength within.",
        },
        {
            "kanji": "靜智花",
            "reading": "せいちか",
            "overall_meaning": f"The calm mind of a scholar whose wisdom blooms with quiet grace — bestowed upon {name}.",
            "kanji_details": [
                {"kanji": "靜", "reading": "Sei", "gloss": "stillness; calm and composed clarity of mind"},
                {"kanji": "智", "reading": "Chi", "gloss": "wisdom; intelligence and discernment"},
                {"kanji": "花", "reading": "Ka", "gloss": "flower; elegant beauty and gentle vitality"},
            ],
            "trait_resonance": "靜 reflects the calm, 智 honours the intelligence, 花 captures the elegant beauty of this name.",
        },
        {
            "kanji": "玄誠嵐",
            "reading": "げんせいらん",
            "overall_meaning": f"Deep mystery and unwavering sincerity, fierce as a mountain storm — bestowed upon {name}.",
            "kanji_details": [
                {"kanji": "玄", "reading": "Gen", "gloss": "dark mystery; profound depth and hidden wisdom"},
                {"kanji": "誠", "reading": "Sei", "gloss": "sincerity; truth and wholehearted faithfulness"},
                {"kanji": "嵐", "reading": "Ran", "gloss": "storm; untamed strength and dramatic force"},
            ],
            "trait_resonance": "玄 holds the mystery, 誠 is the sincerity that anchors it, 嵐 gives full voice to the strength within.",
        },
    ]


# Onset cluster simplification map for phonetic extraction
_ONSET_MAP = {
    "sch": "sha", "str": "su",
    "ch": "chi", "sh": "sha", "th": "sa", "ph": "fu", "qu": "ku",
    "wr": "ri",
    "br": "bu", "tr": "to", "dr": "do", "gr": "gu", "fr": "fu",
    "pr": "pu", "kr": "ku", "fl": "fu", "gl": "gu", "cl": "ku",
    "pl": "pu", "bl": "bu",
}

# Consonant-ending map for last-sound extraction
_ENDING_MAP = {
    "n": "n", "l": "ru", "r": "ru", "s": "su",
    "t": "to", "k": "ku", "m": "mu", "d": "do",
}

_VOWELS = set("aeiou")


def extract_phonetic_constraints(name: str) -> tuple[str, str]:
    """Returns (first_unit, last_unit) as romanised approximations."""
    clean = re.sub(r"[^a-z]", "", name.lower())
    if not clean:
        return ("", "")

    # --- First unit ---
    first_unit = ""
    # Try multi-char onset maps longest first
    for length in (3, 2):
        prefix = clean[:length]
        if prefix in _ONSET_MAP:
            first_unit = _ONSET_MAP[prefix]
            break
    if not first_unit:
        # Single consonant + first vowel, or just vowel
        i = 0
        consonants = ""
        while i < len(clean) and clean[i] not in _VOWELS:
            consonants += clean[i]
            i += 1
        if i < len(clean):
            first_unit = consonants + clean[i]
        else:
            first_unit = consonants[:1] if consonants else clean[0]

    # --- Last unit ---
    last_char = clean[-1]
    if last_char in _VOWELS:
        last_unit = last_char
    else:
        last_unit = _ENDING_MAP.get(last_char, "")
        if not last_unit:
            # Find last vowel and append
            for ch in reversed(clean):
                if ch in _VOWELS:
                    last_unit = ch
                    break

    return (first_unit, last_unit)


SYSTEM_PROMPT = """You are a master of ceremonial kanji signature names — the kind bestowed upon artists, scholars, and heroes in classical East Asian tradition.

Your task is to create three kanji signature names for a foreign visitor. These are not mundane given names; they are bestowed names that carry poetic weight, symbolic resonance, and visual beauty.

PHONETIC CONSTRAINTS (provided in the user message — follow them):
- The first kanji must carry a reading close to the provided first-sound approximation.
- The last kanji must carry a reading close to the provided last-sound approximation.
- Middle kanji are phonetically flexible — choose them purely for symbolic beauty and trait resonance.
- Do not stuff every syllable. Three kanji is the default. Four kanji is permitted only as a ceremonial variant when it genuinely enhances the name.

TRAIT AND STYLE CONSTRAINTS (provided in the user message — follow them):
- The user has chosen three personality traits. Your kanji selection must reflect all three traits — each kanji should bear meaning connected to at least one trait.
- If a style preference is given, let it colour the aesthetic register of the name (e.g. "mystical" → kanji associated with shadow, moon, mystery; "bright" → kanji evoking light, sun, vitality).

QUALITY RULES:
1. Prioritise beauty and symbolic meaning over phonetic completeness.
2. Never use kanji with inauspicious or negative meanings (never: 悪, 苦, 死, 暗, 凶, 禍, 憎, 怒, 怨, 病, 廃, 滅, 毒).
3. All three suggestions must feel meaningfully different from each other — vary the symbolic register, not just the characters.
4. The kanji combination must look visually harmonious as a written name.
5. Avoid modern slang kanji or overly rare characters that are unlikely to render.

OUTPUT FORMAT:
Return ONLY a JSON array with exactly 3 objects. No other text, no markdown fences. Each object:
- "kanji": the kanji string (e.g. "樹雅詩") — no hiragana or katakana
- "reading": Japanese reading in hiragana (e.g. "じゅがし")
- "overall_meaning": one English sentence capturing the combined spirit of the name
- "kanji_details": array of per-kanji objects, each with:
    - "kanji": single character
    - "reading": romanised reading (e.g. "Ju")
    - "gloss": brief English meaning tied to a chosen trait where possible
- "trait_resonance": one English sentence (max 25 words) explaining how this specific name reflects the chosen traits and/or style"""


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


VALID_TRAITS = {
    "kind", "strong", "intelligent", "elegant", "cheerful", "mysterious",
    "free-spirited", "sincere", "artistic", "calm", "poetic", "bold",
}
VALID_STYLES = {"elegant", "cool", "mystical", "poetic", "bright", ""}


@app.route("/api/generate", methods=["POST"])
def generate_kanji():
    data = request.get_json()
    name = (data.get("name") or "").strip()

    if not name:
        return jsonify({"error": "Please enter your name."}), 400

    if len(name) > 50:
        return jsonify({"error": "Name is too long (max 50 characters)."}), 400

    traits = data.get("traits") or []
    style = (data.get("style") or "").strip()

    if not isinstance(traits, list) or len(traits) != 3:
        return jsonify({"error": "Exactly 3 traits must be selected."}), 400

    if not all(isinstance(t, str) and t in VALID_TRAITS for t in traits):
        return jsonify({"error": "Invalid trait value(s)."}), 400

    if style not in VALID_STYLES:
        return jsonify({"error": "Invalid style value."}), 400

    if USE_MOCK:
        return jsonify({"suggestions": mock_suggestions(name, traits, style)})

    try:
        client = get_anthropic_client()

        first_unit, last_unit = extract_phonetic_constraints(name)
        phonetic_note = ""
        if first_unit:
            phonetic_note += f"First-sound constraint: the first kanji should carry a reading approximating '{first_unit}' (from the beginning of '{name}'). "
        if last_unit:
            phonetic_note += f"Last-sound constraint: the last kanji should carry a reading approximating '{last_unit}' (from the end of '{name}')."

        user_message = (
            f"Create three kanji signature names for the foreign name: {name}\n\n"
            f"{phonetic_note}\n\n"
            f"Chosen traits: {', '.join(traits)}\n"
            f"Style preference: {style if style else 'none'}"
        )

        import json
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": user_message,
                }
            ],
        )

        text = response.content[0].text.strip()
        suggestions = json.loads(text)

        if not isinstance(suggestions, list) or len(suggestions) != 3:
            return jsonify({"error": "Unexpected response from AI. Please try again."}), 500

        # Basic shape validation
        for s in suggestions:
            if not all(k in s for k in ("kanji", "reading", "overall_meaning", "kanji_details", "trait_resonance")):
                return jsonify({"error": "Unexpected response shape from AI. Please try again."}), 500

        return jsonify({"suggestions": suggestions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5001"))
    app.run(debug=True, host="0.0.0.0", port=port)
