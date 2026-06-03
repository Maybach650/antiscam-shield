import json
import urllib.request
import urllib.error
import time
import random
from pathlib import Path

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "llama-3.3-70b-versatile"
LANGUAGES = ["turkmen", "russian"]
SCAM_TYPES = ["bank", "police", "lottery", "relative"]

def get_api_key():
    env_path = Path.home() / "antiscam_project" / ".env"
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("GROQ_API_KEY="):
                return line.split("=", 1)[1].strip()
    return ""

def call_groq(prompt, api_key):
    payload = json.dumps({
        "model": MODEL,
        "max_tokens": 2000,
        "messages": [{"role": "user", "content": prompt}]
    }).encode("utf-8")
    req = urllib.request.Request(
        GROQ_API_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + api_key,
            "User-Agent": "curl/7.88.1",
            "Accept": "*/*"
        },
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

def build_scam_prompt(language, scam_type, batch_size=5):
    return f"""Generate {batch_size} realistic scam phone call dialogues.
Language: {language}
Scam type: fake {scam_type}
Each dialogue: 4-6 turns between scammer and victim. Keep SHORT.
Respond ONLY with valid JSON array. No markdown, no backticks.
[
  {{
    "language": "{language}",
    "type": "scam",
    "scam_category": "{scam_type}",
    "label": 1,
    "dialogue": "Scammer: ...\\nVictim: ..."
  }}
]"""

def build_normal_prompt(language, batch_size=5):
    return f"""Generate {batch_size} realistic normal phone call dialogues (NOT scam).
Language: {language}
Topics: friends, family, doctor, work, shopping. Keep SHORT.
Respond ONLY with valid JSON array. No markdown, no backticks.
[
  {{
    "language": "{language}",
    "type": "normal",
    "scam_category": "none",
    "label": 0,
    "dialogue": "Caller: ...\\nPerson: ..."
  }}
]"""

def generate_dataset(total_dialogues, api_key, output_file):
    all_data = []
    dialogue_id = 1
    batch_size = 5
    scam_total = int(total_dialogues * 0.7)
    normal_total = total_dialogues - scam_total
    print(f"\nTarget: {total_dialogues} dialogues ({scam_total} scam, {normal_total} normal)")
    print(f"Note: waiting 60s between requests to avoid rate limits")
    print("-" * 50)

    scam_per_combo = max(1, scam_total // (len(SCAM_TYPES) * len(LANGUAGES)))
    for scam_type in SCAM_TYPES:
        for language in LANGUAGES:
            generated = 0
            while generated < scam_per_combo:
                current_batch = min(batch_size, scam_per_combo - generated)
                print(f"Generating {current_batch}x {language} [{scam_type}] scam...")
                try:
                    response = call_groq(build_scam_prompt(language, scam_type, current_batch), api_key)
                    clean = response.replace("```json", "").replace("```", "").strip()
                    batch = json.loads(clean)
                    for item in batch:
                        item["id"] = dialogue_id
                        all_data.append(item)
                        dialogue_id += 1
                    generated += len(batch)
                    print(f"  OK! Got {len(batch)}. Total: {len(all_data)}. Waiting 60s...")
                    time.sleep(60)
                except urllib.error.HTTPError as e:
                    if e.code in [401, 429]:
                        print(f"  Rate limited! Waiting 60s...")
                        time.sleep(60)
                    else:
                        print(f"  HTTP Error {e.code}. Retrying in 10s...")
                        time.sleep(10)
                except json.JSONDecodeError:
                    print(f"  JSON error. Retrying in 10s...")
                    time.sleep(10)
                except Exception as e:
                    print(f"  Error: {e}. Retrying in 10s...")
                    time.sleep(10)

    normal_per_lang = max(1, normal_total // len(LANGUAGES))
    for language in LANGUAGES:
        generated = 0
        while generated < normal_per_lang:
            current_batch = min(batch_size, normal_per_lang - generated)
            print(f"Generating {current_batch}x {language} [normal]...")
            try:
                response = call_groq(build_normal_prompt(language, current_batch), api_key)
                clean = response.replace("```json", "").replace("```", "").strip()
                batch = json.loads(clean)
                for item in batch:
                    item["id"] = dialogue_id
                    all_data.append(item)
                    dialogue_id += 1
                generated += len(batch)
                print(f"  OK! Got {len(batch)}. Total: {len(all_data)}. Waiting 60s...")
                time.sleep(60)
            except urllib.error.HTTPError as e:
                if e.code in [401, 429]:
                    print(f"  Rate limited! Waiting 60s...")
                    time.sleep(60)
                else:
                    print(f"  HTTP Error {e.code}. Retrying in 10s...")
                    time.sleep(10)
            except json.JSONDecodeError:
                print(f"  JSON error. Retrying in 10s...")
                time.sleep(10)
            except Exception as e:
                print(f"  Error: {e}. Retrying in 5s...")
                time.sleep(5)

    random.shuffle(all_data)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print("-" * 50)
    print(f"Done! Total saved: {len(all_data)}")
    print(f"Scam: {len([d for d in all_data if d['type']=='scam'])} | Normal: {len([d for d in all_data if d['type']=='normal'])}")
    print(f"Turkmen: {len([d for d in all_data if d['language']=='turkmen'])} | Russian: {len([d for d in all_data if d['language']=='russian'])}")
    print(f"File: {output_file}")

if __name__ == "__main__":
    import sys
    print("=" * 50)
    print("  AI Anti-Scam Dataset Generator (Groq - FREE)")
    print("=" * 50)
    api_key = get_api_key()
    if api_key:
        print(f"API key loaded: {api_key[:15]}...")
    else:
        api_key = input("Enter your Groq API key: ").strip()
    if not api_key:
        print("Error: API key required!")
        sys.exit(1)
    try:
        total = int(input("How many dialogues? (recommended: 100-300): ").strip())
    except ValueError:
        total = 100
    data_dir = Path.home() / "antiscam_project" / "data"
    data_dir.mkdir(exist_ok=True)
    output = str(data_dir / "antiscam_dataset_large.json")
    print(f"Output: {output}")
    generate_dataset(total, api_key, output)