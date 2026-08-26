
Gemini
Neuer Chat
Chats durchsuchen
Bilder
Videos
Mediathek
Neues Notebook
Kia PV5
Untitled notebook
Alle Notebooks
MTG Card Scanner Python Script
Skateboard in Duel Commander Cloud
Ein-Buchstaben-Kennzeichen in Deutschland zulässig?
Lustige Leetspeak-Kennzeichen für GL
Online-Zulassung KFZ im RBK
Biases in Search Autocomplete
Pairwise Comparisons in Reranked Lists
Packliste Bikepacking-Tour: 4 Tage
Wirkaufendeinauto Auszahlung: Wann kommt Geld?
THG-Quote effektiv nutzen
Kia PV5: Multi-SIM nutzen
HUK Vollkasko Zusatzfahrer Kosten
Upgrading Kia PV5 Audio Systems
Neuzulassung eines E-Autos online
Bedeutung von HSN und TSN
Ladesysteme bei Aldi im Überblick
Vorbereitung auf den Kia PV5 Kauf
Lachsforelle mit Kräutern füllen
Deckliste Für WhatsApp Aufbereitet
Kia PV5 Internet: Kosten und Optionen
EnBW Ladevorgang mit BMW i3 beenden
Intervallfasten: 16:8 Methode erklärt
Passat 2008 Diesel Preisermittlung
Vertrag: Käufer-Vorsicht & Besonderheiten
Digitale Unterschrift für Handwerkeraufträge
Fahrzeugzulassung am Hauptwohnsitz
Elektroautos: Kofferraum, Preis, Reichweite
Zu versteuerndes Einkommen Familie mit Kind
Zinsberechnung: Geldanlage vs. Kredit
Kia PV5 Plus vs. Elite Unterschiede
Winterkompletträder für Kia PV5
Mountain Riddle Solved And Explained
Bußgeldstelle Köln kontaktieren wegen Blitzer
Autoverkauf und Weiterfahrt nach Termin
ADAC Pannenhilfe bei Batterieproblemen
Bergische Waffeln Rezept Ohne Milch
Hausratversicherung wechseln: Kündigung & Fristen
ich möchte ein AUto verkaufen, dass sich realistischerweise so für 20-21K verkaufen lässt. ein Händler hat mir bisher 15K geboten, jetzt fragt mich ein anderer händler nach einer preisvorstellung. mit 18K wäre ich sehr zufrieden, 16-17K meine schmerzgrenze. was sage ich
Mineralwasser als Scheibenwischwasser?
Kia PV5: 2-Zonen-Klimaautomatik Ausstattung
Zinseszins-Vergleich: 5 % vs. 7 %
Größte Kia Händler Deutschlands

Aktivitäten
Burscheid, Deutschland
Basierend auf deinen Orten (Zuhause)
Standort aktualisieren
Unterhaltung mit Gemini
write a python skript that uses Gemini models (use current models!) to First identify magic the gathering cards on an image (taken with a smarthphone), then "justify" them", then recognize them (there are multiple options: use existing image based recognition, parse text and find card with best full text match from the scryfall api https://scryfall.com/docs/api or best, a combined approach) and output them as a JSON file. also mark on the input image where cards have been recognized, which have been succesfully matched and which have not.

This Python script uses the official Google GenAI SDK (google-genai), OpenCV for perspective transformation and annotation, and Scryfall's REST API for verification and metadata enrichment.  
Google Cloud Documentation

Prerequisites
Bash
pip install google-genai opencv-python pillow pydantic requests
Make sure your Gemini API key is exported:

Bash
export GEMINI_API_KEY="your_api_key_here"
Complete Python Script (mtg_scanner.py)
Python
import json
import os
import re
import time
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import requests
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Pydantic Schemas for Structured Gemini Vision Output
# ---------------------------------------------------------
class CornerPoints(BaseModel):
    """Normalized coordinates (0-1000) for the 4 corners of the card (top-left, top-right, bottom-right, bottom-left)."""
    top_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    top_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")

class DetectedCard(BaseModel):
    card_name_raw: str = Field(..., description="The card name as read from the top of the card")
    set_code: Optional[str] = Field(None, description="3-letter set code if visible in bottom left corner (e.g., NEO, MH2, LTR)")
    collector_number: Optional[str] = Field(None, description="Collector number if visible in bottom left (e.g., 145, 023)")
    mana_cost_or_type: Optional[str] = Field(None, description="Mana cost or type line snippet to aid disambiguation")
    box_2d: List[int] = Field(..., description="Bounding box as [ymin, xmin, ymax, xmax] normalized to 0-1000")
    corners: Optional[CornerPoints] = Field(None, description="4 corners for perspective alignment/justification")

class CardDetectionResult(BaseModel):
    cards: List[DetectedCard] = Field(default_factory=list, description="List of detected MTG cards in the image")


# ---------------------------------------------------------
# Scryfall API Matcher
# ---------------------------------------------------------
class ScryfallClient:
    BASE_URL = "https://api.scryfall.com"
    HEADERS = {"User-Agent": "MTGCardScannerBot/1.0", "Accept": "application/json"}

    @classmethod
    def match_card(cls, detected: DetectedCard) -> Optional[dict]:
        """
        Combined matching approach:
        1. Set code + collector number (exact print match).
        2. Fuzzy name lookup (+ set filter if present).
        3. Full-text search fallback.
        """
        # Strategy 1: Exact Set Code + Collector Number
        if detected.set_code and detected.collector_number:
            set_clean = re.sub(r"[^a-zA-Z0-9]", "", detected.set_code).lower()
            num_clean = re.sub(r"[^a-zA-Z0-9]", "", detected.collector_number)
            url = f"{cls.BASE_URL}/cards/{set_clean}/{num_clean}"
            try:
                resp = requests.get(url, headers=cls.HEADERS, timeout=5)
                time.sleep(0.08)  # Respect Scryfall 50-100ms rate limit
                if resp.status_code == 200:
                    return resp.json()
            except requests.RequestException:
                pass

        # Strategy 2: Fuzzy Name Search
        if detected.card_name_raw:
            params = {"fuzzy": detected.card_name_raw}
            if detected.set_code:
                params["set"] = detected.set_code.lower()
            try:
                resp = requests.get(f"{cls.BASE_URL}/cards/named", params=params, headers=cls.HEADERS, timeout=5)
                time.sleep(0.08)
                if resp.status_code == 200:
                    return resp.json()
            except requests.RequestException:
                pass

        # Strategy 3: Full-text Search Fallback
        if detected.card_name_raw:
            query = f'!"{detected.card_name_raw}"'
            try:
                resp = requests.get(f"{cls.BASE_URL}/cards/search", params={"q": query}, headers=cls.HEADERS, timeout=5)
                time.sleep(0.08)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("total_cards", 0) > 0:
                        return data["data"][0]
            except requests.RequestException:
                pass

        return None


# ---------------------------------------------------------
# Image Justification (Perspective Warp & Cropping)
# ---------------------------------------------------------
def justify_card(image: np.ndarray, detected: DetectedCard, output_dir: str, index: int) -> str:
    """
    Corrects perspective warp using detected 4 corners or crops the bounding box.
    Standard MTG card aspect ratio is 63mm x 88mm (approx 1 : 1.397).
    """
    h, w = image.shape[:2]
    out_w, out_h = 450, 628  # Target resolution for justified card

    if detected.corners:
        # Denormalize coordinates (0-1000 -> image pixels)
        tl = [detected.corners.top_left[1] * w / 1000, detected.corners.top_left[0] * h / 1000]
        tr = [detected.corners.top_right[1] * w / 1000, detected.corners.top_right[0] * h / 1000]
        br = [detected.corners.bottom_right[1] * w / 1000, detected.corners.bottom_right[0] * h / 1000]
        bl = [detected.corners.bottom_left[1] * w / 1000, detected.corners.bottom_left[0] * h / 1000]

        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        dst_pts = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    else:
        # Fallback to standard 2D Bounding Box Crop
        ymin, xmin, ymax, xmax = detected.box_2d
        abs_ymin, abs_xmin = max(0, int(ymin * h / 1000)), max(0, int(xmin * w / 1000))
        abs_ymax, abs_xmax = min(h, int(ymax * h / 1000)), min(w, int(xmax * w / 1000))
        cropped = image[abs_ymin:abs_ymax, abs_xmin:abs_xmax]
        warped = cv2.resize(cropped, (out_w, out_h)) if cropped.size > 0 else np.zeros((out_h, out_w, 3), dtype=np.uint8)

    save_path = os.path.join(output_dir, f"card_justified_{index:02d}.jpg")
    cv2.imwrite(save_path, warped)
    return save_path


# ---------------------------------------------------------
# Main Processing Pipeline
# ---------------------------------------------------------
def scan_mtg_image(image_path: str, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Initialize Gemini Client (uses GEMINI_API_KEY environment variable)
    client = genai.Client()
    pil_image = Image.open(image_path)
    cv_image = cv2.imread(image_path)
    img_h, img_w = cv_image.shape[:2]

    # 2. Extract Cards with Gemini Vision
    prompt = (
        "Identify every Magic: The Gathering card in this image. "
        "For each card:\n"
        "1. Extract its exact 2D bounding box [ymin, xmin, ymax, xmax] (0-1000 normalized).\n"
        "2. Extract the 4 corner points [y, x] in order: top_left, top_right, bottom_right, bottom_left for perspective correction.\n"
        "3. Read the card title as accurately as possible.\n"
        "4. If legible, read the 3-letter set code and collector number from the bottom-left edge."
    )

    print("🤖 Analyzing image with Gemini 2.5...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[pil_image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CardDetectionResult,
            temperature=0.1,
        ),
    )

    detection_data: CardDetectionResult = response.parsed
    print(f"✨ Found {len(detection_data.cards)} card candidate(s).")

    # 3. Match against Scryfall & Justify
    final_results = []
    annotated_img = cv_image.copy()

    for idx, card in enumerate(detection_data.cards, start=1):
        print(f"[{idx}] OCR: '{card.card_name_raw}' (Set: {card.set_code}, No: {card.collector_number})...", end=" ")
        
        # Perspective transform / justify
        justified_path = justify_card(cv_image, card, output_dir, idx)
        
        # Scryfall Match
        scryfall_data = ScryfallClient.match_card(card)
        matched = scryfall_data is not None

        # Coordinates for plotting
        ymin, xmin, ymax, xmax = [
            int(card.box_2d[0] * img_h / 1000),
            int(card.box_2d[1] * img_w / 1000),
            int(card.box_2d[2] * img_h / 1000),
            int(card.box_2d[3] * img_w / 1000),
        ]

        if matched:
            print(f"✅ Matched: {scryfall_data.get('name')} (${scryfall_data.get('prices', {}).get('usd', 'N/A')})")
            box_color = (0, 220, 0)  # Green for success
            display_text = f"{scryfall_data.get('name')} | ${scryfall_data.get('prices', {}).get('usd', 'N/A')}"
        else:
            print("❌ Match Failed")
            box_color = (0, 0, 255)  # Red for unmatched
            display_text = f"Unmatched: {card.card_name_raw or 'Unknown'}"

        # Draw Bounding Box or Polygon
        if card.corners:
            pts = np.array([
                [int(card.corners.top_left[1] * img_w / 1000), int(card.corners.top_left[0] * img_h / 1000)],
                [int(card.corners.top_right[1] * img_w / 1000), int(card.corners.top_right[0] * img_h / 1000)],
                [int(card.corners.bottom_right[1] * img_w / 1000), int(card.corners.bottom_right[0] * img_h / 1000)],
                [int(card.corners.bottom_left[1] * img_w / 1000), int(card.corners.bottom_left[0] * img_h / 1000)],
            ], np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_img, [pts], isClosed=True, color=box_color, thickness=4)
        else:
            cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), box_color, 4)

        # Draw Label Banner
        text_origin = (max(10, xmin), max(30, ymin - 10))
        (text_w, text_h), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            annotated_img,
            (text_origin[0], text_origin[1] - text_h - 6),
            (text_origin[0] + text_w + 6, text_origin[1] + baseline),
            box_color,
            -1,
        )
        cv2.putText(annotated_img, display_text, (text_origin[0] + 3, text_origin[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Prepare JSON payload
        final_results.append({
            "index": idx,
            "detected_raw": card.model_dump(),
            "justified_image_path": justified_path,
            "matched": matched,
            "scryfall_data": {
                "id": scryfall_data.get("id"),
                "name": scryfall_data.get("name"),
                "mana_cost": scryfall_data.get("mana_cost"),
                "type_line": scryfall_data.get("type_line"),
                "set": scryfall_data.get("set"),
                "set_name": scryfall_data.get("set_name"),
                "collector_number": scryfall_data.get("collector_number"),
                "rarity": scryfall_data.get("rarity"),
                "prices_usd": scryfall_data.get("prices", {}).get("usd"),
                "scryfall_uri": scryfall_data.get("scryfall_uri"),
            } if matched else None,
        })

    # 4. Save Outputs
    json_path = os.path.join(output_dir, "mtg_detected_cards.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    annotated_path = os.path.join(output_dir, "annotated_overview.jpg")
    cv2.imwrite(annotated_path, annotated_img)

    print(f"\n📁 Processing complete:")
    print(f"  • JSON Result: {json_path}")
    print(f"  • Annotated Image: {annotated_path}")
    print(f"  • Justified Cards: {output_dir}/card_justified_*.jpg")


if __name__ == "__main__":
    import sys
    img_file = sys.argv[1] if len(sys.argv) > 1 else "cards.jpg"
    if not os.path.exists(img_file):
        print(f"Please provide a valid image file. Usage: python mtg_scanner.py <path_to_image>")
    else:
        scan_mtg_image(img_file)
How the Pipeline Works
Detection & Alignment: Gemini 2.5 inspects the smartphone image, returning normalized 2D bounding boxes and the 4 corner coordinates (top_left, top_right, bottom_right, bottom_left).  
Google Cloud Documentation

Justification: OpenCV's cv2.getPerspectiveTransform uses the 4 detected corners to de-skew, rotate, and rectify the card into a standard 450×628 vertical crop.

Hybrid Recognition: The card is queried against the Scryfall API sequentially: first using exact Set Code + Collector Number (for foil/alternate art precision), followed by fuzzy name lookup.  
LobeHub

Visual Status: Marked with Green for verified matches (including current USD pricing) and Red for unresolved/unmatched cards.

make it so the key is defined in the skript

Here is the updated script with a variable at the top to define your API key directly.

Python
import json
import os
import re
import time
from typing import List, Optional
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import requests
from google import genai
from google.genai import types

# =========================================================
# Configuration: Set your API Key here
# =========================================================
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"


# ---------------------------------------------------------
# Pydantic Schemas for Structured Gemini Vision Output
# ---------------------------------------------------------
class CornerPoints(BaseModel):
    """Normalized coordinates (0-1000) for the 4 corners of the card (top-left, top-right, bottom-right, bottom-left)."""
    top_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    top_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")

class DetectedCard(BaseModel):
    card_name_raw: str = Field(..., description="The card name as read from the top of the card")
    set_code: Optional[str] = Field(None, description="3-letter set code if visible in bottom left corner (e.g., NEO, MH2, LTR)")
    collector_number: Optional[str] = Field(None, description="Collector number if visible in bottom left (e.g., 145, 023)")
    mana_cost_or_type: Optional[str] = Field(None, description="Mana cost or type line snippet to aid disambiguation")
    box_2d: List[int] = Field(..., description="Bounding box as [ymin, xmin, ymax, xmax] normalized to 0-1000")
    corners: Optional[CornerPoints] = Field(None, description="4 corners for perspective alignment/justification")

class CardDetectionResult(BaseModel):
    cards: List[DetectedCard] = Field(default_factory=list, description="List of detected MTG cards in the image")


# ---------------------------------------------------------
# Scryfall API Matcher
# ---------------------------------------------------------
class ScryfallClient:
    BASE_URL = "https://api.scryfall.com"
    HEADERS = {"User-Agent": "MTGCardScannerBot/1.0", "Accept": "application/json"}

    @classmethod
    def match_card(cls, detected: DetectedCard) -> Optional[dict]:
        """
        Combined matching approach:
        1. Set code + collector number (exact print match).
        2. Fuzzy name lookup (+ set filter if present).
        3. Full-text search fallback.
        """
        # Strategy 1: Exact Set Code + Collector Number
        if detected.set_code and detected.collector_number:
            set_clean = re.sub(r"[^a-zA-Z0-9]", "", detected.set_code).lower()
            num_clean = re.sub(r"[^a-zA-Z0-9]", "", detected.collector_number)
            url = f"{cls.BASE_URL}/cards/{set_clean}/{num_clean}"
            try:
                resp = requests.get(url, headers=cls.HEADERS, timeout=5)
                time.sleep(0.08)  # Respect Scryfall 50-100ms rate limit
                if resp.status_code == 200:
                    return resp.json()
            except requests.RequestException:
                pass

        # Strategy 2: Fuzzy Name Search
        if detected.card_name_raw:
            params = {"fuzzy": detected.card_name_raw}
            if detected.set_code:
                params["set"] = detected.set_code.lower()
            try:
                resp = requests.get(f"{cls.BASE_URL}/cards/named", params=params, headers=cls.HEADERS, timeout=5)
                time.sleep(0.08)
                if resp.status_code == 200:
                    return resp.json()
            except requests.RequestException:
                pass

        # Strategy 3: Full-text Search Fallback
        if detected.card_name_raw:
            query = f'!"{detected.card_name_raw}"'
            try:
                resp = requests.get(f"{cls.BASE_URL}/cards/search", params={"q": query}, headers=cls.HEADERS, timeout=5)
                time.sleep(0.08)
                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("total_cards", 0) > 0:
                        return data["data"][0]
            except requests.RequestException:
                pass

        return None


# ---------------------------------------------------------
# Image Justification (Perspective Warp & Cropping)
# ---------------------------------------------------------
def justify_card(image: np.ndarray, detected: DetectedCard, output_dir: str, index: int) -> str:
    h, w = image.shape[:2]
    out_w, out_h = 450, 628  # Target resolution for justified card

    if detected.corners:
        tl = [detected.corners.top_left[1] * w / 1000, detected.corners.top_left[0] * h / 1000]
        tr = [detected.corners.top_right[1] * w / 1000, detected.corners.top_right[0] * h / 1000]
        br = [detected.corners.bottom_right[1] * w / 1000, detected.corners.bottom_right[0] * h / 1000]
        bl = [detected.corners.bottom_left[1] * w / 1000, detected.corners.bottom_left[0] * h / 1000]

        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        dst_pts = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    else:
        ymin, xmin, ymax, xmax = detected.box_2d
        abs_ymin, abs_xmin = max(0, int(ymin * h / 1000)), max(0, int(xmin * w / 1000))
        abs_ymax, abs_xmax = min(h, int(ymax * h / 1000)), min(w, int(xmax * w / 1000))
        cropped = image[abs_ymin:abs_ymax, abs_xmin:abs_xmax]
        warped = cv2.resize(cropped, (out_w, out_h)) if cropped.size > 0 else np.zeros((out_h, out_w, 3), dtype=np.uint8)

    save_path = os.path.join(output_dir, f"card_justified_{index:02d}.jpg")
    cv2.imwrite(save_path, warped)
    return save_path


# ---------------------------------------------------------
# Main Processing Pipeline
# ---------------------------------------------------------
def scan_mtg_image(image_path: str, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize Gemini Client with explicit API key
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    pil_image = Image.open(image_path)
    cv_image = cv2.imread(image_path)
    img_h, img_w = cv_image.shape[:2]

    prompt = (
        "Identify every Magic: The Gathering card in this image. "
        "For each card:\n"
        "1. Extract its exact 2D bounding box [ymin, xmin, ymax, xmax] (0-1000 normalized).\n"
        "2. Extract the 4 corner points [y, x] in order: top_left, top_right, bottom_right, bottom_left for perspective correction.\n"
        "3. Read the card title as accurately as possible.\n"
        "4. If legible, read the 3-letter set code and collector number from the bottom-left edge."
    )

    print("🤖 Analyzing image with Gemini 2.5...")
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[pil_image, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CardDetectionResult,
            temperature=0.1,
        ),
    )

    detection_data: CardDetectionResult = response.parsed
    print(f"✨ Found {len(detection_data.cards)} card candidate(s).")

    final_results = []
    annotated_img = cv_image.copy()

    for idx, card in enumerate(detection_data.cards, start=1):
        print(f"[{idx}] OCR: '{card.card_name_raw}' (Set: {card.set_code}, No: {card.collector_number})...", end=" ")
        
        # Perspective transform / justify
        justified_path = justify_card(cv_image, card, output_dir, idx)
        
        # Scryfall Match
        scryfall_data = ScryfallClient.match_card(card)
        matched = scryfall_data is not None

        ymin, xmin, ymax, xmax = [
            int(card.box_2d[0] * img_h / 1000),
            int(card.box_2d[1] * img_w / 1000),
            int(card.box_2d[2] * img_h / 1000),
            int(card.box_2d[3] * img_w / 1000),
        ]

        if matched:
            print(f"✅ Matched: {scryfall_data.get('name')} (${scryfall_data.get('prices', {}).get('usd', 'N/A')})")
            box_color = (0, 220, 0)
            display_text = f"{scryfall_data.get('name')} | ${scryfall_data.get('prices', {}).get('usd', 'N/A')}"
        else:
            print("❌ Match Failed")
            box_color = (0, 0, 255)
            display_text = f"Unmatched: {card.card_name_raw or 'Unknown'}"

        # Draw Polygon / Box
        if card.corners:
            pts = np.array([
                [int(card.corners.top_left[1] * img_w / 1000), int(card.corners.top_left[0] * img_h / 1000)],
                [int(card.corners.top_right[1] * img_w / 1000), int(card.corners.top_right[0] * img_h / 1000)],
                [int(card.corners.bottom_right[1] * img_w / 1000), int(card.corners.bottom_right[0] * img_h / 1000)],
                [int(card.corners.bottom_left[1] * img_w / 1000), int(card.corners.bottom_left[0] * img_h / 1000)],
            ], np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_img, [pts], isClosed=True, color=box_color, thickness=4)
        else:
            cv2.rectangle(annotated_img, (xmin, ymin), (xmax, ymax), box_color, 4)

        # Draw Label
        text_origin = (max(10, xmin), max(30, ymin - 10))
        (text_w, text_h), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(
            annotated_img,
            (text_origin[0], text_origin[1] - text_h - 6),
            (text_origin[0] + text_w + 6, text_origin[1] + baseline),
            box_color,
            -1,
        )
        cv2.putText(annotated_img, display_text, (text_origin[0] + 3, text_origin[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        final_results.append({
            "index": idx,
            "detected_raw": card.model_dump(),
            "justified_image_path": justified_path,
            "matched": matched,
            "scryfall_data": {
                "id": scryfall_data.get("id"),
                "name": scryfall_data.get("name"),
                "mana_cost": scryfall_data.get("mana_cost"),
                "type_line": scryfall_data.get("type_line"),
                "set": scryfall_data.get("set"),
                "set_name": scryfall_data.get("set_name"),
                "collector_number": scryfall_data.get("collector_number"),
                "rarity": scryfall_data.get("rarity"),
                "prices_usd": scryfall_data.get("prices", {}).get("usd"),
                "scryfall_uri": scryfall_data.get("scryfall_uri"),
            } if matched else None,
        })

    json_path = os.path.join(output_dir, "mtg_detected_cards.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    annotated_path = os.path.join(output_dir, "annotated_overview.jpg")
    cv2.imwrite(annotated_path, annotated_img)

    print(f"\n📁 Processing complete:")
    print(f"  • JSON Result: {json_path}")
    print(f"  • Annotated Image: {annotated_path}")
    print(f"  • Justified Cards: {output_dir}/card_justified_*.jpg")


if __name__ == "__main__":
    import sys
    img_file = sys.argv[1] if len(sys.argv) > 1 else "cards.jpg"
    if not os.path.exists(img_file):
        print(f"Please provide a valid image file. Usage: python mtg_scanner.py <path_to_image>")
    else:
        scan_mtg_image(img_file)
if it helps, use https://github.com/fortierq/mtgscan to improve what you have designed

By analyzing mtgscan (by Quentin Fortier), several key architectural insights can improve our pipeline:  
GitHub

Handling Cascaded / Stacked Cards: Real card scans often involve stacked cards where only the top name bar is visible. The prompt is enhanced so Gemini detects both full cards and overlapping/stacked cards.  
Quentin Fortier

Text Normalization & Fuzzy Fallback: MTG cards often contain diacritics, split card slashes (Fire // Ice), or small OCR typos. We add string normalization and Scryfall Autocomplete / Fuzzy lookup.  
Quentin Fortier

Local In-Memory Cache: Decks often contain duplicates (e.g., 4× Lightning Bolt). Caching card queries prevents redundant network requests and avoids hitting Scryfall's rate limit.  
GitHub

Deck Summary Aggregation: Generates a full card manifest and decklist summary (quantities and total estimated deck value) in addition to individual card crops.

Enhanced MTG Scanner Script
Python
import json
import os
import re
import time
import unicodedata
from typing import Dict, List, Optional
import cv2
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import requests
from google import genai
from google.genai import types

# ==============================================================================
# Configuration
# ==============================================================================
GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
MODEL_NAME = "gemini-2.5-flash"

# ==============================================================================
# Pydantic Schemas for Structured Gemini Detection
# ==============================================================================
class CornerPoints(BaseModel):
    """Normalized coordinates (0-1000) for the 4 corners: top-left, top-right, bottom-right, bottom-left."""
    top_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    top_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")

class DetectedCard(BaseModel):
    card_name_raw: str = Field(..., description="Exact card name or visible header text")
    set_code: Optional[str] = Field(None, description="3-4 character set code if visible (e.g., MH2, NEO, LTR)")
    collector_number: Optional[str] = Field(None, description="Collector number if visible (e.g., 145, 023a)")
    is_partially_obscured: bool = Field(False, description="True if card is stacked/overlapped (only top name bar visible)")
    box_2d: List[int] = Field(..., description="Bounding box [ymin, xmin, ymax, xmax] (0-1000 normalized)")
    corners: Optional[CornerPoints] = Field(None, description="Perspective corners (0-1000 normalized)")

class CardDetectionResult(BaseModel):
    cards: List[DetectedCard] = Field(default_factory=list, description="All detected MTG cards in the image")


# ==============================================================================
# Robust Scryfall Matcher (with Caching & mtgscan-style Normalization)
# ==============================================================================
class ScryfallEngine:
    BASE_URL = "https://api.scryfall.com"
    HEADERS = {"User-Agent": "MTGScanGemini/2.0", "Accept": "application/json"}
    _cache: Dict[str, Optional[dict]] = {}

    @staticmethod
    def normalize_name(text: str) -> str:
        """Removes diacritics, extra spaces, and common OCR artifacts."""
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^\w\s\-\'/]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def match(cls, card: DetectedCard) -> Optional[dict]:
        clean_name = cls.normalize_name(card.card_name_raw)
        cache_key = f"{clean_name}|{card.set_code}|{card.collector_number}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        result = cls._resolve_card(card, clean_name)
        cls._cache[cache_key] = result
        return result

    @classmethod
    def _resolve_card(cls, card: DetectedCard, clean_name: str) -> Optional[dict]:
        # 1. Exact Set + Collector Number
        if card.set_code and card.collector_number:
            s_code = re.sub(r"[^a-zA-Z0-9]", "", card.set_code).lower()
            c_num = re.sub(r"[^a-zA-Z0-9]", "", card.collector_number)
            try:
                r = requests.get(f"{cls.BASE_URL}/cards/{s_code}/{c_num}", headers=cls.HEADERS, timeout=5)
                time.sleep(0.08)
                if r.status_code == 200:
                    return r.json()
            except requests.RequestException:
                pass

        if not clean_name:
            return None

        # 2. Scryfall Exact / Fuzzy Search
        try:
            params = {"fuzzy": clean_name}
            if card.set_code:
                params["set"] = card.set_code.lower()
            r = requests.get(f"{cls.BASE_URL}/cards/named", params=params, headers=cls.HEADERS, timeout=5)
            time.sleep(0.08)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass

        # 3. Scryfall Autocomplete Suggestions (mtgscan typo-tolerance fallback)
        try:
            r = requests.get(f"{cls.BASE_URL}/cards/autocomplete", params={"q": clean_name}, headers=cls.HEADERS, timeout=5)
            time.sleep(0.08)
            if r.status_code == 200:
                suggestions = r.json().get("data", [])
                if suggestions:
                    # Query the top autocomplete match
                    top_match = suggestions[0]
                    r_top = requests.get(f"{cls.BASE_URL}/cards/named", params={"exact": top_match}, headers=cls.HEADERS, timeout=5)
                    time.sleep(0.08)
                    if r_top.status_code == 200:
                        return r_top.json()
        except requests.RequestException:
            pass

        return None


# ==============================================================================
# Perspective Warping & Justification
# ==============================================================================
def justify_and_crop(image: np.ndarray, card: DetectedCard, out_dir: str, index: int) -> str:
    """Warp card to canonical 63x88 MTG aspect ratio (or crop visible header for stacked cards)."""
    h, w = image.shape[:2]
    out_w, out_h = 450, 628

    if card.corners and not card.is_partially_obscured:
        tl = [card.corners.top_left[1] * w / 1000.0, card.corners.top_left[0] * h / 1000.0]
        tr = [card.corners.top_right[1] * w / 1000.0, card.corners.top_right[0] * h / 1000.0]
        br = [card.corners.bottom_right[1] * w / 1000.0, card.corners.bottom_right[0] * h / 1000.0]
        bl = [card.corners.bottom_left[1] * w / 1000.0, card.corners.bottom_left[0] * h / 1000.0]

        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        dst_pts = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    else:
        # Standard bounding box crop
        ymin, xmin, ymax, xmax = card.box_2d
        y1, x1 = max(0, int(ymin * h / 1000.0)), max(0, int(xmin * w / 1000.0))
        y2, x2 = min(h, int(ymax * h / 1000.0)), min(w, int(xmax * w / 1000.0))
        cropped = image[y1:y2, x1:x2]
        warped = cropped if cropped.size > 0 else np.zeros((out_h, out_w, 3), dtype=np.uint8)

    save_path = os.path.join(out_dir, f"card_{index:02d}.jpg")
    cv2.imwrite(save_path, warped)
    return save_path


# ==============================================================================
# Pipeline Execution
# ==============================================================================
def process_mtg_image(image_path: str, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)

    client = genai.Client(api_key=GEMINI_API_KEY)
    pil_img = Image.open(image_path)
    cv_img = cv2.imread(image_path)
    img_h, img_w = cv_img.shape[:2]

    # mtgscan-inspired prompt: handles both isolated cards and cascaded/stacked decks
    prompt = (
        "Identify all Magic: The Gathering cards in this photo (including stacked or partially overlapping cards).\n"
        "For each card:\n"
        "1. Extract its normalized 2D bounding box `box_2d` [ymin, xmin, ymax, xmax] (scale 0-1000).\n"
        "2. Extract its 4 outer corners `corners` in order: top_left, top_right, bottom_right, bottom_left (scale 0-1000).\n"
        "3. Read the card title as accurately as possible (`card_name_raw`).\n"
        "4. If legible, read the 3-letter set code and collector number from the bottom-left border.\n"
        "5. Set `is_partially_obscured` to true if the card is stacked under another card and only its top title bar is visible."
    )

    print(f"🔍 Analyzing '{image_path}' with {MODEL_NAME}...")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[pil_img, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CardDetectionResult,
            temperature=0.1,
        ),
    )

    detection_data: CardDetectionResult = response.parsed
    print(f"🃏 Located {len(detection_data.cards)} card candidate(s).\n")

    annotated = cv_img.copy()
    results = []
    deck_counts: Dict[str, int] = {}
    total_value_usd = 0.0

    for idx, card in enumerate(detection_data.cards, start=1):
        print(f"[{idx:02d}] Extracted: '{card.card_name_raw}' | Set: {card.set_code or '-'} | Stk: {card.is_partially_obscured}", end=" -> ")
        
        # 1. Perspective unwarp / crop
        justified_file = justify_and_crop(cv_img, card, output_dir, idx)

        # 2. Scryfall matching
        match = ScryfallEngine.match(card)

        # 3. Annotation styling
        if match:
            c_name = match.get("name", "Unknown")
            price_usd = float(match.get("prices", {}).get("usd") or 0.0)
            total_value_usd += price_usd
            deck_counts[c_name] = deck_counts.get(c_name, 0) + 1

            label_text = f"{c_name} (${price_usd:.2f})"
            color = (0, 200, 0)  # Green
            print(f"✅ Matched: {c_name} (${price_usd:.2f})")
        else:
            label_text = f"Unmatched: {card.card_name_raw or 'Unknown'}"
            color = (0, 0, 230)  # Red
            print("❌ Match Failed")

        # 4. Draw overlays
        if card.corners and not card.is_partially_obscured:
            poly_pts = np.array([
                [int(card.corners.top_left[1] * img_w / 1000.0), int(card.corners.top_left[0] * img_h / 1000.0)],
                [int(card.corners.top_right[1] * img_w / 1000.0), int(card.corners.top_right[0] * img_h / 1000.0)],
                [int(card.corners.bottom_right[1] * img_w / 1000.0), int(card.corners.bottom_right[0] * img_h / 1000.0)],
                [int(card.corners.bottom_left[1] * img_w / 1000.0), int(card.corners.bottom_left[0] * img_h / 1000.0)],
            ], np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [poly_pts], isClosed=True, color=color, thickness=3)
        else:
            ymin, xmin, ymax, xmax = card.box_2d
            p1 = (int(xmin * img_w / 1000.0), int(ymin * img_h / 1000.0))
            p2 = (int(xmax * img_w / 1000.0), int(ymax * img_h / 1000.0))
            cv2.rectangle(annotated, p1, p2, color, 3)

        # Label tag
        ymin, xmin = int(card.box_2d[0] * img_h / 1000.0), int(card.box_2d[1] * img_w / 1000.0)
        label_pos = (max(10, xmin), max(25, ymin - 8))
        (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (label_pos[0], label_pos[1] - th - 4), (label_pos[0] + tw + 6, label_pos[1] + bl), color, -1)
        cv2.putText(annotated, label_text, (label_pos[0] + 3, label_pos[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # 5. Append JSON payload
        results.append({
            "index": idx,
            "detected_raw": card.model_dump(),
            "justified_image": justified_file,
            "matched": match is not None,
            "scryfall": {
                "id": match.get("id"),
                "name": match.get("name"),
                "mana_cost": match.get("mana_cost"),
                "type_line": match.get("type_line"),
                "set": match.get("set"),
                "set_name": match.get("set_name"),
                "collector_number": match.get("collector_number"),
                "rarity": match.get("rarity"),
                "price_usd": match.get("prices", {}).get("usd"),
                "scryfall_uri": match.get("scryfall_uri")
            } if match else None
        })

    # Save Output Artifacts
    output_manifest = {
        "summary": {
            "total_cards_detected": len(detection_data.cards),
            "matched_cards_count": sum(1 for r in results if r["matched"]),
            "estimated_deck_value_usd": round(total_value_usd, 2),
            "decklist": [f"{qty}x {name}" for name, qty in deck_counts.items()]
        },
        "cards": results
    }

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_manifest, f, indent=2, ensure_ascii=False)

    annotated_path = os.path.join(output_dir, "annotated_overview.jpg")
    cv2.imwrite(annotated_path, annotated)

    print(f"\n✨ Scan Complete!")
    print(f"📊 Deck Summary: {len(deck_counts)} unique cards | Total Value: ${total_value_usd:.2f}")
    print(f"💾 Results saved to: '{json_path}' and '{annotated_path}'")


if __name__ == "__main__":
    import sys
    img_input = sys.argv[1] if len(sys.argv) > 1 else "cards.jpg"
    if os.path.exists(img_input):
        process_mtg_image(img_input)
    else:
        print(f"Usage: python mtg_scanner.py <path_to_image>")
make it load the gemini key from a .env file

Install python-dotenv first:

Bash
pip install python-dotenv
Create a .env file in the same directory as the script:

Code-Snippet
GEMINI_API_KEY=your_actual_api_key_here
Updated Script
Python
import json
import os
import re
import time
import unicodedata
from typing import Dict, List, Optional
import cv2
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import requests
from google import genai
from google.genai import types

# ==============================================================================
# Environment Configuration
# ==============================================================================
load_dotenv()  # Loads variables from .env into os.environ

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please set it in your .env file.")

MODEL_NAME = "gemini-2.5-flash"

# ==============================================================================
# Pydantic Schemas for Structured Gemini Detection
# ==============================================================================
class CornerPoints(BaseModel):
    """Normalized coordinates (0-1000) for the 4 corners: top-left, top-right, bottom-right, bottom-left."""
    top_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    top_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")

class DetectedCard(BaseModel):
    card_name_raw: str = Field(..., description="Exact card name or visible header text")
    set_code: Optional[str] = Field(None, description="3-4 character set code if visible (e.g., MH2, NEO, LTR)")
    collector_number: Optional[str] = Field(None, description="Collector number if visible (e.g., 145, 023a)")
    is_partially_obscured: bool = Field(False, description="True if card is stacked/overlapped (only top name bar visible)")
    box_2d: List[int] = Field(..., description="Bounding box [ymin, xmin, ymax, xmax] (0-1000 normalized)")
    corners: Optional[CornerPoints] = Field(None, description="Perspective corners (0-1000 normalized)")

class CardDetectionResult(BaseModel):
    cards: List[DetectedCard] = Field(default_factory=list, description="All detected MTG cards in the image")


# ==============================================================================
# Robust Scryfall Matcher (with Caching & mtgscan-style Normalization)
# ==============================================================================
class ScryfallEngine:
    BASE_URL = "https://api.scryfall.com"
    HEADERS = {"User-Agent": "MTGScanGemini/2.0", "Accept": "application/json"}
    _cache: Dict[str, Optional[dict]] = {}

    @staticmethod
    def normalize_name(text: str) -> str:
        """Removes diacritics, extra spaces, and common OCR artifacts."""
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^\w\s\-\'/]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def match(cls, card: DetectedCard) -> Optional[dict]:
        clean_name = cls.normalize_name(card.card_name_raw)
        cache_key = f"{clean_name}|{card.set_code}|{card.collector_number}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        result = cls._resolve_card(card, clean_name)
        cls._cache[cache_key] = result
        return result

    @classmethod
    def _resolve_card(cls, card: DetectedCard, clean_name: str) -> Optional[dict]:
        # 1. Exact Set + Collector Number
        if card.set_code and card.collector_number:
            s_code = re.sub(r"[^a-zA-Z0-9]", "", card.set_code).lower()
            c_num = re.sub(r"[^a-zA-Z0-9]", "", card.collector_number)
            try:
                r = requests.get(f"{cls.BASE_URL}/cards/{s_code}/{c_num}", headers=cls.HEADERS, timeout=5)
                time.sleep(0.08)
                if r.status_code == 200:
                    return r.json()
            except requests.RequestException:
                pass

        if not clean_name:
            return None

        # 2. Scryfall Exact / Fuzzy Search
        try:
            params = {"fuzzy": clean_name}
            if card.set_code:
                params["set"] = card.set_code.lower()
            r = requests.get(f"{cls.BASE_URL}/cards/named", params=params, headers=cls.HEADERS, timeout=5)
            time.sleep(0.08)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass

        # 3. Scryfall Autocomplete Suggestions (mtgscan typo-tolerance fallback)
        try:
            r = requests.get(f"{cls.BASE_URL}/cards/autocomplete", params={"q": clean_name}, headers=cls.HEADERS, timeout=5)
            time.sleep(0.08)
            if r.status_code == 200:
                suggestions = r.json().get("data", [])
                if suggestions:
                    top_match = suggestions[0]
                    r_top = requests.get(f"{cls.BASE_URL}/cards/named", params={"exact": top_match}, headers=cls.HEADERS, timeout=5)
                    time.sleep(0.08)
                    if r_top.status_code == 200:
                        return r_top.json()
        except requests.RequestException:
            pass

        return None


# ==============================================================================
# Perspective Warping & Justification
# ==============================================================================
def justify_and_crop(image: np.ndarray, card: DetectedCard, out_dir: str, index: int) -> str:
    h, w = image.shape[:2]
    out_w, out_h = 450, 628

    if card.corners and not card.is_partially_obscured:
        tl = [card.corners.top_left[1] * w / 1000.0, card.corners.top_left[0] * h / 1000.0]
        tr = [card.corners.top_right[1] * w / 1000.0, card.corners.top_right[0] * h / 1000.0]
        br = [card.corners.bottom_right[1] * w / 1000.0, card.corners.bottom_right[0] * h / 1000.0]
        bl = [card.corners.bottom_left[1] * w / 1000.0, card.corners.bottom_left[0] * h / 1000.0]

        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        dst_pts = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    else:
        ymin, xmin, ymax, xmax = card.box_2d
        y1, x1 = max(0, int(ymin * h / 1000.0)), max(0, int(xmin * w / 1000.0))
        y2, x2 = min(h, int(ymax * h / 1000.0)), min(w, int(xmax * w / 1000.0))
        cropped = image[y1:y2, x1:x2]
        warped = cropped if cropped.size > 0 else np.zeros((out_h, out_w, 3), dtype=np.uint8)

    save_path = os.path.join(out_dir, f"card_{index:02d}.jpg")
    cv2.imwrite(save_path, warped)
    return save_path


# ==============================================================================
# Pipeline Execution
# ==============================================================================
def process_mtg_image(image_path: str, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)

    client = genai.Client(api_key=GEMINI_API_KEY)
    pil_img = Image.open(image_path)
    cv_img = cv2.imread(image_path)
    img_h, img_w = cv_img.shape[:2]

    prompt = (
        "Identify all Magic: The Gathering cards in this photo (including stacked or partially overlapping cards).\n"
        "For each card:\n"
        "1. Extract its normalized 2D bounding box `box_2d` [ymin, xmin, ymax, xmax] (scale 0-1000).\n"
        "2. Extract its 4 outer corners `corners` in order: top_left, top_right, bottom_right, bottom_left (scale 0-1000).\n"
        "3. Read the card title as accurately as possible (`card_name_raw`).\n"
        "4. If legible, read the 3-letter set code and collector number from the bottom-left border.\n"
        "5. Set `is_partially_obscured` to true if the card is stacked under another card and only its top title bar is visible."
    )

    print(f"🔍 Analyzing '{image_path}' with {MODEL_NAME}...")
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[pil_img, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CardDetectionResult,
            temperature=0.1,
        ),
    )

    detection_data: CardDetectionResult = response.parsed
    print(f"🃏 Located {len(detection_data.cards)} card candidate(s).\n")

    annotated = cv_img.copy()
    results = []
    deck_counts: Dict[str, int] = {}
    total_value_usd = 0.0

    for idx, card in enumerate(detection_data.cards, start=1):
        print(f"[{idx:02d}] Extracted: '{card.card_name_raw}' | Set: {card.set_code or '-'} | Stk: {card.is_partially_obscured}", end=" -> ")
        
        justified_file = justify_and_crop(cv_img, card, output_dir, idx)
        match = ScryfallEngine.match(card)

        if match:
            c_name = match.get("name", "Unknown")
            price_usd = float(match.get("prices", {}).get("usd") or 0.0)
            total_value_usd += price_usd
            deck_counts[c_name] = deck_counts.get(c_name, 0) + 1

            label_text = f"{c_name} (${price_usd:.2f})"
            color = (0, 200, 0)  # Green
            print(f"✅ Matched: {c_name} (${price_usd:.2f})")
        else:
            label_text = f"Unmatched: {card.card_name_raw or 'Unknown'}"
            color = (0, 0, 230)  # Red
            print("❌ Match Failed")

        if card.corners and not card.is_partially_obscured:
            poly_pts = np.array([
                [int(card.corners.top_left[1] * img_w / 1000.0), int(card.corners.top_left[0] * img_h / 1000.0)],
                [int(card.corners.top_right[1] * img_w / 1000.0), int(card.corners.top_right[0] * img_h / 1000.0)],
                [int(card.corners.bottom_right[1] * img_w / 1000.0), int(card.corners.bottom_right[0] * img_h / 1000.0)],
                [int(card.corners.bottom_left[1] * img_w / 1000.0), int(card.corners.bottom_left[0] * img_h / 1000.0)],
            ], np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [poly_pts], isClosed=True, color=color, thickness=3)
        else:
            ymin, xmin, ymax, xmax = card.box_2d
            p1 = (int(xmin * img_w / 1000.0), int(ymin * img_h / 1000.0))
            p2 = (int(xmax * img_w / 1000.0), int(ymax * img_h / 1000.0))
            cv2.rectangle(annotated, p1, p2, color, 3)

        ymin, xmin = int(card.box_2d[0] * img_h / 1000.0), int(card.box_2d[1] * img_w / 1000.0)
        label_pos = (max(10, xmin), max(25, ymin - 8))
        (tw, th), bl = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated, (label_pos[0], label_pos[1] - th - 4), (label_pos[0] + tw + 6, label_pos[1] + bl), color, -1)
        cv2.putText(annotated, label_text, (label_pos[0] + 3, label_pos[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        results.append({
            "index": idx,
            "detected_raw": card.model_dump(),
            "justified_image": justified_file,
            "matched": match is not None,
            "scryfall": {
                "id": match.get("id"),
                "name": match.get("name"),
                "mana_cost": match.get("mana_cost"),
                "type_line": match.get("type_line"),
                "set": match.get("set"),
                "set_name": match.get("set_name"),
                "collector_number": match.get("collector_number"),
                "rarity": match.get("rarity"),
                "price_usd": match.get("prices", {}).get("usd"),
                "scryfall_uri": match.get("scryfall_uri")
            } if match else None
        })

    output_manifest = {
        "summary": {
            "total_cards_detected": len(detection_data.cards),
            "matched_cards_count": sum(1 for r in results if r["matched"]),
            "estimated_deck_value_usd": round(total_value_usd, 2),
            "decklist": [f"{qty}x {name}" for name, qty in deck_counts.items()]
        },
        "cards": results
    }

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_manifest, f, indent=2, ensure_ascii=False)

    annotated_path = os.path.join(output_dir, "annotated_overview.jpg")
    cv2.imwrite(annotated_path, annotated)

    print(f"\n✨ Scan Complete!")
    print(f"📊 Deck Summary: {len(deck_counts)} unique cards | Total Value: ${total_value_usd:.2f}")
    print(f"💾 Results saved to: '{json_path}' and '{annotated_path}'")


if __name__ == "__main__":
    import sys
    img_input = sys.argv[1] if len(sys.argv) > 1 else "cards.jpg"
    if os.path.exists(img_input):
        process_mtg_image(img_input)
    else:
        print(f"Usage: python mtg_scanner.py <path_to_image>")
there are 3 application scenarios:

1. Deck scan: output decklist

2. Cards scan: output all cards from one image

3. collection scan: parse all cards from all images in directory. output list of per image data with: cards recognised, number of unrecognized cards



this all should be possible

Here is the unified CLI script implementing all three scenarios with a modular architecture and dedicated output formatting.

CLI Usage
Bash
# Scenario 1: Deck scan (aggregates counts, formats standard MTG decklist .txt & JSON)
python mtg_scanner.py deck path/to/deck_photo.jpg

# Scenario 2: Cards scan (crops, justifies, and returns detailed card data + marked image)
python mtg_scanner.py cards path/to/cards_photo.jpg

# Scenario 3: Collection scan (processes all images in a folder, exports audit summary)
python mtg_scanner.py collection path/to/image_folder/
Complete Script (mtg_scanner.py)
Python
import argparse
import glob
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import requests
from google import genai
from google.genai import types

# ==============================================================================
# Environment Configuration
# ==============================================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please define it in your .env file.")

MODEL_NAME = "gemini-2.5-flash"
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ==============================================================================
# Structured Output Schemas for Gemini Vision
# ==============================================================================
class CornerPoints(BaseModel):
    top_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    top_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")

class DetectedCard(BaseModel):
    card_name_raw: str = Field(..., description="Exact card name or visible header text")
    set_code: Optional[str] = Field(None, description="3-4 character set code if visible")
    collector_number: Optional[str] = Field(None, description="Collector number if visible")
    is_partially_obscured: bool = Field(False, description="True if stacked/overlapped card")
    box_2d: List[int] = Field(..., description="Bounding box [ymin, xmin, ymax, xmax] (0-1000)")
    corners: Optional[CornerPoints] = Field(None, description="Perspective corners (0-1000)")

class CardDetectionResult(BaseModel):
    cards: List[DetectedCard] = Field(default_factory=list, description="All detected cards")


# ==============================================================================
# Scryfall Matcher with Memory Cache
# ==============================================================================
class ScryfallEngine:
    BASE_URL = "https://api.scryfall.com"
    HEADERS = {"User-Agent": "MTGScannerSuite/3.0", "Accept": "application/json"}
    _cache: Dict[str, Optional[dict]] = {}

    @staticmethod
    def normalize_name(text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^\w\s\-\'/]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def match(cls, card: DetectedCard) -> Optional[dict]:
        clean_name = cls.normalize_name(card.card_name_raw)
        cache_key = f"{clean_name}|{card.set_code}|{card.collector_number}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        result = cls._resolve(card, clean_name)
        cls._cache[cache_key] = result
        return result

    @classmethod
    def _resolve(cls, card: DetectedCard, clean_name: str) -> Optional[dict]:
        # 1. Exact Set + Collector Number
        if card.set_code and card.collector_number:
            s_code = re.sub(r"[^a-zA-Z0-9]", "", card.set_code).lower()
            c_num = re.sub(r"[^a-zA-Z0-9]", "", card.collector_number)
            try:
                r = requests.get(f"{cls.BASE_URL}/cards/{s_code}/{c_num}", headers=cls.HEADERS, timeout=5)
                time.sleep(0.08)
                if r.status_code == 200:
                    return r.json()
            except requests.RequestException:
                pass

        if not clean_name:
            return None

        # 2. Fuzzy Search
        try:
            params = {"fuzzy": clean_name}
            if card.set_code:
                params["set"] = card.set_code.lower()
            r = requests.get(f"{cls.BASE_URL}/cards/named", params=params, headers=cls.HEADERS, timeout=5)
            time.sleep(0.08)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass

        # 3. Autocomplete Typo Fallback
        try:
            r = requests.get(f"{cls.BASE_URL}/cards/autocomplete", params={"q": clean_name}, headers=cls.HEADERS, timeout=5)
            time.sleep(0.08)
            if r.status_code == 200:
                suggestions = r.json().get("data", [])
                if suggestions:
                    r_top = requests.get(f"{cls.BASE_URL}/cards/named", params={"exact": suggestions[0]}, headers=cls.HEADERS, timeout=5)
                    time.sleep(0.08)
                    if r_top.status_code == 200:
                        return r_top.json()
        except requests.RequestException:
            pass

        return None


# ==============================================================================
# Image Justification & Annotation
# ==============================================================================
def justify_card(image: np.ndarray, card: DetectedCard, output_path: str) -> str:
    h, w = image.shape[:2]
    out_w, out_h = 450, 628

    if card.corners and not card.is_partially_obscured:
        tl = [card.corners.top_left[1] * w / 1000.0, card.corners.top_left[0] * h / 1000.0]
        tr = [card.corners.top_right[1] * w / 1000.0, card.corners.top_right[0] * h / 1000.0]
        br = [card.corners.bottom_right[1] * w / 1000.0, card.corners.bottom_right[0] * h / 1000.0]
        bl = [card.corners.bottom_left[1] * w / 1000.0, card.corners.bottom_left[0] * h / 1000.0]

        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        dst_pts = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    else:
        ymin, xmin, ymax, xmax = card.box_2d
        y1, x1 = max(0, int(ymin * h / 1000.0)), max(0, int(xmin * w / 1000.0))
        y2, x2 = min(h, int(ymax * h / 1000.0)), min(w, int(xmax * w / 1000.0))
        cropped = image[y1:y2, x1:x2]
        warped = cropped if cropped.size > 0 else np.zeros((out_h, out_w, 3), dtype=np.uint8)

    cv2.imwrite(output_path, warped)
    return output_path


def annotate_image(image: np.ndarray, card: DetectedCard, matched_data: Optional[dict]):
    h, w = image.shape[:2]
    matched = matched_data is not None

    if matched:
        name = matched_data.get("name", "Card")
        price = matched_data.get("prices", {}).get("usd")
        label = f"{name} (${price})" if price else name
        color = (0, 200, 0)
    else:
        label = f"Unmatched: {card.card_name_raw or 'Unknown'}"
        color = (0, 0, 230)

    if card.corners and not card.is_partially_obscured:
        poly_pts = np.array([
            [int(card.corners.top_left[1] * w / 1000.0), int(card.corners.top_left[0] * h / 1000.0)],
            [int(card.corners.top_right[1] * w / 1000.0), int(card.corners.top_right[0] * h / 1000.0)],
            [int(card.corners.bottom_right[1] * w / 1000.0), int(card.corners.bottom_right[0] * h / 1000.0)],
            [int(card.corners.bottom_left[1] * w / 1000.0), int(card.corners.bottom_left[0] * h / 1000.0)],
        ], np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [poly_pts], isClosed=True, color=color, thickness=3)
    else:
        ymin, xmin, ymax, xmax = card.box_2d
        p1 = (int(xmin * w / 1000.0), int(ymin * h / 1000.0))
        p2 = (int(xmax * w / 1000.0), int(ymax * h / 1000.0))
        cv2.rectangle(image, p1, p2, color, 3)

    ymin, xmin = int(card.box_2d[0] * h / 1000.0), int(card.box_2d[1] * w / 1000.0)
    label_pos = (max(10, xmin), max(25, ymin - 8))
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(image, (label_pos[0], label_pos[1] - th - 4), (label_pos[0] + tw + 6, label_pos[1] + bl), color, -1)
    cv2.putText(image, label, (label_pos[0] + 3, label_pos[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# ==============================================================================
# Core Vision & Recognition Pipeline
# ==============================================================================
def process_single_image(image_path: str, output_dir: str, save_crops: bool = True) -> Tuple[List[dict], int, int]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    pil_img = Image.open(image_path)
    cv_img = cv2.imread(image_path)
    
    crops_dir = os.path.join(output_dir, f"{Path(image_path).stem}_crops")
    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    prompt = (
        "Identify all Magic: The Gathering cards in this photo (isolated or stacked/overlapping).\n"
        "Extract:\n"
        "1. Normalized box `box_2d` [ymin, xmin, ymax, xmax] (0-1000).\n"
        "2. Four outer `corners` in order: top_left, top_right, bottom_right, bottom_left (0-1000).\n"
        "3. Card title `card_name_raw`.\n"
        "4. 3-letter set code and collector number if visible in the bottom-left margin.\n"
        "5. `is_partially_obscured` (true if overlapping/stacked under another card)."
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[pil_img, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CardDetectionResult,
            temperature=0.1,
        ),
    )

    detection_data: CardDetectionResult = response.parsed
    annotated = cv_img.copy()
    processed_cards = []
    unrecognized_count = 0

    for idx, card in enumerate(detection_data.cards, start=1):
        justified_img_path = None
        if save_crops:
            crop_file = os.path.join(crops_dir, f"card_{idx:02d}.jpg")
            justified_img_path = justify_card(cv_img, card, crop_file)

        match = ScryfallEngine.match(card)
        if not match:
            unrecognized_count += 1

        annotate_image(annotated, card, match)

        processed_cards.append({
            "index": idx,
            "detected_raw": card.model_dump(),
            "justified_image": justified_img_path,
            "matched": match is not None,
            "scryfall": {
                "id": match.get("id"),
                "name": match.get("name"),
                "mana_cost": match.get("mana_cost"),
                "type_line": match.get("type_line"),
                "set": match.get("set"),
                "set_name": match.get("set_name"),
                "collector_number": match.get("collector_number"),
                "rarity": match.get("rarity"),
                "price_usd": match.get("prices", {}).get("usd"),
                "scryfall_uri": match.get("scryfall_uri")
            } if match else None
        })

    annotated_save_path = os.path.join(output_dir, f"{Path(image_path).stem}_annotated.jpg")
    cv2.imwrite(annotated_save_path, annotated)

    return processed_cards, len(detection_data.cards), unrecognized_count


# ==============================================================================
# Scenario 1: Deck Scan
# ==============================================================================
def scenario_deck_scan(image_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- [Scenario 1] Deck Scan: '{image_path}' ---")
    
    cards, total_detected, unrecognized = process_single_image(image_path, output_dir, save_crops=True)
    
    decklist_counts: Dict[str, int] = {}
    for c in cards:
        if c["matched"]:
            name = c["scryfall"]["name"]
            decklist_counts[name] = decklist_counts.get(name, 0) + 1

    # Standard Arena/MTGO decklist format (.txt)
    txt_path = os.path.join(output_dir, "decklist.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for name, count in sorted(decklist_counts.items(), key=lambda x: x[0]):
            f.write(f"{count} {name}\n")

    json_payload = {
        "scenario": "deck_scan",
        "image": image_path,
        "total_cards_detected": total_detected,
        "recognized_cards_count": total_detected - unrecognized,
        "unrecognized_cards_count": unrecognized,
        "decklist": [f"{count}x {name}" for name, count in sorted(decklist_counts.items(), key=lambda x: x[0])],
        "cards": cards
    }

    json_path = os.path.join(output_dir, "deck_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Decklist written to: {txt_path}")
    print(f"✅ Full scan details: {json_path}")


# ==============================================================================
# Scenario 2: Cards Scan
# ==============================================================================
def scenario_cards_scan(image_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- [Scenario 2] Single Image Cards Scan: '{image_path}' ---")
    
    cards, total_detected, unrecognized = process_single_image(image_path, output_dir, save_crops=True)

    json_payload = {
        "scenario": "cards_scan",
        "image": image_path,
        "total_detected": total_detected,
        "recognized_count": total_detected - unrecognized,
        "unrecognized_count": unrecognized,
        "cards": cards
    }

    json_path = os.path.join(output_dir, "cards_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Recognized {total_detected - unrecognized}/{total_detected} cards.")
    print(f"✅ Crops, annotated image, and JSON written to '{output_dir}/'")


# ==============================================================================
# Scenario 3: Collection Scan
# ==============================================================================
def scenario_collection_scan(directory_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- [Scenario 3] Collection Scan: Directory '{directory_path}' ---")

    all_files = [
        f for f in glob.glob(os.path.join(directory_path, "*"))
        if Path(f).suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    if not all_files:
        print(f"⚠️ No valid image files found in '{directory_path}'.")
        return

    per_image_results = []
    total_cards_all = 0
    total_unrecognized_all = 0

    for idx, img_path in enumerate(all_files, start=1):
        print(f"[{idx}/{len(all_files)}] Processing: {os.path.basename(img_path)}...")
        cards, total, unrec = process_single_image(img_path, output_dir, save_crops=True)
        
        recognized_cards = [c for c in cards if c["matched"]]
        
        per_image_results.append({
            "image_filename": os.path.basename(img_path),
            "image_path": img_path,
            "total_detected": total,
            "recognized_cards": [c["scryfall"]["name"] for c in recognized_cards],
            "recognized_cards_details": recognized_cards,
            "number_of_unrecognized_cards": unrec,
        })
        
        total_cards_all += total
        total_unrecognized_all += unrec

    manifest = {
        "scenario": "collection_scan",
        "directory": directory_path,
        "summary": {
            "total_images_processed": len(all_files),
            "total_cards_detected": total_cards_all,
            "total_cards_recognized": total_cards_all - total_unrecognized_all,
            "total_cards_unrecognized": total_unrecognized_all
        },
        "per_image_data": per_image_results
    }

    manifest_path = os.path.join(output_dir, "collection_summary.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n✨ Collection Scan Complete!")
    print(f"📊 Processed {len(all_files)} images, {total_cards_all} total cards detected.")
    print(f"📁 Manifest exported to: {manifest_path}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="MTG Vision & Recognition Suite (Gemini 2.5 + Scryfall)")
    subparsers = parser.add_subparsers(dest="scenario", required=True, help="Processing Mode")

    # Scenario 1: Deck Scan
    deck_parser = subparsers.add_parser("deck", help="Scan deck image and generate formatted decklist")
    deck_parser.add_argument("image", type=str, help="Path to deck photo")
    deck_parser.add_argument("--out", type=str, default="output_deck", help="Output directory")

    # Scenario 2: Cards Scan
    cards_parser = subparsers.add_parser("cards", help="Scan single image and extract all cards")
    cards_parser.add_argument("image", type=str, help="Path to card photo")
    cards_parser.add_argument("--out", type=str, default="output_cards", help="Output directory")

    # Scenario 3: Collection Scan
    coll_parser = subparsers.add_parser("collection", help="Scan directory of photos with per-image audit")
    coll_parser.add_argument("dir", type=str, help="Path to images directory")
    coll_parser.add_argument("--out", type=str, default="output_collection", help="Output directory")

    args = parser.parse_args()

    if args.scenario == "deck":
        scenario_deck_scan(args.image, args.out)
    elif args.scenario == "cards":
        scenario_cards_scan(args.image, args.out)
    elif args.scenario == "collection":
        scenario_collection_scan(args.dir, args.out)


if __name__ == "__main__":
    main()
give me a readme file (with import instructions etc)

Markdown
# MTG Card Scanner & Recognition Suite

An automated Magic: The Gathering card recognition pipeline powered by **Gemini 2.5 Flash Vision**, **OpenCV** perspective justification, and the **Scryfall API**.

The tool supports smartphone camera shots of isolated cards, binder pages, playmats, and stacked/cascaded deck piles across 3 primary operational modes.

---

## Features

- **Multi-Card Detection:** Detects single or multiple cards within a single image.
- **Perspective Justification:** Automatically detects card corners and warps skewed smartphone angles into rectified $450 \times 628$ crops.
- **Overlapping/Stacked Card Handling:** Detects partially visible cards in deck piles.
- **Hybrid Identification:** Combines exact Set Code + Collector Number matching, fuzzy string searching, and Scryfall autocomplete fallback.
- **Visual Validation:** Generates overview images with color-coded bounding polygons/boxes (Green = Matched, Red = Unmatched) and price overlays.
- **3 Application Modes:** Deck scanning (formatted text decklists), single-image card extraction, and bulk collection audits.

---

## Prerequisites

- Python 3.10+
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/mtg-card-scanner.git](https://github.com/yourusername/mtg-card-scanner.git)
cd mtg-card-scanner
2. Create and Activate a Virtual Environment
Bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
3. Install Required Dependencies
Bash
pip install google-genai opencv-python pillow pydantic requests python-dotenv
4. Configure Your API Key
Create a .env file in the root directory:

Bash
touch .env
Add your Gemini API key inside .env:

Code-Snippet
GEMINI_API_KEY="AIzaSyYourActualKeyGoesHere"
Usage Guide
The script provides 3 distinct subcommands: deck, cards, and collection.

1. Deck Scan (deck)
Scans an image of a deck (including overlapping piles), aggregates card quantities, and outputs both an Arena/MTGO-compatible decklist.txt and a JSON manifest.

Bash
python mtg_scanner.py deck path/to/deck_photo.jpg --out output_deck
Outputs:

output_deck/decklist.txt: Standard format (e.g., 4 Lightning Bolt, 2 Counterspell)

output_deck/deck_result.json: Summary stats + structured card metadata

output_deck/<name>_annotated.jpg: Annotated image with price labels

output_deck/<name>_crops/: Individual justified card images

2. Cards Scan (cards)
Scans an image with one or more cards (e.g., binder page, spread on a table), justifies/crops each card, and resolves card details and prices.

Bash
python mtg_scanner.py cards path/to/cards_photo.jpg --out output_cards
Outputs:

output_cards/cards_result.json: Full card metadata and Scryfall IDs

output_cards/<name>_annotated.jpg: Labeled bounding boxes/polygons

output_cards/<name>_crops/card_01.jpg, card_02.jpg, ...

3. Collection Scan (collection)
Iterates over an entire directory of photos, runs the detection pipeline on each, and exports a directory-wide summary audit showing recognized cards and unrecognized counts per image.

Bash
python mtg_scanner.py collection path/to/image_folder/ --out output_collection
Outputs:

output_collection/collection_summary.json: Aggregated metrics across all images:

Total images processed

Total cards detected / recognized / unrecognized

Per-image breakdown of recognized card names and error counts

Annotated images and cropped cards for every photo in the input folder

JSON Output Structure Example
JSON
{
  "scenario": "deck_scan",
  "image": "photos/modern_burn.jpg",
  "total_cards_detected": 60,
  "recognized_cards_count": 59,
  "unrecognized_cards_count": 1,
  "decklist": [
    "4x Lightning Bolt",
    "4x Monastery Swiftspear",
    "4x Rift Bolt"
  ],
  "cards": [
    {
      "index": 1,
      "detected_raw": {
        "card_name_raw": "Lightning Bolt",
        "set_code": "CLB",
        "collector_number": "187",
        "is_partially_obscured": false,
        "box_2d": [120, 45, 410, 240]
      },
      "justified_image": "output_deck/modern_burn_crops/card_01.jpg",
      "matched": true,
      "scryfall": {
        "id": "f29ba16f-c8fb-42fe-aabf-87089cb214a7",
        "name": "Lightning Bolt",
        "mana_cost": "{R}",
        "type_line": "Instant",
        "set": "clb",
        "set_name": "Commander Legends: Battle for Baldur's Gate",
        "collector_number": "187",
        "rarity": "common",
        "price_usd": "0.74",
        "scryfall_uri": "[https://scryfall.com/card/clb/187/lightning-bolt](https://scryfall.com/card/clb/187/lightning-bolt)"
      }
    }
  ]
}
Project Structure
Plaintext
├── .env                       # Environment variable containing GEMINI_API_KEY
├── mtg_scanner.py             # Main CLI executable and processing engine
├── README.md                  # Documentation and setup instructions
├── requirements.txt           # Dependency requirements
└── output/                    # Default destination for crops, JSON, and annotated photos
Troubleshooting & Tips
Perspective Distortion: For best results when photographing cards at steep angles, ensure all 4 card corners are in frame and well-lit.

Stacked Decks: When photographing overlapping cards, ensure the top title bar and mana cost are visible.

Scryfall Rate Limits: The script enforces a built-in 80ms delay between API queries and caches identical card lookups to respect Scryfall's API guidelines.

give me the readme as md

Markdown
# MTG Card Scanner & Recognition Suite

An automated Magic: The Gathering card recognition pipeline powered by **Gemini 2.5 Flash Vision**, **OpenCV** perspective justification, and the **Scryfall API**.

The tool supports smartphone camera shots of isolated cards, binder pages, playmats, and stacked/cascaded deck piles across 3 primary operational modes.

---

## Features

- **Multi-Card Detection:** Detects single or multiple cards within a single image.
- **Perspective Justification:** Automatically detects card corners and warps skewed smartphone angles into rectified 450x628 crops.
- **Overlapping/Stacked Card Handling:** Detects partially visible cards in deck piles.
- **Hybrid Identification:** Combines exact Set Code + Collector Number matching, fuzzy string searching, and Scryfall autocomplete fallback.
- **Visual Validation:** Generates overview images with color-coded bounding polygons/boxes (Green = Matched, Red = Unmatched) and price overlays.
- **3 Application Modes:** Deck scanning (formatted text decklists), single-image card extraction, and bulk collection audits.

---

## Prerequisites

- Python 3.10+
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/mtg-card-scanner.git](https://github.com/yourusername/mtg-card-scanner.git)
cd mtg-card-scanner
2. Create and Activate a Virtual Environment
Bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
3. Install Required Dependencies
Bash
pip install google-genai opencv-python pillow pydantic requests python-dotenv
4. Configure Your API Key
Create a .env file in the root directory:

Bash
touch .env
Add your Gemini API key inside .env:

Code-Snippet
GEMINI_API_KEY="AIzaSyYourActualKeyGoesHere"
Usage Guide
The script provides 3 distinct subcommands: deck, cards, and collection.

1. Deck Scan (deck)
Scans an image of a deck (including overlapping piles), aggregates card quantities, and outputs both an Arena/MTGO-compatible decklist.txt and a JSON manifest.

Bash
python mtg_scanner.py deck path/to/deck_photo.jpg --out output_deck
Outputs:

output_deck/decklist.txt: Standard format (e.g., 4 Lightning Bolt, 2 Counterspell)

output_deck/deck_result.json: Summary stats + structured card metadata

output_deck/<name>_annotated.jpg: Annotated image with price labels

output_deck/<name>_crops/: Individual justified card images

2. Cards Scan (cards)
Scans an image with one or more cards (e.g., binder page, spread on a table), justifies/crops each card, and resolves card details and prices.

Bash
python mtg_scanner.py cards path/to/cards_photo.jpg --out output_cards
Outputs:

output_cards/cards_result.json: Full card metadata and Scryfall IDs

output_cards/<name>_annotated.jpg: Labeled bounding boxes/polygons

output_cards/<name>_crops/card_01.jpg, card_02.jpg, ...

3. Collection Scan (collection)
Iterates over an entire directory of photos, runs the detection pipeline on each, and exports a directory-wide summary audit showing recognized cards and unrecognized counts per image.

Bash
python mtg_scanner.py collection path/to/image_folder/ --out output_collection
Outputs:

output_collection/collection_summary.json: Aggregated metrics across all images:

Total images processed

Total cards detected / recognized / unrecognized

Per-image breakdown of recognized card names and error counts

Annotated images and cropped cards for every photo in the input folder

JSON Output Structure Example
JSON
{
  "scenario": "deck_scan",
  "image": "photos/modern_burn.jpg",
  "total_cards_detected": 60,
  "recognized_cards_count": 59,
  "unrecognized_cards_count": 1,
  "decklist": [
    "4x Lightning Bolt",
    "4x Monastery Swiftspear",
    "4x Rift Bolt"
  ],
  "cards": [
    {
      "index": 1,
      "detected_raw": {
        "card_name_raw": "Lightning Bolt",
        "set_code": "CLB",
        "collector_number": "187",
        "is_partially_obscured": false,
        "box_2d": [120, 45, 410, 240]
      },
      "justified_image": "output_deck/modern_burn_crops/card_01.jpg",
      "matched": true,
      "scryfall": {
        "id": "f29ba16f-c8fb-42fe-aabf-87089cb214a7",
        "name": "Lightning Bolt",
        "mana_cost": "{R}",
        "type_line": "Instant",
        "set": "clb",
        "set_name": "Commander Legends: Battle for Baldur's Gate",
        "collector_number": "187",
        "rarity": "common",
        "price_usd": "0.74",
        "scryfall_uri": "[https://scryfall.com/card/clb/187/lightning-bolt](https://scryfall.com/card/clb/187/lightning-bolt)"
      }
    }
  ]
}
Project Structure
Plaintext
├── .env                       # Environment variable containing GEMINI_API_KEY
├── mtg_scanner.py             # Main CLI executable and processing engine
├── README.md                  # Documentation and setup instructions
├── requirements.txt           # Dependency requirements
└── output/                    # Default destination for crops, JSON, and annotated photos
Troubleshooting & Tips
Perspective Distortion: For best results when photographing cards at steep angles, ensure all 4 card corners are in frame and well-lit.

Stacked Decks: When photographing overlapping cards, ensure the top title bar and mana cost are visible.

Scryfall Rate Limits: The script enforces a built-in 80ms delay between API queries and caches identical card lookups to respect Scryfall's API guidelines.

what is the estimated cost per 100 scans?

Using Gemini 2.5 Flash, the estimated API cost is approximately $0.02 to $0.05 per 100 scans (or $0.00 if using the Google AI Studio free tier). The Scryfall API is completely free to use.

Cost Breakdown per 100 Scans
Component	Usage per 100 Scans	Rate	Estimated Cost
Image Input	100 images (~258 to 1,000 tokens/img)	$0.10 / 1M input tokens	$0.003 – $0.010
Prompt Input	~100 tokens × 100 = 10k tokens	$0.10 / 1M input tokens	$0.001
Structured Output	~500 to 2,000 tokens/scan (depending on card count)	$0.40 / 1M output tokens	$0.020 – $0.080
Scryfall API	~100 to 600 REST calls	Free (Open API)	$0.000
Total Estimated Cost	—	—	~$0.02 – $0.09
Key Variables Affecting Cost
Single vs. Multi-Card Images: Scanning a single card generates very few output tokens (~100 tokens), costing closer to $0.01 per 100 scans. A 60-card deck pile generates a large JSON payload (~2,500 tokens), which increases output token usage.

Image Tokenization: Gemini tokenizes standard-resolution images at approximately 258 tokens per image, keeping vision input costs negligible.

Free Tier: Under Google AI Studio's free tier (up to 15 requests per minute for Flash models), this workflow costs nothing.

make it so it also outputs a list of cardname frequency set etc according to moxfield syntax, e.g.

1 Sol Ring (MYS1)

1 Command Tower (CMR) *F*

1 Arcane Signet (ELD)

To support Moxfield's import syntax, the script has been updated to detect foil cards (via Gemini's vision prompt) and format the decklist with Moxfield's standard format: COUNT Card Name (SET) CN *F* (e.g., 1 Sol Ring (MYS1), 1 Command Tower (CMR) 350 *F*).

Key Additions
Foil Detection: Added is_foil: bool to the Gemini vision schema to recognize foiling reflections/surface sheen.

Moxfield Formatter: Formats lines as COUNT Card Name (SET) CN *F*.

Output File: Automatically writes decklist_moxfield.txt in addition to the standard format.

Updated Script (mtg_scanner.py)
Python
import argparse
import glob
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import requests
from google import genai
from google.genai import types

# ==============================================================================
# Environment Configuration
# ==============================================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please define it in your .env file.")

MODEL_NAME = "gemini-2.5-flash"
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ==============================================================================
# Structured Output Schemas for Gemini Vision
# ==============================================================================
class CornerPoints(BaseModel):
    top_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    top_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")

class DetectedCard(BaseModel):
    card_name_raw: str = Field(..., description="Exact card name or visible header text")
    set_code: Optional[str] = Field(None, description="3-4 character set code if visible")
    collector_number: Optional[str] = Field(None, description="Collector number if visible")
    is_partially_obscured: bool = Field(False, description="True if stacked/overlapped card")
    is_foil: bool = Field(False, description="True if the card appears to be foil/holographic/shiny")
    box_2d: List[int] = Field(..., description="Bounding box [ymin, xmin, ymax, xmax] (0-1000)")
    corners: Optional[CornerPoints] = Field(None, description="Perspective corners (0-1000)")

class CardDetectionResult(BaseModel):
    cards: List[DetectedCard] = Field(default_factory=list, description="All detected cards")


# ==============================================================================
# Scryfall Matcher with Memory Cache
# ==============================================================================
class ScryfallEngine:
    BASE_URL = "https://api.scryfall.com"
    HEADERS = {"User-Agent": "MTGScannerSuite/3.1", "Accept": "application/json"}
    _cache: Dict[str, Optional[dict]] = {}

    @staticmethod
    def normalize_name(text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^\w\s\-\'/]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def match(cls, card: DetectedCard) -> Optional[dict]:
        clean_name = cls.normalize_name(card.card_name_raw)
        cache_key = f"{clean_name}|{card.set_code}|{card.collector_number}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        result = cls._resolve(card, clean_name)
        cls._cache[cache_key] = result
        return result

    @classmethod
    def _resolve(cls, card: DetectedCard, clean_name: str) -> Optional[dict]:
        # 1. Exact Set + Collector Number
        if card.set_code and card.collector_number:
            s_code = re.sub(r"[^a-zA-Z0-9]", "", card.set_code).lower()
            c_num = re.sub(r"[^a-zA-Z0-9]", "", card.collector_number)
            try:
                r = requests.get(f"{cls.BASE_URL}/cards/{s_code}/{c_num}", headers=cls.HEADERS, timeout=5)
                time.sleep(0.08)
                if r.status_code == 200:
                    return r.json()
            except requests.RequestException:
                pass

        if not clean_name:
            return None

        # 2. Fuzzy Search
        try:
            params = {"fuzzy": clean_name}
            if card.set_code:
                params["set"] = card.set_code.lower()
            r = requests.get(f"{cls.BASE_URL}/cards/named", params=params, headers=cls.HEADERS, timeout=5)
            time.sleep(0.08)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass

        # 3. Autocomplete Typo Fallback
        try:
            r = requests.get(f"{cls.BASE_URL}/cards/autocomplete", params={"q": clean_name}, headers=cls.HEADERS, timeout=5)
            time.sleep(0.08)
            if r.status_code == 200:
                suggestions = r.json().get("data", [])
                if suggestions:
                    r_top = requests.get(f"{cls.BASE_URL}/cards/named", params={"exact": suggestions[0]}, headers=cls.HEADERS, timeout=5)
                    time.sleep(0.08)
                    if r_top.status_code == 200:
                        return r_top.json()
        except requests.RequestException:
            pass

        return None


# ==============================================================================
# Moxfield Formatter Helper
# ==============================================================================
def format_moxfield_line(count: int, name: str, set_code: Optional[str], collector_number: Optional[str], is_foil: bool) -> str:
    """
    Formats entry to Moxfield syntax:
    e.g. 1 Sol Ring (MYS1) 187 *F*
         1 Command Tower (CMR)
    """
    parts = [f"{count} {name}"]
    if set_code:
        parts.append(f"({set_code.upper()})")
    if collector_number and set_code:
        parts.append(f"{collector_number}")
    if is_foil:
        parts.append("*F*")
    return " ".join(parts)


# ==============================================================================
# Image Justification & Annotation
# ==============================================================================
def justify_card(image: np.ndarray, card: DetectedCard, output_path: str) -> str:
    h, w = image.shape[:2]
    out_w, out_h = 450, 628

    if card.corners and not card.is_partially_obscured:
        tl = [card.corners.top_left[1] * w / 1000.0, card.corners.top_left[0] * h / 1000.0]
        tr = [card.corners.top_right[1] * w / 1000.0, card.corners.top_right[0] * h / 1000.0]
        br = [card.corners.bottom_right[1] * w / 1000.0, card.corners.bottom_right[0] * h / 1000.0]
        bl = [card.corners.bottom_left[1] * w / 1000.0, card.corners.bottom_left[0] * h / 1000.0]

        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        dst_pts = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    else:
        ymin, xmin, ymax, xmax = card.box_2d
        y1, x1 = max(0, int(ymin * h / 1000.0)), max(0, int(xmin * w / 1000.0))
        y2, x2 = min(h, int(ymax * h / 1000.0)), min(w, int(xmax * w / 1000.0))
        cropped = image[y1:y2, x1:x2]
        warped = cropped if cropped.size > 0 else np.zeros((out_h, out_w, 3), dtype=np.uint8)

    cv2.imwrite(output_path, warped)
    return output_path


def annotate_image(image: np.ndarray, card: DetectedCard, matched_data: Optional[dict]):
    h, w = image.shape[:2]
    matched = matched_data is not None

    if matched:
        name = matched_data.get("name", "Card")
        price = matched_data.get("prices", {}).get("usd")
        foil_tag = " (Foil)" if card.is_foil else ""
        label = f"{name}{foil_tag} (${price})" if price else f"{name}{foil_tag}"
        color = (0, 200, 0)
    else:
        label = f"Unmatched: {card.card_name_raw or 'Unknown'}"
        color = (0, 0, 230)

    if card.corners and not card.is_partially_obscured:
        poly_pts = np.array([
            [int(card.corners.top_left[1] * w / 1000.0), int(card.corners.top_left[0] * h / 1000.0)],
            [int(card.corners.top_right[1] * w / 1000.0), int(card.corners.top_right[0] * h / 1000.0)],
            [int(card.corners.bottom_right[1] * w / 1000.0), int(card.corners.bottom_right[0] * h / 1000.0)],
            [int(card.corners.bottom_left[1] * w / 1000.0), int(card.corners.bottom_left[0] * h / 1000.0)],
        ], np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [poly_pts], isClosed=True, color=color, thickness=3)
    else:
        ymin, xmin, ymax, xmax = card.box_2d
        p1 = (int(xmin * w / 1000.0), int(ymin * h / 1000.0))
        p2 = (int(xmax * w / 1000.0), int(ymax * h / 1000.0))
        cv2.rectangle(image, p1, p2, color, 3)

    ymin, xmin = int(card.box_2d[0] * h / 1000.0), int(card.box_2d[1] * w / 1000.0)
    label_pos = (max(10, xmin), max(25, ymin - 8))
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(image, (label_pos[0], label_pos[1] - th - 4), (label_pos[0] + tw + 6, label_pos[1] + bl), color, -1)
    cv2.putText(image, label, (label_pos[0] + 3, label_pos[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# ==============================================================================
# Core Vision & Recognition Pipeline
# ==============================================================================
def process_single_image(image_path: str, output_dir: str, save_crops: bool = True) -> Tuple[List[dict], int, int]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    pil_img = Image.open(image_path)
    cv_img = cv2.imread(image_path)
    
    crops_dir = os.path.join(output_dir, f"{Path(image_path).stem}_crops")
    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    prompt = (
        "Identify all Magic: The Gathering cards in this photo (isolated or stacked/overlapping).\n"
        "Extract:\n"
        "1. Normalized box `box_2d` [ymin, xmin, ymax, xmax] (0-1000).\n"
        "2. Four outer `corners` in order: top_left, top_right, bottom_right, bottom_left (0-1000).\n"
        "3. Card title `card_name_raw`.\n"
        "4. 3-letter set code and collector number if visible in the bottom-left margin.\n"
        "5. `is_partially_obscured` (true if overlapping/stacked under another card).\n"
        "6. `is_foil` (true if the card surface has holographic sheen/foil reflection)."
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[pil_img, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CardDetectionResult,
            temperature=0.1,
        ),
    )

    detection_data: CardDetectionResult = response.parsed
    annotated = cv_img.copy()
    processed_cards = []
    unrecognized_count = 0

    for idx, card in enumerate(detection_data.cards, start=1):
        justified_img_path = None
        if save_crops:
            crop_file = os.path.join(crops_dir, f"card_{idx:02d}.jpg")
            justified_img_path = justify_card(cv_img, card, crop_file)

        match = ScryfallEngine.match(card)
        if not match:
            unrecognized_count += 1

        annotate_image(annotated, card, match)

        processed_cards.append({
            "index": idx,
            "detected_raw": card.model_dump(),
            "justified_image": justified_img_path,
            "matched": match is not None,
            "scryfall": {
                "id": match.get("id"),
                "name": match.get("name"),
                "mana_cost": match.get("mana_cost"),
                "type_line": match.get("type_line"),
                "set": match.get("set"),
                "set_name": match.get("set_name"),
                "collector_number": match.get("collector_number"),
                "rarity": match.get("rarity"),
                "price_usd": match.get("prices", {}).get("usd"),
                "scryfall_uri": match.get("scryfall_uri")
            } if match else None
        })

    annotated_save_path = os.path.join(output_dir, f"{Path(image_path).stem}_annotated.jpg")
    cv2.imwrite(annotated_save_path, annotated)

    return processed_cards, len(detection_data.cards), unrecognized_count


# ==============================================================================
# Scenario 1: Deck Scan
# ==============================================================================
def scenario_deck_scan(image_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- [Scenario 1] Deck Scan: '{image_path}' ---")
    
    cards, total_detected, unrecognized = process_single_image(image_path, output_dir, save_crops=True)
    
    # Aggregate counts keyed by (name, set_code, collector_number, is_foil)
    moxfield_counts: Dict[Tuple[str, Optional[str], Optional[str], bool], int] = {}
    standard_counts: Dict[str, int] = {}

    for c in cards:
        if c["matched"]:
            name = c["scryfall"]["name"]
            set_code = c["scryfall"].get("set")
            collector_num = c["scryfall"].get("collector_number")
            is_foil = c["detected_raw"].get("is_foil", False)

            key = (name, set_code, collector_num, is_foil)
            moxfield_counts[key] = moxfield_counts.get(key, 0) + 1
            standard_counts[name] = standard_counts.get(name, 0) + 1

    # 1. Standard decklist.txt
    txt_path = os.path.join(output_dir, "decklist.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for name, count in sorted(standard_counts.items()):
            f.write(f"{count} {name}\n")

    # 2. Moxfield Syntax decklist_moxfield.txt
    moxfield_lines = []
    moxfield_txt_path = os.path.join(output_dir, "decklist_moxfield.txt")
    with open(moxfield_txt_path, "w", encoding="utf-8") as f:
        for (name, set_code, collector_num, is_foil), count in sorted(moxfield_counts.items(), key=lambda x: x[0][0]):
            line = format_moxfield_line(count, name, set_code, collector_num, is_foil)
            moxfield_lines.append(line)
            f.write(f"{line}\n")

    json_payload = {
        "scenario": "deck_scan",
        "image": image_path,
        "total_cards_detected": total_detected,
        "recognized_cards_count": total_detected - unrecognized,
        "unrecognized_cards_count": unrecognized,
        "moxfield_decklist": moxfield_lines,
        "cards": cards
    }

    json_path = os.path.join(output_dir, "deck_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Moxfield Decklist: {moxfield_txt_path}")
    print(f"✅ Standard Decklist: {txt_path}")
    print(f"✅ Full scan details: {json_path}")


# ==============================================================================
# Scenario 2: Cards Scan
# ==============================================================================
def scenario_cards_scan(image_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- [Scenario 2] Single Image Cards Scan: '{image_path}' ---")
    
    cards, total_detected, unrecognized = process_single_image(image_path, output_dir, save_crops=True)

    moxfield_entries = []
    for c in cards:
        if c["matched"]:
            name = c["scryfall"]["name"]
            set_code = c["scryfall"].get("set")
            collector_num = c["scryfall"].get("collector_number")
            is_foil = c["detected_raw"].get("is_foil", False)
            moxfield_entries.append(format_moxfield_line(1, name, set_code, collector_num, is_foil))

    json_payload = {
        "scenario": "cards_scan",
        "image": image_path,
        "total_detected": total_detected,
        "recognized_count": total_detected - unrecognized,
        "unrecognized_count": unrecognized,
        "moxfield_list": moxfield_entries,
        "cards": cards
    }

    json_path = os.path.join(output_dir, "cards_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Recognized {total_detected - unrecognized}/{total_detected} cards.")
    print(f"✅ Crops, annotated image, and JSON written to '{output_dir}/'")


# ==============================================================================
# Scenario 3: Collection Scan
# ==============================================================================
def scenario_collection_scan(directory_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- [Scenario 3] Collection Scan: Directory '{directory_path}' ---")

    all_files = [
        f for f in glob.glob(os.path.join(directory_path, "*"))
        if Path(f).suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    if not all_files:
        print(f"⚠️ No valid image files found in '{directory_path}'.")
        return

    per_image_results = []
    total_cards_all = 0
    total_unrecognized_all = 0
    collection_moxfield_counts: Dict[Tuple[str, Optional[str], Optional[str], bool], int] = {}

    for idx, img_path in enumerate(all_files, start=1):
        print(f"[{idx}/{len(all_files)}] Processing: {os.path.basename(img_path)}...")
        cards, total, unrec = process_single_image(img_path, output_dir, save_crops=True)
        
        recognized_cards = [c for c in cards if c["matched"]]
        
        for c in recognized_cards:
            name = c["scryfall"]["name"]
            set_code = c["scryfall"].get("set")
            collector_num = c["scryfall"].get("collector_number")
            is_foil = c["detected_raw"].get("is_foil", False)
            key = (name, set_code, collector_num, is_foil)
            collection_moxfield_counts[key] = collection_moxfield_counts.get(key, 0) + 1

        per_image_results.append({
            "image_filename": os.path.basename(img_path),
            "image_path": img_path,
            "total_detected": total,
            "recognized_cards": [c["scryfall"]["name"] for c in recognized_cards],
            "recognized_cards_details": recognized_cards,
            "number_of_unrecognized_cards": unrec,
        })
        
        total_cards_all += total
        total_unrecognized_all += unrec

    # Save complete collection Moxfield list
    coll_moxfield_txt = os.path.join(output_dir, "collection_moxfield.txt")
    with open(coll_moxfield_txt, "w", encoding="utf-8") as f:
        for (name, set_code, collector_num, is_foil), count in sorted(collection_moxfield_counts.items(), key=lambda x: x[0][0]):
            f.write(f"{format_moxfield_line(count, name, set_code, collector_num, is_foil)}\n")

    manifest = {
        "scenario": "collection_scan",
        "directory": directory_path,
        "summary": {
            "total_images_processed": len(all_files),
            "total_cards_detected": total_cards_all,
            "total_cards_recognized": total_cards_all - total_unrecognized_all,
            "total_cards_unrecognized": total_unrecognized_all
        },
        "per_image_data": per_image_results
    }

    manifest_path = os.path.join(output_dir, "collection_summary.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n✨ Collection Scan Complete!")
    print(f"📊 Processed {len(all_files)} images, {total_cards_all} cards detected.")
    print(f"📄 Moxfield Collection Export: {coll_moxfield_txt}")
    print(f"📁 Manifest exported to: {manifest_path}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="MTG Vision & Recognition Suite with Moxfield Export")
    subparsers = parser.add_subparsers(dest="scenario", required=True, help="Processing Mode")

    # Scenario 1: Deck Scan
    deck_parser = subparsers.add_parser("deck", help="Scan deck image and generate formatted decklists")
    deck_parser.add_argument("image", type=str, help="Path to deck photo")
    deck_parser.add_argument("--out", type=str, default="output_deck", help="Output directory")

    # Scenario 2: Cards Scan
    cards_parser = subparsers.add_parser("cards", help="Scan single image and extract all cards")
    cards_parser.add_argument("image", type=str, help="Path to card photo")
    cards_parser.add_argument("--out", type=str, default="output_cards", help="Output directory")

    # Scenario 3: Collection Scan
    coll_parser = subparsers.add_parser("collection", help="Scan directory of photos with per-image audit")
    coll_parser.add_argument("dir", type=str, help="Path to images directory")
    coll_parser.add_argument("--out", type=str, default="output_collection", help="Output directory")

    args = parser.parse_args()

    if args.scenario == "deck":
        scenario_deck_scan(args.image, args.out)
    elif args.scenario == "cards":
        scenario_cards_scan(args.image, args.out)
    elif args.scenario == "collection":
        scenario_collection_scan(args.dir, args.out)


if __name__ == "__main__":
    main()
print it (the moxfield format) in the output in terminal as well

Here is the updated script. The terminal output now displays a formatted Moxfield list for all three modes (deck, cards, and collection).

Python
import argparse
import glob
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import requests
from google import genai
from google.genai import types

# ==============================================================================
# Environment Configuration
# ==============================================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please define it in your .env file.")

MODEL_NAME = "gemini-2.5-flash"
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ==============================================================================
# Structured Output Schemas for Gemini Vision
# ==============================================================================
class CornerPoints(BaseModel):
    top_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    top_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")

class DetectedCard(BaseModel):
    card_name_raw: str = Field(..., description="Exact card name or visible header text")
    set_code: Optional[str] = Field(None, description="3-4 character set code if visible")
    collector_number: Optional[str] = Field(None, description="Collector number if visible")
    is_partially_obscured: bool = Field(False, description="True if stacked/overlapped card")
    is_foil: bool = Field(False, description="True if the card appears to be foil/holographic/shiny")
    box_2d: List[int] = Field(..., description="Bounding box [ymin, xmin, ymax, xmax] (0-1000)")
    corners: Optional[CornerPoints] = Field(None, description="Perspective corners (0-1000)")

class CardDetectionResult(BaseModel):
    cards: List[DetectedCard] = Field(default_factory=list, description="All detected cards")


# ==============================================================================
# Scryfall Matcher with Memory Cache
# ==============================================================================
class ScryfallEngine:
    BASE_URL = "https://api.scryfall.com"
    HEADERS = {"User-Agent": "MTGScannerSuite/3.2", "Accept": "application/json"}
    _cache: Dict[str, Optional[dict]] = {}

    @staticmethod
    def normalize_name(text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^\w\s\-\'/]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def match(cls, card: DetectedCard) -> Optional[dict]:
        clean_name = cls.normalize_name(card.card_name_raw)
        cache_key = f"{clean_name}|{card.set_code}|{card.collector_number}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        result = cls._resolve(card, clean_name)
        cls._cache[cache_key] = result
        return result

    @classmethod
    def _resolve(cls, card: DetectedCard, clean_name: str) -> Optional[dict]:
        # 1. Exact Set + Collector Number
        if card.set_code and card.collector_number:
            s_code = re.sub(r"[^a-zA-Z0-9]", "", card.set_code).lower()
            c_num = re.sub(r"[^a-zA-Z0-9]", "", card.collector_number)
            try:
                r = requests.get(f"{cls.BASE_URL}/cards/{s_code}/{c_num}", headers=cls.HEADERS, timeout=5)
                time.sleep(0.08)
                if r.status_code == 200:
                    return r.json()
            except requests.RequestException:
                pass

        if not clean_name:
            return None

        # 2. Fuzzy Search
        try:
            params = {"fuzzy": clean_name}
            if card.set_code:
                params["set"] = card.set_code.lower()
            r = requests.get(f"{cls.BASE_URL}/cards/named", params=params, headers=cls.HEADERS, timeout=5)
            time.sleep(0.08)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass

        # 3. Autocomplete Typo Fallback
        try:
            r = requests.get(f"{cls.BASE_URL}/cards/autocomplete", params={"q": clean_name}, headers=cls.HEADERS, timeout=5)
            time.sleep(0.08)
            if r.status_code == 200:
                suggestions = r.json().get("data", [])
                if suggestions:
                    r_top = requests.get(f"{cls.BASE_URL}/cards/named", params={"exact": suggestions[0]}, headers=cls.HEADERS, timeout=5)
                    time.sleep(0.08)
                    if r_top.status_code == 200:
                        return r_top.json()
        except requests.RequestException:
            pass

        return None


# ==============================================================================
# Moxfield Formatter Helper
# ==============================================================================
def format_moxfield_line(count: int, name: str, set_code: Optional[str], collector_number: Optional[str], is_foil: bool) -> str:
    parts = [f"{count} {name}"]
    if set_code:
        parts.append(f"({set_code.upper()})")
    if collector_number and set_code:
        parts.append(f"{collector_number}")
    if is_foil:
        parts.append("*F*")
    return " ".join(parts)


def print_moxfield_terminal(lines: List[str], header: str = "MOXFIELD IMPORT LIST"):
    print(f"\n{'='*20} {header} {'='*20}")
    if lines:
        for line in lines:
            print(line)
    else:
        print("(No matched cards to display)")
    print("=" * (42 + len(header)) + "\n")


# ==============================================================================
# Image Justification & Annotation
# ==============================================================================
def justify_card(image: np.ndarray, card: DetectedCard, output_path: str) -> str:
    h, w = image.shape[:2]
    out_w, out_h = 450, 628

    if card.corners and not card.is_partially_obscured:
        tl = [card.corners.top_left[1] * w / 1000.0, card.corners.top_left[0] * h / 1000.0]
        tr = [card.corners.top_right[1] * w / 1000.0, card.corners.top_right[0] * h / 1000.0]
        br = [card.corners.bottom_right[1] * w / 1000.0, card.corners.bottom_right[0] * h / 1000.0]
        bl = [card.corners.bottom_left[1] * w / 1000.0, card.corners.bottom_left[0] * h / 1000.0]

        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        dst_pts = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    else:
        ymin, xmin, ymax, xmax = card.box_2d
        y1, x1 = max(0, int(ymin * h / 1000.0)), max(0, int(xmin * w / 1000.0))
        y2, x2 = min(h, int(ymax * h / 1000.0)), min(w, int(xmax * w / 1000.0))
        cropped = image[y1:y2, x1:x2]
        warped = cropped if cropped.size > 0 else np.zeros((out_h, out_w, 3), dtype=np.uint8)

    cv2.imwrite(output_path, warped)
    return output_path


def annotate_image(image: np.ndarray, card: DetectedCard, matched_data: Optional[dict]):
    h, w = image.shape[:2]
    matched = matched_data is not None

    if matched:
        name = matched_data.get("name", "Card")
        price = matched_data.get("prices", {}).get("usd")
        foil_tag = " (Foil)" if card.is_foil else ""
        label = f"{name}{foil_tag} (${price})" if price else f"{name}{foil_tag}"
        color = (0, 200, 0)
    else:
        label = f"Unmatched: {card.card_name_raw or 'Unknown'}"
        color = (0, 0, 230)

    if card.corners and not card.is_partially_obscured:
        poly_pts = np.array([
            [int(card.corners.top_left[1] * w / 1000.0), int(card.corners.top_left[0] * h / 1000.0)],
            [int(card.corners.top_right[1] * w / 1000.0), int(card.corners.top_right[0] * h / 1000.0)],
            [int(card.corners.bottom_right[1] * w / 1000.0), int(card.corners.bottom_right[0] * h / 1000.0)],
            [int(card.corners.bottom_left[1] * w / 1000.0), int(card.corners.bottom_left[0] * h / 1000.0)],
        ], np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [poly_pts], isClosed=True, color=color, thickness=3)
    else:
        ymin, xmin, ymax, xmax = card.box_2d
        p1 = (int(xmin * w / 1000.0), int(ymin * h / 1000.0))
        p2 = (int(xmax * w / 1000.0), int(ymax * h / 1000.0))
        cv2.rectangle(image, p1, p2, color, 3)

    ymin, xmin = int(card.box_2d[0] * h / 1000.0), int(card.box_2d[1] * w / 1000.0)
    label_pos = (max(10, xmin), max(25, ymin - 8))
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(image, (label_pos[0], label_pos[1] - th - 4), (label_pos[0] + tw + 6, label_pos[1] + bl), color, -1)
    cv2.putText(image, label, (label_pos[0] + 3, label_pos[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# ==============================================================================
# Core Vision & Recognition Pipeline
# ==============================================================================
def process_single_image(image_path: str, output_dir: str, save_crops: bool = True) -> Tuple[List[dict], int, int]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    pil_img = Image.open(image_path)
    cv_img = cv2.imread(image_path)
    
    crops_dir = os.path.join(output_dir, f"{Path(image_path).stem}_crops")
    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    prompt = (
        "Identify all Magic: The Gathering cards in this photo (isolated or stacked/overlapping).\n"
        "Extract:\n"
        "1. Normalized box `box_2d` [ymin, xmin, ymax, xmax] (0-1000).\n"
        "2. Four outer `corners` in order: top_left, top_right, bottom_right, bottom_left (0-1000).\n"
        "3. Card title `card_name_raw`.\n"
        "4. 3-letter set code and collector number if visible in the bottom-left margin.\n"
        "5. `is_partially_obscured` (true if overlapping/stacked under another card).\n"
        "6. `is_foil` (true if the card surface has holographic sheen/foil reflection)."
    )

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[pil_img, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CardDetectionResult,
            temperature=0.1,
        ),
    )

    detection_data: CardDetectionResult = response.parsed
    annotated = cv_img.copy()
    processed_cards = []
    unrecognized_count = 0

    for idx, card in enumerate(detection_data.cards, start=1):
        justified_img_path = None
        if save_crops:
            crop_file = os.path.join(crops_dir, f"card_{idx:02d}.jpg")
            justified_img_path = justify_card(cv_img, card, crop_file)

        match = ScryfallEngine.match(card)
        if not match:
            unrecognized_count += 1

        annotate_image(annotated, card, match)

        processed_cards.append({
            "index": idx,
            "detected_raw": card.model_dump(),
            "justified_image": justified_img_path,
            "matched": match is not None,
            "scryfall": {
                "id": match.get("id"),
                "name": match.get("name"),
                "mana_cost": match.get("mana_cost"),
                "type_line": match.get("type_line"),
                "set": match.get("set"),
                "set_name": match.get("set_name"),
                "collector_number": match.get("collector_number"),
                "rarity": match.get("rarity"),
                "price_usd": match.get("prices", {}).get("usd"),
                "scryfall_uri": match.get("scryfall_uri")
            } if match else None
        })

    annotated_save_path = os.path.join(output_dir, f"{Path(image_path).stem}_annotated.jpg")
    cv2.imwrite(annotated_save_path, annotated)

    return processed_cards, len(detection_data.cards), unrecognized_count


# ==============================================================================
# Scenario 1: Deck Scan
# ==============================================================================
def scenario_deck_scan(image_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- [Scenario 1] Deck Scan: '{image_path}' ---")
    
    cards, total_detected, unrecognized = process_single_image(image_path, output_dir, save_crops=True)
    
    moxfield_counts: Dict[Tuple[str, Optional[str], Optional[str], bool], int] = {}
    standard_counts: Dict[str, int] = {}

    for c in cards:
        if c["matched"]:
            name = c["scryfall"]["name"]
            set_code = c["scryfall"].get("set")
            collector_num = c["scryfall"].get("collector_number")
            is_foil = c["detected_raw"].get("is_foil", False)

            key = (name, set_code, collector_num, is_foil)
            moxfield_counts[key] = moxfield_counts.get(key, 0) + 1
            standard_counts[name] = standard_counts.get(name, 0) + 1

    # Standard decklist.txt
    txt_path = os.path.join(output_dir, "decklist.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for name, count in sorted(standard_counts.items()):
            f.write(f"{count} {name}\n")

    # Moxfield decklist_moxfield.txt
    moxfield_lines = []
    moxfield_txt_path = os.path.join(output_dir, "decklist_moxfield.txt")
    with open(moxfield_txt_path, "w", encoding="utf-8") as f:
        for (name, set_code, collector_num, is_foil), count in sorted(moxfield_counts.items(), key=lambda x: x[0][0]):
            line = format_moxfield_line(count, name, set_code, collector_num, is_foil)
            moxfield_lines.append(line)
            f.write(f"{line}\n")

    # Terminal output of Moxfield lines
    print_moxfield_terminal(moxfield_lines, "DECK MOXFIELD LIST")

    json_payload = {
        "scenario": "deck_scan",
        "image": image_path,
        "total_cards_detected": total_detected,
        "recognized_cards_count": total_detected - unrecognized,
        "unrecognized_cards_count": unrecognized,
        "moxfield_decklist": moxfield_lines,
        "cards": cards
    }

    json_path = os.path.join(output_dir, "deck_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Moxfield Decklist: {moxfield_txt_path}")
    print(f"✅ Standard Decklist: {txt_path}")
    print(f"✅ Full scan details: {json_path}")


# ==============================================================================
# Scenario 2: Cards Scan
# ==============================================================================
def scenario_cards_scan(image_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- [Scenario 2] Single Image Cards Scan: '{image_path}' ---")
    
    cards, total_detected, unrecognized = process_single_image(image_path, output_dir, save_crops=True)

    moxfield_entries = []
    for c in cards:
        if c["matched"]:
            name = c["scryfall"]["name"]
            set_code = c["scryfall"].get("set")
            collector_num = c["scryfall"].get("collector_number")
            is_foil = c["detected_raw"].get("is_foil", False)
            moxfield_entries.append(format_moxfield_line(1, name, set_code, collector_num, is_foil))

    # Terminal output of Moxfield lines
    print_moxfield_terminal(moxfield_entries, "RECOGNIZED CARDS (MOXFIELD FORMAT)")

    json_payload = {
        "scenario": "cards_scan",
        "image": image_path,
        "total_detected": total_detected,
        "recognized_count": total_detected - unrecognized,
        "unrecognized_count": unrecognized,
        "moxfield_list": moxfield_entries,
        "cards": cards
    }

    json_path = os.path.join(output_dir, "cards_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Recognized {total_detected - unrecognized}/{total_detected} cards.")
    print(f"✅ Crops, annotated image, and JSON written to '{output_dir}/'")


# ==============================================================================
# Scenario 3: Collection Scan
# ==============================================================================
def scenario_collection_scan(directory_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- [Scenario 3] Collection Scan: Directory '{directory_path}' ---")

    all_files = [
        f for f in glob.glob(os.path.join(directory_path, "*"))
        if Path(f).suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    if not all_files:
        print(f"⚠️ No valid image files found in '{directory_path}'.")
        return

    per_image_results = []
    total_cards_all = 0
    total_unrecognized_all = 0
    collection_moxfield_counts: Dict[Tuple[str, Optional[str], Optional[str], bool], int] = {}

    for idx, img_path in enumerate(all_files, start=1):
        print(f"[{idx}/{len(all_files)}] Processing: {os.path.basename(img_path)}...")
        cards, total, unrec = process_single_image(img_path, output_dir, save_crops=True)
        
        recognized_cards = [c for c in cards if c["matched"]]
        
        for c in recognized_cards:
            name = c["scryfall"]["name"]
            set_code = c["scryfall"].get("set")
            collector_num = c["scryfall"].get("collector_number")
            is_foil = c["detected_raw"].get("is_foil", False)
            key = (name, set_code, collector_num, is_foil)
            collection_moxfield_counts[key] = collection_moxfield_counts.get(key, 0) + 1

        per_image_results.append({
            "image_filename": os.path.basename(img_path),
            "image_path": img_path,
            "total_detected": total,
            "recognized_cards": [c["scryfall"]["name"] for c in recognized_cards],
            "recognized_cards_details": recognized_cards,
            "number_of_unrecognized_cards": unrec,
        })
        
        total_cards_all += total
        total_unrecognized_all += unrec

    # Save and print complete collection Moxfield list
    coll_moxfield_lines = [
        format_moxfield_line(count, name, set_code, collector_num, is_foil)
        for (name, set_code, collector_num, is_foil), count in sorted(collection_moxfield_counts.items(), key=lambda x: x[0][0])
    ]

    coll_moxfield_txt = os.path.join(output_dir, "collection_moxfield.txt")
    with open(coll_moxfield_txt, "w", encoding="utf-8") as f:
        for line in coll_moxfield_lines:
            f.write(f"{line}\n")

    print_moxfield_terminal(coll_moxfield_lines, "COLLECTION MOXFIELD LIST")

    manifest = {
        "scenario": "collection_scan",
        "directory": directory_path,
        "summary": {
            "total_images_processed": len(all_files),
            "total_cards_detected": total_cards_all,
            "total_cards_recognized": total_cards_all - total_unrecognized_all,
            "total_cards_unrecognized": total_unrecognized_all
        },
        "per_image_data": per_image_results
    }

    manifest_path = os.path.join(output_dir, "collection_summary.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"✨ Collection Scan Complete!")
    print(f"📊 Processed {len(all_files)} images, {total_cards_all} cards detected.")
    print(f"📄 Moxfield Collection Export: {coll_moxfield_txt}")
    print(f"📁 Manifest exported to: {manifest_path}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="MTG Vision & Recognition Suite with Moxfield Export")
    subparsers = parser.add_subparsers(dest="scenario", required=True, help="Processing Mode")

    # Scenario 1: Deck Scan
    deck_parser = subparsers.add_parser("deck", help="Scan deck image and generate formatted decklists")
    deck_parser.add_argument("image", type=str, help="Path to deck photo")
    deck_parser.add_argument("--out", type=str, default="output_deck", help="Output directory")

    # Scenario 2: Cards Scan
    cards_parser = subparsers.add_parser("cards", help="Scan single image and extract all cards")
    cards_parser.add_argument("image", type=str, help="Path to card photo")
    cards_parser.add_argument("--out", type=str, default="output_cards", help="Output directory")

    # Scenario 3: Collection Scan
    coll_parser = subparsers.add_parser("collection", help="Scan directory of photos with per-image audit")
    coll_parser.add_argument("dir", type=str, help="Path to images directory")
    coll_parser.add_argument("--out", type=str, default="output_collection", help="Output directory")

    args = parser.parse_args()

    if args.scenario == "deck":
        scenario_deck_scan(args.image, args.out)
    elif args.scenario == "cards":
        scenario_cards_scan(args.image, args.out)
    elif args.scenario == "collection":
        scenario_collection_scan(args.dir, args.out)


if __name__ == "__main__":
    main()
some changes: model is version 3.6, not 2.5

also i get : Direct use of automatic function calling (AFC) in Models.generate_content is not recommended. Instead, we recommend to use AFC in Chat.send_message. Similarly, direct use of AFC in Models.generate_content_stream is not recommended. Instead, we recommend to use AFC in Chat.send_message_stream.

To resolve the automatic function calling (AFC) deprecation warning and update the model version, two changes are made:  
Medium

Model Version: Set to "gemini-3.6-flash" (configurable via the .env file as GEMINI_MODEL).

Chat-Based Execution: Switched from client.models.generate_content(...) to client.chats.create(...) + chat.send_message(...), which adheres to the SDK's recommended execution pattern for single and multi-turn schema generation without triggering the AFC warning.  
GitHub

Updated Script (mtg_scanner.py)
Python
import argparse
import glob
import json
import os
import re
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from dotenv import load_dotenv
import numpy as np
from PIL import Image
from pydantic import BaseModel, Field
import requests
from google import genai
from google.genai import types

# ==============================================================================
# Environment Configuration
# ==============================================================================
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found. Please define it in your .env file.")

# Model configuration
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ==============================================================================
# Structured Output Schemas for Gemini Vision
# ==============================================================================
class CornerPoints(BaseModel):
    top_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    top_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")

class DetectedCard(BaseModel):
    card_name_raw: str = Field(..., description="Exact card name or visible header text")
    set_code: Optional[str] = Field(None, description="3-4 character set code if visible")
    collector_number: Optional[str] = Field(None, description="Collector number if visible")
    is_partially_obscured: bool = Field(False, description="True if stacked/overlapped card")
    is_foil: bool = Field(False, description="True if the card appears to be foil/holographic/shiny")
    box_2d: List[int] = Field(..., description="Bounding box [ymin, xmin, ymax, xmax] (0-1000)")
    corners: Optional[CornerPoints] = Field(None, description="Perspective corners (0-1000)")

class CardDetectionResult(BaseModel):
    cards: List[DetectedCard] = Field(default_factory=list, description="All detected cards")


# ==============================================================================
# Scryfall Matcher with Memory Cache
# ==============================================================================
class ScryfallEngine:
    BASE_URL = "https://api.scryfall.com"
    HEADERS = {"User-Agent": "MTGScannerSuite/3.3", "Accept": "application/json"}
    _cache: Dict[str, Optional[dict]] = {}

    @staticmethod
    def normalize_name(text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
        text = re.sub(r"[^\w\s\-\'/]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def match(cls, card: DetectedCard) -> Optional[dict]:
        clean_name = cls.normalize_name(card.card_name_raw)
        cache_key = f"{clean_name}|{card.set_code}|{card.collector_number}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        result = cls._resolve(card, clean_name)
        cls._cache[cache_key] = result
        return result

    @classmethod
    def _resolve(cls, card: DetectedCard, clean_name: str) -> Optional[dict]:
        # 1. Exact Set + Collector Number
        if card.set_code and card.collector_number:
            s_code = re.sub(r"[^a-zA-Z0-9]", "", card.set_code).lower()
            c_num = re.sub(r"[^a-zA-Z0-9]", "", card.collector_number)
            try:
                r = requests.get(f"{cls.BASE_URL}/cards/{s_code}/{c_num}", headers=cls.HEADERS, timeout=5)
                time.sleep(0.08)
                if r.status_code == 200:
                    return r.json()
            except requests.RequestException:
                pass

        if not clean_name:
            return None

        # 2. Fuzzy Search
        try:
            params = {"fuzzy": clean_name}
            if card.set_code:
                params["set"] = card.set_code.lower()
            r = requests.get(f"{cls.BASE_URL}/cards/named", params=params, headers=cls.HEADERS, timeout=5)
            time.sleep(0.08)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass

        # 3. Autocomplete Typo Fallback
        try:
            r = requests.get(f"{cls.BASE_URL}/cards/autocomplete", params={"q": clean_name}, headers=cls.HEADERS, timeout=5)
            time.sleep(0.08)
            if r.status_code == 200:
                suggestions = r.json().get("data", [])
                if suggestions:
                    r_top = requests.get(f"{cls.BASE_URL}/cards/named", params={"exact": suggestions[0]}, headers=cls.HEADERS, timeout=5)
                    time.sleep(0.08)
                    if r_top.status_code == 200:
                        return r_top.json()
        except requests.RequestException:
            pass

        return None


# ==============================================================================
# Moxfield Formatter Helpers
# ==============================================================================
def format_moxfield_line(count: int, name: str, set_code: Optional[str], collector_number: Optional[str], is_foil: bool) -> str:
    parts = [f"{count} {name}"]
    if set_code:
        parts.append(f"({set_code.upper()})")
    if collector_number and set_code:
        parts.append(f"{collector_number}")
    if is_foil:
        parts.append("*F*")
    return " ".join(parts)


def print_moxfield_terminal(lines: List[str], header: str = "MOXFIELD IMPORT LIST"):
    print(f"\n{'='*20} {header} {'='*20}")
    if lines:
        for line in lines:
            print(line)
    else:
        print("(No matched cards to display)")
    print("=" * (42 + len(header)) + "\n")


# ==============================================================================
# Image Justification & Annotation
# ==============================================================================
def justify_card(image: np.ndarray, card: DetectedCard, output_path: str) -> str:
    h, w = image.shape[:2]
    out_w, out_h = 450, 628

    if card.corners and not card.is_partially_obscured:
        tl = [card.corners.top_left[1] * w / 1000.0, card.corners.top_left[0] * h / 1000.0]
        tr = [card.corners.top_right[1] * w / 1000.0, card.corners.top_right[0] * h / 1000.0]
        br = [card.corners.bottom_right[1] * w / 1000.0, card.corners.bottom_right[0] * h / 1000.0]
        bl = [card.corners.bottom_left[1] * w / 1000.0, card.corners.bottom_left[0] * h / 1000.0]

        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        dst_pts = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(image, matrix, (out_w, out_h))
    else:
        ymin, xmin, ymax, xmax = card.box_2d
        y1, x1 = max(0, int(ymin * h / 1000.0)), max(0, int(xmin * w / 1000.0))
        y2, x2 = min(h, int(ymax * h / 1000.0)), min(w, int(xmax * w / 1000.0))
        cropped = image[y1:y2, x1:x2]
        warped = cropped if cropped.size > 0 else np.zeros((out_h, out_w, 3), dtype=np.uint8)

    cv2.imwrite(output_path, warped)
    return output_path


def annotate_image(image: np.ndarray, card: DetectedCard, matched_data: Optional[dict]):
    h, w = image.shape[:2]
    matched = matched_data is not None

    if matched:
        name = matched_data.get("name", "Card")
        price = matched_data.get("prices", {}).get("usd")
        foil_tag = " (Foil)" if card.is_foil else ""
        label = f"{name}{foil_tag} (${price})" if price else f"{name}{foil_tag}"
        color = (0, 200, 0)
    else:
        label = f"Unmatched: {card.card_name_raw or 'Unknown'}"
        color = (0, 0, 230)

    if card.corners and not card.is_partially_obscured:
        poly_pts = np.array([
            [int(card.corners.top_left[1] * w / 1000.0), int(card.corners.top_left[0] * h / 1000.0)],
            [int(card.corners.top_right[1] * w / 1000.0), int(card.corners.top_right[0] * h / 1000.0)],
            [int(card.corners.bottom_right[1] * w / 1000.0), int(card.corners.bottom_right[0] * h / 1000.0)],
            [int(card.corners.bottom_left[1] * w / 1000.0), int(card.corners.bottom_left[0] * h / 1000.0)],
        ], np.int32).reshape((-1, 1, 2))
        cv2.polylines(image, [poly_pts], isClosed=True, color=color, thickness=3)
    else:
        ymin, xmin, ymax, xmax = card.box_2d
        p1 = (int(xmin * w / 1000.0), int(ymin * h / 1000.0))
        p2 = (int(xmax * w / 1000.0), int(ymax * h / 1000.0))
        cv2.rectangle(image, p1, p2, color, 3)

    ymin, xmin = int(card.box_2d[0] * h / 1000.0), int(card.box_2d[1] * w / 1000.0)
    label_pos = (max(10, xmin), max(25, ymin - 8))
    (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(image, (label_pos[0], label_pos[1] - th - 4), (label_pos[0] + tw + 6, label_pos[1] + bl), color, -1)
    cv2.putText(image, label, (label_pos[0] + 3, label_pos[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)


# ==============================================================================
# Core Vision & Recognition Pipeline
# ==============================================================================
def process_single_image(image_path: str, output_dir: str, save_crops: bool = True) -> Tuple[List[dict], int, int]:
    client = genai.Client(api_key=GEMINI_API_KEY)
    pil_img = Image.open(image_path)
    cv_img = cv2.imread(image_path)
    
    crops_dir = os.path.join(output_dir, f"{Path(image_path).stem}_crops")
    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    prompt = (
        "Identify all Magic: The Gathering cards in this photo (isolated or stacked/overlapping).\n"
        "Extract:\n"
        "1. Normalized box `box_2d` [ymin, xmin, ymax, xmax] (0-1000).\n"
        "2. Four outer `corners` in order: top_left, top_right, bottom_right, bottom_left (0-1000).\n"
        "3. Card title `card_name_raw`.\n"
        "4. 3-letter set code and collector number if visible in the bottom-left margin.\n"
        "5. `is_partially_obscured` (true if overlapping/stacked under another card).\n"
        "6. `is_foil` (true if the card surface has holographic sheen/foil reflection)."
    )

    # Use Chat.send_message to eliminate AFC warning
    chat = client.chats.create(
        model=MODEL_NAME,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=CardDetectionResult,
            temperature=0.1,
        ),
    )
    
    response = chat.send_message(message=[pil_img, prompt])
    detection_data: CardDetectionResult = response.parsed
    
    annotated = cv_img.copy()
    processed_cards = []
    unrecognized_count = 0

    for idx, card in enumerate(detection_data.cards, start=1):
        justified_img_path = None
        if save_crops:
            crop_file = os.path.join(crops_dir, f"card_{idx:02d}.jpg")
            justified_img_path = justify_card(cv_img, card, crop_file)

        match = ScryfallEngine.match(card)
        if not match:
            unrecognized_count += 1

        annotate_image(annotated, card, match)

        processed_cards.append({
            "index": idx,
            "detected_raw": card.model_dump(),
            "justified_image": justified_img_path,
            "matched": match is not None,
            "scryfall": {
                "id": match.get("id"),
                "name": match.get("name"),
                "mana_cost": match.get("mana_cost"),
                "type_line": match.get("type_line"),
                "set": match.get("set"),
                "set_name": match.get("set_name"),
                "collector_number": match.get("collector_number"),
                "rarity": match.get("rarity"),
                "price_usd": match.get("prices", {}).get("usd"),
                "scryfall_uri": match.get("scryfall_uri")
            } if match else None
        })

    annotated_save_path = os.path.join(output_dir, f"{Path(image_path).stem}_annotated.jpg")
    cv2.imwrite(annotated_save_path, annotated)

    return processed_cards, len(detection_data.cards), unrecognized_count


# ==============================================================================
# Scenario 1: Deck Scan
# ==============================================================================
def scenario_deck_scan(image_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- [Scenario 1] Deck Scan ({MODEL_NAME}): '{image_path}' ---")
    
    cards, total_detected, unrecognized = process_single_image(image_path, output_dir, save_crops=True)
    
    moxfield_counts: Dict[Tuple[str, Optional[str], Optional[str], bool], int] = {}
    standard_counts: Dict[str, int] = {}

    for c in cards:
        if c["matched"]:
            name = c["scryfall"]["name"]
            set_code = c["scryfall"].get("set")
            collector_num = c["scryfall"].get("collector_number")
            is_foil = c["detected_raw"].get("is_foil", False)

            key = (name, set_code, collector_num, is_foil)
            moxfield_counts[key] = moxfield_counts.get(key, 0) + 1
            standard_counts[name] = standard_counts.get(name, 0) + 1

    txt_path = os.path.join(output_dir, "decklist.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for name, count in sorted(standard_counts.items()):
            f.write(f"{count} {name}\n")

    moxfield_lines = []
    moxfield_txt_path = os.path.join(output_dir, "decklist_moxfield.txt")
    with open(moxfield_txt_path, "w", encoding="utf-8") as f:
        for (name, set_code, collector_num, is_foil), count in sorted(moxfield_counts.items(), key=lambda x: x[0][0]):
            line = format_moxfield_line(count, name, set_code, collector_num, is_foil)
            moxfield_lines.append(line)
            f.write(f"{line}\n")

    print_moxfield_terminal(moxfield_lines, "DECK MOXFIELD LIST")

    json_payload = {
        "scenario": "deck_scan",
        "image": image_path,
        "total_cards_detected": total_detected,
        "recognized_cards_count": total_detected - unrecognized,
        "unrecognized_cards_count": unrecognized,
        "moxfield_decklist": moxfield_lines,
        "cards": cards
    }

    json_path = os.path.join(output_dir, "deck_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Moxfield Decklist: {moxfield_txt_path}")
    print(f"✅ Standard Decklist: {txt_path}")
    print(f"✅ Full scan details: {json_path}")


# ==============================================================================
# Scenario 2: Cards Scan
# ==============================================================================
def scenario_cards_scan(image_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- [Scenario 2] Single Image Cards Scan ({MODEL_NAME}): '{image_path}' ---")
    
    cards, total_detected, unrecognized = process_single_image(image_path, output_dir, save_crops=True)

    moxfield_entries = []
    for c in cards:
        if c["matched"]:
            name = c["scryfall"]["name"]
            set_code = c["scryfall"].get("set")
            collector_num = c["scryfall"].get("collector_number")
            is_foil = c["detected_raw"].get("is_foil", False)
            moxfield_entries.append(format_moxfield_line(1, name, set_code, collector_num, is_foil))

    print_moxfield_terminal(moxfield_entries, "RECOGNIZED CARDS (MOXFIELD FORMAT)")

    json_payload = {
        "scenario": "cards_scan",
        "image": image_path,
        "total_detected": total_detected,
        "recognized_count": total_detected - unrecognized,
        "unrecognized_count": unrecognized,
        "moxfield_list": moxfield_entries,
        "cards": cards
    }

    json_path = os.path.join(output_dir, "cards_result.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, ensure_ascii=False)

    print(f"✅ Recognized {total_detected - unrecognized}/{total_detected} cards.")
    print(f"✅ Crops, annotated image, and JSON written to '{output_dir}/'")


# ==============================================================================
# Scenario 3: Collection Scan
# ==============================================================================
def scenario_collection_scan(directory_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- [Scenario 3] Collection Scan ({MODEL_NAME}): Directory '{directory_path}' ---")

    all_files = [
        f for f in glob.glob(os.path.join(directory_path, "*"))
        if Path(f).suffix.lower() in VALID_IMAGE_EXTENSIONS
    ]

    if not all_files:
        print(f"⚠️ No valid image files found in '{directory_path}'.")
        return

    per_image_results = []
    total_cards_all = 0
    total_unrecognized_all = 0
    collection_moxfield_counts: Dict[Tuple[str, Optional[str], Optional[str], bool], int] = {}

    for idx, img_path in enumerate(all_files, start=1):
        print(f"[{idx}/{len(all_files)}] Processing: {os.path.basename(img_path)}...")
        cards, total, unrec = process_single_image(img_path, output_dir, save_crops=True)
        
        recognized_cards = [c for c in cards if c["matched"]]
        
        for c in recognized_cards:
            name = c["scryfall"]["name"]
            set_code = c["scryfall"].get("set")
            collector_num = c["scryfall"].get("collector_number")
            is_foil = c["detected_raw"].get("is_foil", False)
            key = (name, set_code, collector_num, is_foil)
            collection_moxfield_counts[key] = collection_moxfield_counts.get(key, 0) + 1

        per_image_results.append({
            "image_filename": os.path.basename(img_path),
            "image_path": img_path,
            "total_detected": total,
            "recognized_cards": [c["scryfall"]["name"] for c in recognized_cards],
            "recognized_cards_details": recognized_cards,
            "number_of_unrecognized_cards": unrec,
        })
        
        total_cards_all += total
        total_unrecognized_all += unrec

    coll_moxfield_lines = [
        format_moxfield_line(count, name, set_code, collector_num, is_foil)
        for (name, set_code, collector_num, is_foil), count in sorted(collection_moxfield_counts.items(), key=lambda x: x[0][0])
    ]

    coll_moxfield_txt = os.path.join(output_dir, "collection_moxfield.txt")
    with open(coll_moxfield_txt, "w", encoding="utf-8") as f:
        for line in coll_moxfield_lines:
            f.write(f"{line}\n")

    print_moxfield_terminal(coll_moxfield_lines, "COLLECTION MOXFIELD LIST")

    manifest = {
        "scenario": "collection_scan",
        "directory": directory_path,
        "summary": {
            "total_images_processed": len(all_files),
            "total_cards_detected": total_cards_all,
            "total_cards_recognized": total_cards_all - total_unrecognized_all,
            "total_cards_unrecognized": total_unrecognized_all
        },
        "per_image_data": per_image_results
    }

    manifest_path = os.path.join(output_dir, "collection_summary.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\n✨ Collection Scan Complete!")
    print(f"📊 Processed {len(all_files)} images, {total_cards_all} cards detected.")
    print(f"📄 Moxfield Collection Export: {coll_moxfield_txt}")
    print(f"📁 Manifest exported to: {manifest_path}")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="MTG Vision & Recognition Suite (Chat API + Moxfield)")
    subparsers = parser.add_subparsers(dest="scenario", required=True, help="Processing Mode")

    # Scenario 1: Deck Scan
    deck_parser = subparsers.add_parser("deck", help="Scan deck image and generate formatted decklists")
    deck_parser.add_argument("image", type=str, help="Path to deck photo")
    deck_parser.add_argument("--out", type=str, default="output_deck", help="Output directory")

    # Scenario 2: Cards Scan
    cards_parser = subparsers.add_parser("cards", help="Scan single image and extract all cards")
    cards_parser.add_argument("image", type=str, help="Path to card photo")
    cards_parser.add_argument("--out", type=str, default="output_cards", help="Output directory")

    # Scenario 3: Collection Scan
    coll_parser = subparsers.add_parser("collection", help="Scan directory of photos with per-image audit")
    coll_parser.add_argument("dir", type=str, help="Path to images directory")
    coll_parser.add_argument("--out", type=str, default="output_collection", help="Output directory")

    args = parser.parse_args()

    if args.scenario == "deck":
        scenario_deck_scan(args.image, args.out)
    elif args.scenario == "cards":
        scenario_cards_scan(args.image, args.out)
    elif args.scenario == "collection":
        scenario_collection_scan(args.dir, args.out)


if __name__ == "__main__":
    main()
link the "overlay image" of im2 in the readme (for display on github)

Markdown
# MTG Card Scanner & Recognition Suite

An automated Magic: The Gathering card recognition pipeline powered by **Gemini 3.6 Flash Vision**, **OpenCV** perspective justification, and the **Scryfall API**.

The tool supports smartphone camera shots of isolated cards, binder pages, playmats, and stacked/cascaded deck piles across 3 primary operational modes.

---

## Visual Example: Detection & Overlay

Below is an example of the generated overlay showing recognized cards (green bounding boxes with live Scryfall pricing and foil tags) alongside unrecognized cards (red bounding boxes):

<p align="center">
  <img src="output_cards/im2_annotated.jpg" alt="MTG Card Recognition Overlay" width="750" />
</p>

---

## Features

- **Multi-Card Detection:** Detects single or multiple cards within a single image.
- **Perspective Justification:** Automatically detects card corners and warps skewed smartphone angles into rectified 450x628 crops.
- **Overlapping/Stacked Card Handling:** Detects partially visible cards in deck piles.
- **Hybrid Identification:** Combines exact Set Code + Collector Number matching, fuzzy string searching, and Scryfall autocomplete fallback.
- **Visual Validation:** Generates overview images with color-coded bounding polygons/boxes (Green = Matched, Red = Unmatched) and price overlays.
- **Moxfield Export:** Exports decklists formatted directly for Moxfield (e.g. `1 Sol Ring (MYS1) 187 *F*`) both to disk and straight to the terminal.
- **3 Application Modes:** Deck scanning (formatted text decklists), single-image card extraction, and bulk collection audits.

---

## Prerequisites

- Python 3.10+
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/mtg-card-scanner.git](https://github.com/yourusername/mtg-card-scanner.git)
cd mtg-card-scanner
2. Create and Activate a Virtual Environment
Bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
3. Install Required Dependencies
Bash
pip install google-genai opencv-python pillow pydantic requests python-dotenv
4. Configure Your API Key
Create a .env file in the root directory:

Bash
touch .env
Add your Gemini API key inside .env:

Code-Snippet
GEMINI_API_KEY="AIzaSyYourActualKeyGoesHere"
GEMINI_MODEL="gemini-3.6-flash"
Usage Guide
The script provides 3 distinct subcommands: deck, cards, and collection.

1. Deck Scan (deck)
Scans an image of a deck (including overlapping piles), aggregates card quantities, prints the Moxfield decklist directly in the terminal, and exports standard text and JSON manifests.

Bash
python mtg_scanner.py deck path/to/deck_photo.jpg --out output_deck
Outputs:

Terminal: Formatted Moxfield import list

output_deck/decklist_moxfield.txt: Moxfield format (e.g., 4 Lightning Bolt (CLB) 187)

output_deck/decklist.txt: Standard format (e.g., 4 Lightning Bolt)

output_deck/deck_result.json: Summary stats + structured card metadata

output_deck/<name>_annotated.jpg: Annotated image with price labels

output_deck/<name>_crops/: Individual justified card images

2. Cards Scan (cards)
Scans an image with one or more cards (e.g., binder page, spread on a table), justifies/crops each card, prints the Moxfield list, and resolves card details and prices.

Bash
python mtg_scanner.py cards path/to/im2.jpg --out output_cards
Outputs:

Terminal: Formatted list of all detected cards

output_cards/cards_result.json: Full card metadata and Scryfall IDs

output_cards/im2_annotated.jpg: Labeled bounding boxes/polygons

output_cards/im2_crops/card_01.jpg, card_02.jpg, ...

3. Collection Scan (collection)
Iterates over an entire directory of photos, runs the detection pipeline on each, prints the complete collection list, and exports a directory-wide summary audit showing recognized cards and unrecognized counts per image.

Bash
python mtg_scanner.py collection path/to/image_folder/ --out output_collection
Outputs:

Terminal: Aggregated collection-wide Moxfield import list

output_collection/collection_moxfield.txt: Full combined collection in Moxfield syntax

output_collection/collection_summary.json: Aggregated metrics across all images:

Total images processed

Total cards detected / recognized / unrecognized

Per-image breakdown of recognized card names and error counts

Annotated images and cropped cards for every photo in the input folder

JSON Output Structure Example
JSON
{
  "scenario": "deck_scan",
  "image": "photos/modern_burn.jpg",
  "total_cards_detected": 60,
  "recognized_cards_count": 59,
  "unrecognized_cards_count": 1,
  "moxfield_decklist": [
    "4 Lightning Bolt (CLB) 187",
    "4 Monastery Swiftspear (BRO) 144 *F*",
    "4 Rift Bolt (TSR) 188"
  ],
  "cards": [
    {
      "index": 1,
      "detected_raw": {
        "card_name_raw": "Lightning Bolt",
        "set_code": "CLB",
        "collector_number": "187",
        "is_partially_obscured": false,
        "is_foil": false,
        "box_2d": [120, 45, 410, 240]
      },
      "justified_image": "output_deck/modern_burn_crops/card_01.jpg",
      "matched": true,
      "scryfall": {
        "id": "f29ba16f-c8fb-42fe-aabf-87089cb214a7",
        "name": "Lightning Bolt",
        "mana_cost": "{R}",
        "type_line": "Instant",
        "set": "clb",
        "set_name": "Commander Legends: Battle for Baldur's Gate",
        "collector_number": "187",
        "rarity": "common",
        "price_usd": "0.74",
        "scryfall_uri": "[https://scryfall.com/card/clb/187/lightning-bolt](https://scryfall.com/card/clb/187/lightning-bolt)"
      }
    }
  ]
}
Project Structure
Plaintext
├── .env                       # Environment variable containing GEMINI_API_KEY
├── mtg_scanner.py             # Main CLI executable and processing engine
├── README.md                  # Documentation and setup instructions
└── output_cards/              # Destination for crops, JSON, and annotated photos
    ├── im2_annotated.jpg      # Labeled preview overlay image
    └── im2_crops/             # Rectified card crops
Troubleshooting & Tips
Perspective Distortion: For best results when photographing cards at steep angles, ensure all 4 card corners are in frame and well-lit.

Stacked Decks: When photographing overlapping cards, ensure the top title bar and mana cost are visible.

Scryfall Rate Limits: The script enforces a built-in 80ms delay between API queries and caches identical card lookups to respect Scryfall's API guidelines.

I need the md

2. Create and Activate a Virtual Environment
Bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\\Scripts\\activate.bat

# Windows (PowerShell)
python -m venv venv
venv\\Scripts\\Activate.ps1
3. Install Required Dependencies
Bash
pip install google-genai opencv-python pillow pydantic requests python-dotenv
4. Configure Your API Key
Create a .env file in the root directory:

Bash
touch .env
Add your Gemini API key inside .env:

Code-Snippet
GEMINI_API_KEY="AIzaSyYourActualKeyGoesHere"
GEMINI_MODEL="gemini-3.6-flash"
Usage Guide
The script provides 3 distinct subcommands: deck, cards, and collection.

1. Deck Scan (deck)
Scans an image of a deck (including overlapping piles), aggregates card quantities, prints the Moxfield decklist directly in the terminal, and exports standard text and JSON manifests.

Bash
python mtg_scanner.py deck path/to/deck_photo.jpg --out output_deck
Outputs:

Terminal: Formatted Moxfield import list

output_deck/decklist_moxfield.txt: Moxfield format (e.g., 4 Lightning Bolt (CLB) 187)

output_deck/decklist.txt: Standard format (e.g., 4 Lightning Bolt)

output_deck/deck_result.json: Summary stats + structured card metadata

output_deck/<name>_annotated.jpg: Annotated image with price labels

output_deck/<name>_crops/: Individual justified card images

2. Cards Scan (cards)
Scans an image with one or more cards (e.g., binder page, spread on a table), justifies/crops each card, prints the Moxfield list, and resolves card details and prices.

Bash
python mtg_scanner.py cards path/to/im2.jpg --out output_cards
Outputs:

Terminal: Formatted list of all detected cards

output_cards/cards_result.json: Full card metadata and Scryfall IDs

output_cards/im2_annotated.jpg: Labeled bounding boxes/polygons

output_cards/im2_crops/card_01.jpg, card_02.jpg, ...

3. Collection Scan (collection)
Iterates over an entire directory of photos, runs the detection pipeline on each, prints the complete collection list, and exports a directory-wide summary audit showing recognized cards and unrecognized counts per image.

Bash
python mtg_scanner.py collection path/to/image_folder/ --out output_collection
Outputs:

Terminal: Aggregated collection-wide Moxfield import list

output_collection/collection_moxfield.txt: Full combined collection in Moxfield syntax

output_collection/collection_summary.json: Aggregated metrics across all images:

Total images processed

Total cards detected / recognized / unrecognized

Per-image breakdown of recognized card names and error counts

Annotated images and cropped cards for every photo in the input folder

JSON Output Structure Example
JSON
{
  "scenario": "deck_scan",
  "image": "photos/modern_burn.jpg",
  "total_cards_detected": 60,
  "recognized_cards_count": 59,
  "unrecognized_cards_count": 1,
  "moxfield_decklist": [
    "4 Lightning Bolt (CLB) 187",
    "4 Monastery Swiftspear (BRO) 144 *F*",
    "4 Rift Bolt (TSR) 188"
  ],
  "cards": [
    {
      "index": 1,
      "detected_raw": {
        "card_name_raw": "Lightning Bolt",
        "set_code": "CLB",
        "collector_number": "187",
        "is_partially_obscured": false,
        "is_foil": false,
        "box_2d": [120, 45, 410, 240]
      },
      "justified_image": "output_deck/modern_burn_crops/card_01.jpg",
      "matched": true,
      "scryfall": {
        "id": "f29ba16f-c8fb-42fe-aabf-87089cb214a7",
        "name": "Lightning Bolt",
        "mana_cost": "{R}",
        "type_line": "Instant",
        "set": "clb",
        "set_name": "Commander Legends: Battle for Baldur's Gate",
        "collector_number": "187",
        "rarity": "common",
        "price_usd": "0.74",
        "scryfall_uri": "[https://scryfall.com/card/clb/187/lightning-bolt](https://scryfall.com/card/clb/187/lightning-bolt)"
      }
    }
  ]
}
Project Structure
Plaintext
├── .env                       # Environment variable containing GEMINI_API_KEY
├── mtg_scanner.py             # Main CLI executable and processing engine
├── README.md                  # Documentation and setup instructions
└── output_cards/              # Destination for crops, JSON, and annotated photos
    ├── im2_annotated.jpg      # Labeled preview overlay image
    └── im2_crops/             # Rectified card crops
Troubleshooting & Tips
Perspective Distortion: For best results when photographing cards at steep angles, ensure all 4 card corners are in frame and well-lit.

Stacked Decks: When photographing overlapping cards, ensure the top title bar and mana cost are visible.

Scryfall Rate Limits: The script enforces a built-in 80ms delay between API queries and caches identical card lookups to respect Scryfall's API guidelines.
"""

file_path = "README.md"
with open(file_path, "w", encoding="utf-8") as f:
f.write(readme_content)

print(f"File created successfully: {file_path}")


```text?code_stdout&code_event_index=1
File created successfully: README.md

Your Markdown file is ready:

MD-Symbol
README
 MD 
Geöffnet
You can download README.md directly via the file chip above.






Gemini ist eine KI und kann Fehler machen, auch bei Informationen über Personen. Datenschutz und GeminiWird in einem neuen Fenster geöffnet

Wird analysiert
# MTG Card Scanner & Recognition Suite

An automated Magic: The Gathering card recognition pipeline powered by **Gemini 3.6 Flash Vision**, **OpenCV** perspective justification, and the **Scryfall API**.

The tool supports smartphone camera shots of isolated cards, binder pages, playmats, and stacked/cascaded deck piles across 3 primary operational modes.

---

## Visual Example: Detection & Overlay

Below is an example of the generated overlay showing recognized cards (green bounding boxes with live Scryfall pricing and foil tags) alongside unrecognized cards (red bounding boxes):

<p align="center">
  <img src="output_cards/im2_annotated.jpg" alt="MTG Card Recognition Overlay" width="750" />
</p>

---

## Features

- **Multi-Card Detection:** Detects single or multiple cards within a single image.
- **Perspective Justification:** Automatically detects card corners and warps skewed smartphone angles into rectified 450x628 crops.
- **Overlapping/Stacked Card Handling:** Detects partially visible cards in deck piles.
- **Hybrid Identification:** Combines exact Set Code + Collector Number matching, fuzzy string searching, and Scryfall autocomplete fallback.
- **Visual Validation:** Generates overview images with color-coded bounding polygons/boxes (Green = Matched, Red = Unmatched) and price overlays.
- **Moxfield Export:** Exports decklists formatted directly for Moxfield (e.g. `1 Sol Ring (MYS1) 187 *F*`) both to disk and straight to the terminal.
- **3 Application Modes:** Deck scanning (formatted text decklists), single-image card extraction, and bulk collection audits.

---

## Prerequisites

- Python 3.10+
- A Google Gemini API key ([Google AI Studio](https://aistudio.google.com/))

---

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/mtg-card-scanner.git
cd mtg-card-scanner
```

### 2. Create and Activate a Virtual Environment
```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

### 3. Install Required Dependencies
```bash
pip install google-genai opencv-python pillow pydantic requests python-dotenv
```

### 4. Configure Your API Key
Create a `.env` file in the root directory:
```bash
touch .env
```
Add your Gemini API key inside `.env`:
```env
GEMINI_API_KEY="AIzaSyYourActualKeyGoesHere"
GEMINI_MODEL="gemini-3.6-flash"
```

---

## Usage Guide

The script provides 3 distinct subcommands: `deck`, `cards`, and `collection`.

### 1. Deck Scan (`deck`)
Scans an image of a deck (including overlapping piles), aggregates card quantities, prints the Moxfield decklist directly in the terminal, and exports standard text and JSON manifests.

```bash
python mtg_scanner.py deck path/to/deck_photo.jpg --out output_deck
```

**Outputs:**
- Terminal: Formatted Moxfield import list
- `output_deck/decklist_moxfield.txt`: Moxfield format (e.g., `4 Lightning Bolt (CLB) 187`)
- `output_deck/decklist.txt`: Standard format (e.g., `4 Lightning Bolt`)
- `output_deck/deck_result.json`: Summary stats + structured card metadata
- `output_deck/<name>_annotated.jpg`: Annotated image with price labels
- `output_deck/<name>_crops/`: Individual justified card images

---

### 2. Cards Scan (`cards`)
Scans an image with one or more cards (e.g., binder page, spread on a table), justifies/crops each card, prints the Moxfield list, and resolves card details and prices.

```bash
python mtg_scanner.py cards path/to/im2.jpg --out output_cards
```

**Outputs:**
- Terminal: Formatted list of all detected cards
- `output_cards/cards_result.json`: Full card metadata and Scryfall IDs
- `output_cards/im2_annotated.jpg`: Labeled bounding boxes/polygons
- `output_cards/im2_crops/card_01.jpg`, `card_02.jpg`, ...

---

### 3. Collection Scan (`collection`)
Iterates over an entire directory of photos, runs the detection pipeline on each, prints the complete collection list, and exports a directory-wide summary audit showing recognized cards and unrecognized counts per image.

```bash
python mtg_scanner.py collection path/to/image_folder/ --out output_collection
```

**Outputs:**
- Terminal: Aggregated collection-wide Moxfield import list
- `output_collection/collection_moxfield.txt`: Full combined collection in Moxfield syntax
- `output_collection/collection_summary.json`: Aggregated metrics across all images:
  - Total images processed
  - Total cards detected / recognized / unrecognized
  - Per-image breakdown of recognized card names and error counts
- Annotated images and cropped cards for every photo in the input folder

---

## JSON Output Structure Example

```json
{
  "scenario": "deck_scan",
  "image": "photos/modern_burn.jpg",
  "total_cards_detected": 60,
  "recognized_cards_count": 59,
  "unrecognized_cards_count": 1,
  "moxfield_decklist": [
    "4 Lightning Bolt (CLB) 187",
    "4 Monastery Swiftspear (BRO) 144 *F*",
    "4 Rift Bolt (TSR) 188"
  ],
  "cards": [
    {
      "index": 1,
      "detected_raw": {
        "card_name_raw": "Lightning Bolt",
        "set_code": "CLB",
        "collector_number": "187",
        "is_partially_obscured": false,
        "is_foil": false,
        "box_2d": [120, 45, 410, 240]
      },
      "justified_image": "output_deck/modern_burn_crops/card_01.jpg",
      "matched": true,
      "scryfall": {
        "id": "f29ba16f-c8fb-42fe-aabf-87089cb214a7",
        "name": "Lightning Bolt",
        "mana_cost": "{R}",
        "type_line": "Instant",
        "set": "clb",
        "set_name": "Commander Legends: Battle for Baldur's Gate",
        "collector_number": "187",
        "rarity": "common",
        "price_usd": "0.74",
        "scryfall_uri": "https://scryfall.com/card/clb/187/lightning-bolt"
      }
    }
  ]
}
```

---

## Project Structure

```text
├── .env                       # Environment variable containing GEMINI_API_KEY
├── mtg_scanner.py             # Main CLI executable and processing engine
├── README.md                  # Documentation and setup instructions
└── output_cards/              # Destination for crops, JSON, and annotated photos
    ├── im2_annotated.jpg      # Labeled preview overlay image
    └── im2_crops/             # Rectified card crops
```

---

## Troubleshooting & Tips

- **Perspective Distortion:** For best results when photographing cards at steep angles, ensure all 4 card corners are in frame and well-lit.
- **Stacked Decks:** When photographing overlapping cards, ensure the top title bar and mana cost are visible.
- **Scryfall Rate Limits:** The script enforces a built-in 80ms delay between API queries and caches identical card lookups to respect Scryfall's API guidelines.
README.md
README.md wird angezeigt.
