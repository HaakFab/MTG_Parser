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

MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ==============================================================================
# Structured Output Schemas for Gemini Vision (Multilingual Aware)
# ==============================================================================
class CornerPoints(BaseModel):
    top_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    top_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_right: List[int] = Field(..., description="[y, x] normalized (0-1000)")
    bottom_left: List[int] = Field(..., description="[y, x] normalized (0-1000)")

class DetectedCard(BaseModel):
    card_name_raw: str = Field(..., description="Exact card name printed on the card (German, English, etc.)")
    card_name_en: Optional[str] = Field(None, description="Official English Oracle card name")
    set_code: Optional[str] = Field(None, description="3-4 character set code if visible")
    collector_number: Optional[str] = Field(None, description="Collector number if visible")
    is_partially_obscured: bool = Field(False, description="True if stacked/overlapped card")
    is_foil: bool = Field(False, description="True if the card appears to be foil/holographic/shiny")
    box_2d: List[int] = Field(..., description="Bounding box [ymin, xmin, ymax, xmax] (0-1000)")
    corners: Optional[CornerPoints] = Field(None, description="Perspective corners (0-1000)")

class CardDetectionResult(BaseModel):
    cards: List[DetectedCard] = Field(default_factory=list, description="All detected cards")


# ==============================================================================
# Multilingual Scryfall Engine
# ==============================================================================
class ScryfallEngine:
    BASE_URL = "https://api.scryfall.com"
    HEADERS = {"User-Agent": "MTGScannerSuite/4.0", "Accept": "application/json"}
    _cache: Dict[str, Optional[dict]] = {}

    @staticmethod
    def clean(text: Optional[str]) -> str:
        if not text:
            return ""
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def match(cls, card: DetectedCard) -> Optional[dict]:
        cache_key = f"{card.card_name_raw}|{card.card_name_en}|{card.set_code}|{card.collector_number}"
        if cache_key in cls._cache:
            return cls._cache[cache_key]

        result = cls._resolve(card)
        cls._cache[cache_key] = result
        return result

    @classmethod
    def _resolve(cls, card: DetectedCard) -> Optional[dict]:
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

        # 2. English / Primary Name Search
        for name in [card.card_name_en, card.card_name_raw]:
            if not name:
                continue
            clean_name = cls.clean(name)
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

        # 3. Foreign / Printed Name Search (supports German, French, etc.)
        if card.card_name_raw:
            raw_clean = cls.clean(card.card_name_raw).replace('"', '')
            queries = [
                f'!"{raw_clean}" lang:any',
                f'printed_name:"{raw_clean}"',
                f'{raw_clean} lang:any'
            ]
            for query in queries:
                try:
                    r = requests.get(f"{cls.BASE_URL}/cards/search", params={"q": query}, headers=cls.HEADERS, timeout=5)
                    time.sleep(0.08)
                    if r.status_code == 200:
                        data = r.json()
                        if data.get("total_cards", 0) > 0:
                            return data["data"][0]
                except requests.RequestException:
                    pass

        # 4. Autocomplete Typo Fallback
        for name in [card.card_name_en, card.card_name_raw]:
            if not name:
                continue
            try:
                r = requests.get(f"{cls.BASE_URL}/cards/autocomplete", params={"q": cls.clean(name)}, headers=cls.HEADERS, timeout=5)
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
# Vision Pipeline
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
        "Cards can be printed in German, English, or other languages.\n"
        "Extract:\n"
        "1. Normalized box `box_2d` [ymin, xmin, ymax, xmax] (0-1000).\n"
        "2. Four outer `corners` in order: top_left, top_right, bottom_right, bottom_left (0-1000).\n"
        "3. `card_name_raw`: Exact name printed on the card.\n"
        "4. `card_name_en`: The canonical English Oracle name for this card.\n"
        "5. `set_code` and `collector_number` if visible in the bottom-left margin.\n"
        "6. `is_partially_obscured` (true if overlapping/stacked under another card).\n"
        "7. `is_foil` (true if the card surface has holographic sheen/foil reflection)."
    )

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
    print(f"\n--- [Scenario 2] Cards Scan ({MODEL_NAME}): '{image_path}' ---")
    
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
    parser = argparse.ArgumentParser(description="Multilingual MTG Vision & Recognition Suite")
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
