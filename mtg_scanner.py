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
# Structured Output Schemas for Gemini Vision
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
    is_foil: bool = Field(False, description="True if card is foil/shiny")
    box_2d: List[int] = Field(..., description="Bounding box [ymin, xmin, ymax, xmax] (0-1000)")
    corners: Optional[CornerPoints] = Field(None, description="Corners (0-1000)")

class CardDetectionResult(BaseModel):
    cards: List[DetectedCard] = Field(default_factory=list, description="All detected cards")


# ==============================================================================
# High-Throughput Scryfall Batch Matcher (POST /cards/collection)
# ==============================================================================
class ScryfallBatchEngine:
    COLLECTION_URL = "https://api.scryfall.com/cards/collection"
    NAMED_URL = "https://api.scryfall.com/cards/named"
    SEARCH_URL = "https://api.scryfall.com/cards/search"
    HEADERS = {"User-Agent": "MTGScannerSuite/5.0", "Accept": "application/json"}

    @classmethod
    def resolve_all_cards(cls, detected_cards: List[DetectedCard]) -> List[Optional[dict]]:
        """
        Batches all cards into chunks of up to 75 items to resolve in 1 single HTTP POST request.
        Falls back to individual search for any unresolved foreign card names.
        """
        resolved: List[Optional[dict]] = [None] * len(detected_cards)
        identifiers = []
        mapping = []

        for idx, card in enumerate(detected_cards):
            query_name = (card.card_name_en or card.card_name_raw or "").strip()
            if not query_name:
                continue

            # If set code & collector number exist, match exact print
            if card.set_code and card.collector_number:
                set_clean = re.sub(r"[^a-zA-Z0-9]", "", card.set_code).lower()
                num_clean = re.sub(r"[^a-zA-Z0-9]", "", card.collector_number)
                identifiers.append({"set": set_clean, "collector_number": num_clean})
            else:
                identifiers.append({"name": query_name})
            mapping.append(idx)

        # 1. Execute Batch Request (chunks of 75)
        for i in range(0, len(identifiers), 75):
            chunk_identifiers = identifiers[i:i+75]
            chunk_mapping = mapping[i:i+75]

            try:
                resp = requests.post(
                    cls.COLLECTION_URL,
                    json={"identifiers": chunk_identifiers},
                    headers=cls.HEADERS,
                    timeout=10
                )
                if resp.status_code == 200:
                    data = resp.json()
                    # Map matched results
                    for match in data.get("data", []):
                        # Match by name or collector number
                        m_name = match.get("name", "").lower()
                        for pos, orig_idx in enumerate(chunk_mapping):
                            if resolved[orig_idx] is not None:
                                continue
                            c = detected_cards[orig_idx]
                            raw_n = (c.card_name_raw or "").lower()
                            en_n = (c.card_name_en or "").lower()
                            if m_name == raw_n or m_name == en_n or m_name.startswith(en_n) or raw_n in m_name:
                                resolved[orig_idx] = match
                                break
                time.sleep(0.1)
            except requests.RequestException:
                pass

        # 2. Precision Individual Fallback for remaining unmatched cards (with rate limit backoff)
        for idx, card in enumerate(detected_cards):
            if resolved[idx] is None:
                resolved[idx] = cls._single_card_lookup(card)

        return resolved

    @classmethod
    def _single_card_lookup(cls, card: DetectedCard) -> Optional[dict]:
        names_to_try = [card.card_name_en, card.card_name_raw]
        for name in names_to_try:
            if not name:
                continue
            clean_name = re.sub(r"\s+", " ", name).strip()
            
            # Fuzzy match
            res = cls._get_with_retry(cls.NAMED_URL, params={"fuzzy": clean_name})
            if res:
                return res

            # German/foreign language search
            query = f'!"{clean_name}" lang:any'
            res = cls._get_with_retry(cls.SEARCH_URL, params={"q": query})
            if res and res.get("total_cards", 0) > 0:
                return res["data"][0]

        return None

    @classmethod
    def _get_with_retry(cls, url: str, params: dict, max_retries: int = 3) -> Optional[dict]:
        for attempt in range(max_retries):
            try:
                resp = requests.get(url, params=params, headers=cls.HEADERS, timeout=5)
                if resp.status_code == 200:
                    time.sleep(0.08)
                    return resp.json()
                elif resp.status_code == 429:
                    # Rate limit hit -> wait and retry with backoff
                    wait_sec = float(resp.headers.get("Retry-After", 1.5 * (attempt + 1)))
                    time.sleep(wait_sec)
                    continue
                else:
                    time.sleep(0.08)
                    return None
            except requests.RequestException:
                time.sleep(0.5)
        return None


# ==============================================================================
# Annotation & Formatting Helpers
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
        label = f"Unmatched: {card.card_name_raw or card.card_name_en or 'Unknown'}"
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
# Vision & Batch Recognition Pipeline
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
        "Cards can be in German or English.\n"
        "Extract:\n"
        "1. Normalized box `box_2d` [ymin, xmin, ymax, xmax] (0-1000).\n"
        "2. Four outer `corners` in order: top_left, top_right, bottom_right, bottom_left (0-1000).\n"
        "3. `card_name_raw`: Exact printed name on card (e.g. 'Vitalitätsschub', 'Thorn of the Black Rose').\n"
        "4. `card_name_en`: The canonical English Oracle name.\n"
        "5. `set_code` and `collector_number` if legible.\n"
        "6. `is_partially_obscured` (true if overlapping/stacked).\n"
        "7. `is_foil` (true if shiny/reflective)."
    )

    print(f"🔍 Analyzing image with {MODEL_NAME}...")
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
    print(f"🃏 Located {len(detection_data.cards)} card candidate(s). Resolving with Scryfall Batch API...")

    # Batch resolve via Scryfall POST /cards/collection
    resolved_matches = ScryfallBatchEngine.resolve_all_cards(detection_data.cards)

    annotated = cv_img.copy()
    processed_cards = []
    unrecognized_count = 0

    for idx, (card, match) in enumerate(zip(detection_data.cards, resolved_matches), start=1):
        justified_img_path = None
        if save_crops:
            crop_file = os.path.join(crops_dir, f"card_{idx:02d}.jpg")
            justified_img_path = justify_card(cv_img, card, crop_file)

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
# Scenario Handlers
# ==============================================================================
def scenario_deck_scan(image_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
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

    moxfield_lines = [
        format_moxfield_line(count, name, set_code, collector_num, is_foil)
        for (name, set_code, collector_num, is_foil), count in sorted(moxfield_counts.items(), key=lambda x: x[0][0])
    ]

    moxfield_txt_path = os.path.join(output_dir, "decklist_moxfield.txt")
    with open(moxfield_txt_path, "w", encoding="utf-8") as f:
        for line in moxfield_lines:
            f.write(f"{line}\n")

    print_moxfield_terminal(moxfield_lines, "DECK MOXFIELD LIST")
    print(f"✅ Recognized {total_detected - unrecognized}/{total_detected} cards.")
    print(f"✅ Files saved to '{output_dir}/'")


def scenario_cards_scan(image_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
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
    print(f"✅ Recognized {total_detected - unrecognized}/{total_detected} cards.")
    print(f"✅ Files saved to '{output_dir}/'")


def scenario_collection_scan(directory_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    all_files = [f for f in glob.glob(os.path.join(directory_path, "*")) if Path(f).suffix.lower() in VALID_IMAGE_EXTENSIONS]

    total_cards_all = 0
    total_unrecognized_all = 0
    collection_moxfield_counts: Dict[Tuple[str, Optional[str], Optional[str], bool], int] = {}

    for idx, img_path in enumerate(all_files, start=1):
        print(f"[{idx}/{len(all_files)}] Processing: {os.path.basename(img_path)}...")
        cards, total, unrec = process_single_image(img_path, output_dir, save_crops=True)
        
        for c in [c for c in cards if c["matched"]]:
            name = c["scryfall"]["name"]
            set_code = c["scryfall"].get("set")
            collector_num = c["scryfall"].get("collector_number")
            is_foil = c["detected_raw"].get("is_foil", False)
            key = (name, set_code, collector_num, is_foil)
            collection_moxfield_counts[key] = collection_moxfield_counts.get(key, 0) + 1

        total_cards_all += total
        total_unrecognized_all += unrec

    coll_moxfield_lines = [
        format_moxfield_line(count, name, set_code, collector_num, is_foil)
        for (name, set_code, collector_num, is_foil), count in sorted(collection_moxfield_counts.items(), key=lambda x: x[0][0])
    ]

    print_moxfield_terminal(coll_moxfield_lines, "COLLECTION MOXFIELD LIST")
    print(f"✨ Processed {len(all_files)} images, {total_cards_all - total_unrecognized_all}/{total_cards_all} cards recognized.")


# ==============================================================================
# CLI Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Multilingual MTG Vision Suite (Batch Scryfall Resolution)")
    subparsers = parser.add_subparsers(dest="scenario", required=True, help="Processing Mode")

    deck_parser = subparsers.add_parser("deck", help="Scan deck image")
    deck_parser.add_argument("image", type=str, help="Path to deck photo")
    deck_parser.add_argument("--out", type=str, default="output_deck")

    cards_parser = subparsers.add_parser("cards", help="Scan single cards image")
    cards_parser.add_argument("image", type=str, help="Path to card photo")
    cards_parser.add_argument("--out", type=str, default="output_cards")

    coll_parser = subparsers.add_parser("collection", help="Scan directory")
    coll_parser.add_argument("dir", type=str, help="Path to images directory")
    coll_parser.add_argument("--out", type=str, default="output_collection")

    args = parser.parse_args()

    if args.scenario == "deck":
        scenario_deck_scan(args.image, args.out)
    elif args.scenario == "cards":
        scenario_cards_scan(args.image, args.out)
    elif args.scenario == "collection":
        scenario_collection_scan(args.dir, args.out)


if __name__ == "__main__":
    main()
