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
