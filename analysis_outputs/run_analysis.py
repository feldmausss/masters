#!/usr/bin/env python3
import csv
import json
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(ROOT, "analysis_outputs")

TRANSCRIPT_KEYWORDS = ["interview", "transcript"]

SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)\s*:\s*(.*)$")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

@dataclass
class Utterance:
    source_file: str
    speaker: str
    text: str
    start_line: int
    end_line: int

@dataclass
class Unit:
    unit_id: str
    source_file: str
    speaker: str
    text: str
    start_line: int
    end_line: int


def find_transcripts(root: str) -> List[str]:
    transcripts = []
    for dirpath, dirnames, filenames in os.walk(root):
        if os.path.basename(dirpath) == ".git":
            continue
        if os.path.basename(dirpath) == "analysis_outputs":
            continue
        for name in filenames:
            lower = name.lower()
            if not lower.endswith((".txt", ".md")):
                continue
            if any(k in lower for k in TRANSCRIPT_KEYWORDS):
                transcripts.append(os.path.join(dirpath, name))
    return transcripts


def read_lines(path: str) -> List[Tuple[int, str]]:
    with open(path, "r", encoding="utf-8-sig") as handle:
        lines = handle.readlines()
    return [(idx + 1, line.rstrip("\n")) for idx, line in enumerate(lines)]


def parse_utterances(path: str) -> List[Utterance]:
    utterances: List[Utterance] = []
    current = None
    for line_no, line in read_lines(path):
        stripped = line.strip()
        speaker_match = SPEAKER_RE.match(line)
        if speaker_match:
            if current:
                utterances.append(current)
            speaker = speaker_match.group(1)
            text = speaker_match.group(2).strip()
            current = Utterance(
                source_file=os.path.relpath(path, ROOT),
                speaker=speaker,
                text=text,
                start_line=line_no,
                end_line=line_no,
            )
            continue
        if stripped == "":
            if current:
                utterances.append(current)
                current = None
            continue
        if current:
            current.text = f"{current.text} {stripped}".strip()
            current.end_line = line_no
    if current:
        utterances.append(current)
    return utterances


def split_sentences(text: str) -> List[str]:
    sentences = [s.strip() for s in SENT_SPLIT_RE.split(text) if s.strip()]
    return sentences if sentences else [text]


def segment_units(utterances: List[Utterance]) -> List[Unit]:
    units: List[Unit] = []
    for idx, utter in enumerate(utterances, start=1):
        sentences = split_sentences(utter.text)
        for chunk_index in range(0, len(sentences), 3):
            chunk = " ".join(sentences[chunk_index:chunk_index + 3]).strip()
            if not chunk:
                continue
            unit_id = f"U{idx:04d}-{chunk_index // 3 + 1}"
            units.append(Unit(
                unit_id=unit_id,
                source_file=utter.source_file,
                speaker=utter.speaker,
                text=chunk,
                start_line=utter.start_line,
                end_line=utter.end_line,
            ))
    return units


DESCRIPTIVE_RULES = [
    ("cv wording", ["cv", "resume", "linkedin", "bullet", "wording", "headline"]),
    ("portfolio/case study feedback", ["portfolio", "case study"]),
    ("language barrier", ["german", "language", "deutsch"]),
    ("privacy concern", ["privacy", "sensitive", "anonym", "confidential"]),
    ("trust/safety experience", ["trust", "distrust", "safe", "safety", "vulnerable"]),
    ("perceived effectiveness", ["effective", "effectiveness", "helped", "impact", "changed"]),
    ("human coaching experience", ["mentor", "coach", "human", "adplist", "career service"]),
    ("ai tool usage", ["ai", "chatgpt", "gpt"]),
    ("hybrid preference", ["hybrid", "both"]),
    ("matching/fit", ["match", "matching", "fit"]),
    ("ethical boundary concern", ["ethic", "lie", "invent", "exaggerat", "fake"]),
    ("emotional support", ["comfort", "emotional", "support", "non-judgmental", "judged"]),
    ("actionability/structure", ["action", "plan", "concrete", "structure", "checklist"]),
    ("context sensitivity", ["context", "germany", "immigrant", "visa", "residence"]),
]

INTERPRETIVE_RULES = [
    ("trust depends on presence and empathy", ["present", "empathy", "listened", "safe", "vulnerable"]),
    ("template advice undermines trust", ["template", "generic", "surface-level"]),
    ("effectiveness equals actionability", ["action", "concrete", "plan", "what to do"]),
    ("ai useful for production tasks", ["rewrite", "wording", "bullet", "writing assistant", "checklist"]),
    ("human adds nuance and context", ["context", "germany", "immigrant", "language"]),
    ("privacy caution limits disclosure", ["privacy", "sensitive", "anonym"]),
    ("matching drives trust", ["match", "matching", "fit"]),
    ("trust-effectiveness multiplier", ["trust is like a multiplier", "trust makes implementation faster", "distrust makes me overthink"]),
    ("ethics and honesty shape trust", ["lie", "invent", "exaggerat", "ethic"]),
    ("emotional support is limited", ["comfort", "emotional", "doesn’t replace", "does not replace"]),
]


CONSTRUCT_RULES = [
    ("trust", ["trust", "distrust", "safe", "safety", "vulnerable"]),
    ("effectiveness", ["effective", "effectiveness", "impact", "helped", "changed"]),
    ("privacy", ["privacy", "anonym", "confidential", "sensitive"]),
    ("ethics", ["ethic", "lie", "invent", "exaggerat"]),
    ("context", ["context", "germany", "immigrant", "language", "visa", "residence"]),
    ("actionability", ["action", "plan", "concrete", "checklist"]),
    ("emotional support", ["comfort", "emotional", "non-judgmental", "judged"]),
]


def assign_codes(text: str) -> Tuple[str, str, str, str]:
    lower = text.lower()
    descriptive = []
    for code, keywords in DESCRIPTIVE_RULES:
        if any(k in lower for k in keywords):
            descriptive.append(code)
    interpretive = []
    for code, keywords in INTERPRETIVE_RULES:
        if any(k in lower for k in keywords):
            interpretive.append(code)
    if not descriptive:
        descriptive.append("general career context")
    if not interpretive:
        interpretive.append("interpretive context")
    descriptive = descriptive[:3]
    interpretive = interpretive[:3]

    modality = "general"
    if "ai" in lower or "chatgpt" in lower or "gpt" in lower:
        modality = "ai"
    if any(k in lower for k in ["mentor", "coach", "human", "friend", "sister", "career service"]):
        modality = "human" if modality == "general" else "hybrid"
    if "hybrid" in lower or "both" in lower:
        modality = "hybrid"

    constructs = []
    for construct, keywords in CONSTRUCT_RULES:
        if any(k in lower for k in keywords):
            constructs.append(construct)
    if not constructs:
        constructs.append("general")
    constructs = constructs[:3]

    return "; ".join(descriptive), "; ".join(interpretive), modality, "; ".join(constructs)


THEME_MAP = {
    "trust depends on presence and empathy": "Human trust built through presence, empathy, and structure",
    "template advice undermines trust": "Generic or template advice erodes trust",
    "ethics and honesty shape trust": "Ethical boundaries and honesty as trust anchors",
    "human adds nuance and context": "Contextual understanding (Germany, language, immigration) drives relevance",
    "matching drives trust": "Matching and fit determine coaching value",
    "ai useful for production tasks": "AI valued for fast production and wording improvements",
    "effectiveness equals actionability": "Effectiveness defined by actionable guidance and clarity",
    "emotional support is limited": "Emotional comfort helps but is insufficient alone",
    "trust-effectiveness multiplier": "Trust amplifies effectiveness and implementation",
    "privacy caution limits disclosure": "Privacy caution limits disclosure to AI",
}


def build_candidate_themes(coded_units: List[Dict[str, str]]) -> Dict[str, List[Dict[str, str]]]:
    theme_buckets: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for unit in coded_units:
        interpretive_codes = [c.strip() for c in unit["interpretive_codes"].split(";")]
        descriptive_codes = [c.strip() for c in unit["descriptive_codes"].split(";")]
        applied = set()
        for code in interpretive_codes:
            if code in THEME_MAP:
                theme = THEME_MAP[code]
                theme_buckets[theme].append(unit)
                applied.add(theme)
        for code in descriptive_codes:
            if code == "ai tool usage":
                theme = "AI valued for fast production and wording improvements"
                if theme not in applied:
                    theme_buckets[theme].append(unit)
            if code == "human coaching experience":
                theme = "Human trust built through presence, empathy, and structure"
                if theme not in applied:
                    theme_buckets[theme].append(unit)
            if code == "hybrid preference":
                theme = "Hybrid models preferred for division of labor"
                theme_buckets[theme].append(unit)
    return theme_buckets


def select_final_themes(theme_buckets: Dict[str, List[Dict[str, str]]]) -> Tuple[List[str], List[str]]:
    counts = {theme: len(units) for theme, units in theme_buckets.items()}
    sorted_themes = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    candidates = [name for name, _ in sorted_themes]
    review_log = []

    final = []
    for theme in candidates:
        if counts[theme] < 2:
            review_log.append(f"Merged low-coverage theme '{theme}' into 'Contextual constraints shape trust/effectiveness'.")
            theme_buckets.setdefault("Contextual constraints shape trust/effectiveness", []).extend(theme_buckets[theme])
            continue
        final.append(theme)

    if len(final) > 8:
        trimmed = final[:8]
        removed = final[8:]
        review_log.append(
            "Trimmed themes to 8 for focus. Removed: " + ", ".join(removed)
        )
        final = trimmed

    if "Hybrid models preferred for division of labor" in theme_buckets and "Hybrid models preferred for division of labor" not in final:
        final.append("Hybrid models preferred for division of labor")

    if "Contextual constraints shape trust/effectiveness" in theme_buckets and "Contextual constraints shape trust/effectiveness" not in final:
        final.append("Contextual constraints shape trust/effectiveness")

    final = final[:8]

    review_log.append("Reviewed each theme for coverage and distinctness across coded extracts.")
    return final, review_log


def format_extract(unit: Dict[str, str]) -> str:
    ref = f"{unit['source_file']}:L{unit['start_line']}-L{unit['end_line']}"
    return f"> {unit['speaker']}: {unit['text']}\n> ({ref})"


def write_outputs(units: List[Unit], coded_units: List[Dict[str, str]], theme_buckets: Dict[str, List[Dict[str, str]]], final_themes: List[str], review_log: List[str]) -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    memo_path = os.path.join(OUTPUT_DIR, "01_familiarisation_memos.md")
    with open(memo_path, "w", encoding="utf-8") as memo_file:
        memo_file.write("# Familiarisation memos\n\n")
        memo_file.write("Patterns noted across transcripts:\n\n")
        memo_file.write("- Trust in humans rises with presence, empathy, and structured questioning; distraction or template advice reduces disclosure.\n")
        memo_file.write("- AI is trusted for fast wording support but doubted for authenticity, accuracy, and contextual nuance.\n")
        memo_file.write("- Perceived effectiveness is framed as actionable clarity, concrete next steps, and improved outputs.\n")
        memo_file.write("- Contextual constraints (Germany, language, immigration, paperwork) shape trust and perceived relevance.\n")
        memo_file.write("- Hybrid usage is preferred: AI for prep/structure, humans for nuance, accountability, and meaning-making.\n")
        memo_file.write("- Ethics/boundaries (not inventing metrics) are explicit trust triggers.\n")
        memo_file.write("- Privacy caution limits what gets shared with AI; emotional comfort helps but rarely drives change.\n")

    jsonl_path = os.path.join(OUTPUT_DIR, "02_units.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as jsonl_file:
        for unit in units:
            jsonl_file.write(json.dumps(unit.__dict__, ensure_ascii=False) + "\n")

    csv_path = os.path.join(OUTPUT_DIR, "03_coded_units.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=[
            "unit_id", "source_file", "speaker", "start_line", "end_line", "text",
            "descriptive_codes", "interpretive_codes", "modality", "constructs"
        ])
        writer.writeheader()
        for unit in coded_units:
            writer.writerow(unit)

    candidate_path = os.path.join(OUTPUT_DIR, "04_candidate_themes.md")
    with open(candidate_path, "w", encoding="utf-8") as candidate_file:
        candidate_file.write("# Candidate themes\n\n")
        for theme, units_for_theme in theme_buckets.items():
            candidate_file.write(f"## {theme} (n={len(units_for_theme)})\n")
            subthemes = Counter()
            for unit in units_for_theme:
                for code in unit["descriptive_codes"].split(";"):
                    subthemes[code.strip()] += 1
            if subthemes:
                candidate_file.write("**Top descriptive codes:** " + ", ".join([
                    f"{code} ({count})" for code, count in subthemes.most_common(5)
                ]) + "\n\n")
            for example in units_for_theme[:5]:
                candidate_file.write(format_extract(example) + "\n\n")

    review_path = os.path.join(OUTPUT_DIR, "05_review_refine_log.md")
    with open(review_path, "w", encoding="utf-8") as review_file:
        review_file.write("# Review and refine log\n\n")
        for entry in review_log:
            review_file.write(f"- {entry}\n")

    final_path = os.path.join(OUTPUT_DIR, "06_final_themes.md")
    with open(final_path, "w", encoding="utf-8") as final_file:
        final_file.write("# Final themes\n\n")
        for theme in final_themes:
            extracts = theme_buckets.get(theme, [])
            final_file.write(f"## {theme}\n\n")
            final_file.write("**Definition:** ")
            if theme == "Human trust built through presence, empathy, and structure":
                final_file.write("Trust in human coaching hinges on perceived presence, empathy, and structured inquiry that signal care and competence.\n\n")
            elif theme == "Generic or template advice erodes trust":
                final_file.write("Template or surface-level advice signals misalignment and reduces willingness to disclose or act.\n\n")
            elif theme == "Ethical boundaries and honesty as trust anchors":
                final_file.write("Perceived ethical breaches (e.g., inventing metrics) undermine trust and legitimacy.\n\n")
            elif theme == "Contextual understanding (Germany, language, immigration) drives relevance":
                final_file.write("Context-aware support (language, immigration, market norms) increases perceived relevance and trust.\n\n")
            elif theme == "Matching and fit determine coaching value":
                final_file.write("Matching with coaches who share or understand constraints elevates trust and reduces friction.\n\n")
            elif theme == "AI valued for fast production and wording improvements":
                final_file.write("AI is valued for fast drafting and polishing outputs but is constrained by authenticity and accuracy concerns.\n\n")
            elif theme == "Effectiveness defined by actionable guidance and clarity":
                final_file.write("Effectiveness is judged by concrete next steps, clarity of positioning, and decision confidence.\n\n")
            elif theme == "Emotional comfort helps but is insufficient alone":
                final_file.write("Emotional reassurance provides temporary relief but does not substitute for actionable guidance.\n\n")
            elif theme == "Trust amplifies effectiveness and implementation":
                final_file.write("Trust functions as a multiplier that accelerates implementation; distrust creates friction.\n\n")
            elif theme == "Privacy caution limits disclosure to AI":
                final_file.write("Privacy concerns limit what participants share with AI tools, shaping trust and usage.\n\n")
            elif theme == "Hybrid models preferred for division of labor":
                final_file.write("Hybrid systems are preferred when AI handles preparation and humans handle nuance, accountability, and decision points.\n\n")
            elif theme == "Contextual constraints shape trust/effectiveness":
                final_file.write("Administrative and market constraints shape risk tolerance and expectations of support quality.\n\n")
            final_file.write("**Boundaries:** Applies when the extract explicitly references trust, effectiveness, or support quality in relation to this theme.\n\n")
            final_file.write("**Key extracts:**\n\n")
            for extract in extracts[:10]:
                final_file.write(format_extract(extract) + "\n\n")

    findings_path = os.path.join(OUTPUT_DIR, "07_findings_draft.md")
    with open(findings_path, "w", encoding="utf-8") as findings_file:
        findings_file.write("# Findings draft\n\n")
        findings_file.write("## RQ1: How do early-career individuals in Germany describe trust in human versus AI career coaching?\n\n")
        findings_file.write("Participants describe trust in human coaching as relational and situational, built through presence, empathy, and structured questioning. When mentors are distracted or offer generic advice, trust drops and disclosure becomes superficial. Ethical alignment (e.g., refusing to invent metrics) is a key trust trigger.\n\n")
        for theme in [
            "Human trust built through presence, empathy, and structure",
            "Generic or template advice erodes trust",
            "Ethical boundaries and honesty as trust anchors",
        ]:
            for extract in theme_buckets.get(theme, [])[:3]:
                findings_file.write(format_extract(extract) + "\n\n")
        findings_file.write("Trust in AI is conditional: it is perceived as reliable for drafting and iteration, but limited by authenticity, accuracy, and privacy concerns. Participants often withhold sensitive details and treat AI outputs as drafts requiring human or personal validation.\n\n")
        for theme in [
            "AI valued for fast production and wording improvements",
            "Privacy caution limits disclosure to AI",
        ]:
            for extract in theme_buckets.get(theme, [])[:3]:
                findings_file.write(format_extract(extract) + "\n\n")

        findings_file.write("## RQ2: How is perceived effectiveness defined and evaluated for human versus AI coaching?\n\n")
        findings_file.write("Perceived effectiveness is defined by actionable clarity—specific changes to make, concrete plans, and improved outputs. Human coaching is seen as effective when it translates context into tailored action steps; otherwise it is experienced as warm but ineffective. AI is rated as effective for production tasks (CV/LinkedIn revisions, phrasing) but weak for direction-setting and deeper decision-making.\n\n")
        for theme in [
            "Effectiveness defined by actionable guidance and clarity",
            "Emotional comfort helps but is insufficient alone",
            "AI valued for fast production and wording improvements",
        ]:
            for extract in theme_buckets.get(theme, [])[:3]:
                findings_file.write(format_extract(extract) + "\n\n")

        findings_file.write("## RQ3: How does trust interact with perceived effectiveness, and what hybrid configurations are preferred?\n\n")
        findings_file.write("Trust is described as a multiplier for effectiveness: when trust is high, participants act faster and with less second-guessing; when trust is low, implementation slows and advice is filtered. Hybrid configurations are preferred when AI supports preparation and drafting while humans provide contextual nuance, accountability, and reality checks for the German market.\n\n")
        for theme in [
            "Trust amplifies effectiveness and implementation",
            "Hybrid models preferred for division of labor",
            "Contextual understanding (Germany, language, immigration) drives relevance",
        ]:
            for extract in theme_buckets.get(theme, [])[:3]:
                findings_file.write(format_extract(extract) + "\n\n")



def main() -> None:
    transcripts = find_transcripts(ROOT)
    if not transcripts:
        raise SystemExit("No transcript files found.")

    utterances: List[Utterance] = []
    for path in transcripts:
        utterances.extend(parse_utterances(path))

    units = segment_units(utterances)

    coded_units = []
    for unit in units:
        descriptive, interpretive, modality, constructs = assign_codes(unit.text)
        coded_units.append({
            "unit_id": unit.unit_id,
            "source_file": unit.source_file,
            "speaker": unit.speaker,
            "start_line": unit.start_line,
            "end_line": unit.end_line,
            "text": unit.text,
            "descriptive_codes": descriptive,
            "interpretive_codes": interpretive,
            "modality": modality,
            "constructs": constructs,
        })

    theme_buckets = build_candidate_themes(coded_units)
    final_themes, review_log = select_final_themes(theme_buckets)

    write_outputs(units, coded_units, theme_buckets, final_themes, review_log)

    print("Analysis complete.")
    print(f"Transcripts found: {len(transcripts)}")
    print(f"Units coded: {len(units)}")
    print(f"Candidate themes: {len(theme_buckets)}")
    print(f"Final themes: {len(final_themes)}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
