"""
IITGN RAG - Data Ingestion Pipeline (Steps 1 to 6 of the design document)

Step 1: Inventory    - list every file in the data folder
Step 2: Parse        - separate each file's header from its content
Step 3: Clean        - fix text and remove noise
Step 4: Deduplicate  - remove exact and near-duplicate documents
Step 5: Enrich       - add category, department, date and audience
Step 6: Chunk        - split documents into searchable chunks

Output: output/manifest.csv   (one row per file, with its final status)
        output/parsed/*.json  (one parsed document per file)
        output/chunks.jsonl   (one chunk per line)

Run: python data_ingest.py
"""

import csv
import hashlib
import json
import re
import shutil
import unicodedata
from collections import Counter
from datetime import date, datetime
from itertools import groupby
from pathlib import Path

import ftfy
from datasketch import MinHash, MinHashLSH
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

from config import LOCAL_EMBEDDING_MODEL

# ---------- Settings ----------

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
PARSED_DIR = OUTPUT_DIR / "parsed"
CHUNKS_FILE = OUTPUT_DIR / "chunks.jsonl"
MANIFEST_FILE = OUTPUT_DIR / "manifest.csv"

# Status of a document that is still moving through the pipeline
ACTIVE = "active"

# Cleaning (Step 3)
MIN_DOC_CHARS = 20        # Shorter documents are treated as empty
HEADER_MIN_REPEATS = 4    # A page header repeats at least this many times
HEADER_MAX_CHARS = 100    # A page header is never longer than this
INVISIBLE_CHARS = "".join(map(chr, [0x200B, 0x200C, 0x200D, 0xFEFF, 0x00AD]))
INDIAN_SCRIPTS = f"{chr(0x0900)}-{chr(0x0DFF)}"

# Deduplication (Step 4)
SHINGLE_WORDS = 5           # Documents are compared in groups of 5 words
NEAR_DUP_THRESHOLD = 0.9    # 90% similar documents count as duplicates
NEAR_DUP_MIN_WORDS = 50     # Short pages look alike, so they are not compared
MINHASH_SIZE = 128          # Higher is more accurate but slower

# Metadata (Step 5)
MIN_YEAR = 2008                    # The year IIT Gandhinagar was founded
MAX_YEAR = date.today().year + 1   # Allows next academic year's documents
DATE_SEARCH_CHARS = 1500           # A document's date is near its start
DEPARTMENT_SEARCH_CHARS = 200      # A person's department is near the top

# Chunking (Step 6)
CHUNK_SIZE = 450       # In tokens, leaves room for the title line (model limit is 512)
CHUNK_OVERLAP = 45     # In tokens, 10% of chunk size
MAX_TITLE_LINE_TOKENS = 60   # 450 + 60 + 2 special tokens = the model limit of 512
MIN_CHUNK_CHARS = 30   # Shorter chunks carry no useful meaning
MIN_TABLE_ROWS = 3     # Fewer consecutive rows are treated as plain text

# Chunks are measured in the embedding model's own tokens
tokenizer = AutoTokenizer.from_pretrained(LOCAL_EMBEDDING_MODEL)
tokenizer.model_max_length = 10**9   # Allows counting long text without warnings


def count_tokens(text):
    """Return how many tokens the embedding model sees in the text."""
    return len(tokenizer.encode(text, add_special_tokens=False))


# Split at headings first, then paragraphs, lines, sentences and words
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=count_tokens,
    separators=["\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n\n", "\n", ". ", " ", ""],
)

# The first matching rule decides the category, so order matters
CATEGORY_RULES = [
    ("admissions", r"admissions?|cutoff|coap"),
    ("administration", r"tenders?|careers?|advertisement|advt|recruitment|rti|nirf|statutes?"
                       r"|ordinances?|bog|aac|alc|annual reports?|ar 20\d\d|annualletter"),
    ("fees", r"fees?|scholarships?|stipends?|fellowships?"),
    ("hostel", r"hostels?|housing|dining|dinning|mess|guest house|guest php"),
    ("placements", r"cds|placements?|internships?"),
    ("library", r"library"),
    ("faculty", r"faculty|srs|professors?|ccl iitgn ac in team"),
    ("research", r"research|publications?|patents?|theses|r d newsletter"),
    ("events", r"events?|workshops?|lectures?|conclave|sports|ccl|newsletter|vol\d+ issue\d*"),
    ("campus", r"campus|safety|shuttle|transport"),
    ("academics", r"academics?|courses?|curriculum|calendar|deanslist|dean s list"
                  r"|btech|b tech|mtech|m tech|phd|semester"),
]

# (department, short form used in URLs, full name used in text)
DEPARTMENT_RULES = [
    ("CSE", r"cse", r"computer science"),
    ("EE", r"ee", r"electrical engineering"),
    ("ME", r"me", r"mechanical engineering"),
    ("CE", r"civil", r"civil engineering"),
    ("CL", r"chemical", r"chemical engineering"),
    ("BE", r"bioe|biomed", r"biological sciences|biological engineering|biomedical"),
    ("MSE", r"mse", r"materials science|materials engineering"),
    ("CH", r"chemistry", r"chemistry"),
    ("PH", r"phy", r"physics"),
    ("MA", r"maths", r"mathematics"),
    ("ES", r"earths", r"earth sciences?"),
    ("CGS", r"cogs", r"cognitive science"),
    ("HSS", r"hss", r"humanities"),
    ("AR", r"archaeology", r"archaeology"),
]

# The first matching rule decides the audience
AUDIENCE_RULES = [
    ("PhD", r"phd|ph d|doctoral"),
    ("PG", r"mtech|m tech|msc|m sc|academics ma|admissions ma|ma\d+|postgraduate|pg|emasters"),
    ("UG", r"btech|b tech|undergraduate|ug"),
]

# Finds dates such as "Dated: 12/08/2025" or "Date 30 December 2024"
# Deadlines such as "Last date 31 December" are not the document's date
DATED_PATTERN = re.compile(
    r"(?<!last )(?<!closing )(?<!due )(?<!expiry )\bdated?\b\s*[:\-]?\s*"
    r"(\d{1,2}(?:st|nd|rd|th)?[\s./-]+(?:\d{1,2}|[a-z]{3,9})[\s,./-]+\d{2,4})",
    re.IGNORECASE,
)
DATE_FORMATS = ["%d %m %Y", "%d %b %Y", "%d %B %Y", "%d %m %y", "%d %b %y"]


# ---------- Helpers ----------

def active_docs(documents):
    """Return documents that have not been skipped."""
    return [doc for doc in documents if doc["status"] == ACTIVE]


def print_step(number, name, detail):
    """Show the result of a step."""
    print(f"[Step {number}/6] {name}: {detail}")


# ---------- Step 1: Build an inventory ----------

def step_1_build_inventory():
    """Create one record per file in the data folder."""
    documents = []

    for path in sorted(DATA_DIR.rglob("*.txt")):
        raw = path.read_text(encoding="utf-8")
        relative_path = path.relative_to(DATA_DIR).as_posix()
        is_webpage = raw.startswith("Title:")

        documents.append({
            "doc_id": hashlib.sha1(relative_path.encode()).hexdigest()[:12],
            "path": relative_path,
            "domain": relative_path.split("/")[0],
            "source_url": raw.split("\n")[1].removeprefix("Source:").strip() if is_webpage else "",
            "file_type": "webpage" if is_webpage else path.name.split(".")[-2].lower(),
            "sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "last_modified": file_date(path.stat().st_mtime),
            "crawl_date": file_date(path.stat().st_ctime),   # File creation date on Windows
            "status": ACTIVE,
            "duplicate_of": "",
            "num_chunks": 0,
            "raw": raw,
        })

    print_step(1, "Inventory", f"found {len(documents)} files")
    return documents


def file_date(timestamp):
    """Convert a file timestamp to YYYY-MM-DD."""
    return datetime.fromtimestamp(timestamp).date().isoformat()


# ---------- Step 2: Parse by file type ----------

def step_2_parse(documents):
    """Separate each file's header from its content and save it as JSON."""
    shutil.rmtree(PARSED_DIR, ignore_errors=True)   # Remove results of the last run
    PARSED_DIR.mkdir(parents=True)

    for doc in documents:
        parse_document(doc)
        save_parsed_document(doc)

    unsupported = sum(doc["status"] == "unsupported_format" for doc in documents)
    print_step(2, "Parse", f"parsed {len(documents)} files, {unsupported} have an unsupported format")


def parse_document(doc):
    """Read the title, flags and content from a raw file."""
    raw = doc.pop("raw")
    doc["flags"] = []

    if doc["file_type"] == "webpage":
        # Web page format: Title line, then Source line, then content
        lines = raw.split("\n", 2)
        doc["title"] = lines[0].removeprefix("Title:").strip()
        doc["text_markdown"] = lines[2] if len(lines) > 2 else ""
    else:
        # Converted file format (PDF, XLSX...): #flags line, then content
        first_line, _, body = raw.partition("\n")
        flags = first_line.removeprefix("#flags:").strip()
        doc["flags"] = flags.split(",") if flags else []
        doc["title"] = title_from_filename(doc["path"].split("/")[-1])
        doc["text_markdown"] = body

    if "unsupported_format" in doc["flags"]:
        doc["status"] = "unsupported_format"


def title_from_filename(name):
    """Turn 'FAQs_about_courses__7746d349.pdf.txt' into 'FAQs about courses'."""
    name = remove_file_hash(name)
    return re.sub(r"[_\-\s]+", " ", name).strip()


def remove_file_hash(path):
    """Remove the '__7746d349.pdf.txt' ending from a file path."""
    return re.sub(r"__[0-9a-f]{8}\..*$", "", path)


def save_parsed_document(doc):
    """Save the normalized document as one JSON file."""
    fields = ["doc_id", "title", "text_markdown", "source_url", "file_type"]
    parsed = {field: doc[field] for field in fields}
    with open(PARSED_DIR / f"{doc['doc_id']}.json", "w", encoding="utf-8") as file:
        json.dump(parsed, file, ensure_ascii=False, indent=2)


# ---------- Step 3: Clean and normalize ----------

def step_3_clean(documents):
    """Clean every document and skip the ones left empty."""
    empty = 0

    for doc in active_docs(documents):
        doc["text_markdown"] = clean_text(doc["text_markdown"], doc["file_type"])
        if len(doc["text_markdown"]) < MIN_DOC_CHARS:
            doc["status"] = "empty"
            empty += 1

    print_step(3, "Clean", f"kept {len(active_docs(documents))} documents, {empty} were empty")


def clean_text(text, file_type):
    """Fix Unicode, keep English text and remove noise."""
    # Repair broken Unicode
    text = ftfy.fix_text(text)
    # Convert ligatures and non-breaking spaces to plain characters
    text = unicodedata.normalize("NFKC", text)
    # Remove invisible characters
    text = re.sub(f"[{INVISIBLE_CHARS}]", "", text)
    # Drop lines that are mostly non-English
    text = "\n".join(line for line in text.split("\n") if is_english_line(line))
    # Remove Indian-script words left inside English lines
    text = re.sub(f"[{INDIAN_SCRIPTS}]+", "", text)
    # Collapse repeated spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)
    # Shorten form blanks and dotted lines
    text = re.sub(r"([_.\-=])\1{4,}", r"\1\1\1", text)
    text = "\n".join(clean_line(line) for line in text.split("\n"))
    if file_type == "pdf":
        text = remove_page_headers(text)
    # Keep at most one blank line in a row
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_english_line(line):
    """Keep a line unless most of its letters are non-English."""
    letters = [char for char in line if char.isalpha()]
    # Latin and Greek are allowed, since formulas use Greek letters
    non_english = sum(ord(char) > 0x3FF for char in letters)
    return non_english <= len(letters) / 2


def clean_line(line):
    """Trim a line and drop empty table cells from its end."""
    line = line.strip()
    if line.endswith("|"):
        line = line.rstrip("| ")
        line = line + " |" if line else ""
    return line


def remove_page_headers(text):
    """Remove repeated headers such as 'Annual Report 2024-25 | www.iitgn.ac.in'."""
    lines = text.split("\n")
    counts = Counter(lines)

    def is_page_header(line):
        # Page breaks are not in the data, so headers are found by their layout
        has_header_layout = "|" in line or line.lower().startswith("annual report")
        return (
            has_header_layout
            and len(line) <= HEADER_MAX_CHARS
            and counts[line] >= HEADER_MIN_REPEATS
        )

    return "\n".join(line for line in lines if not is_page_header(line))


# ---------- Step 4: Deduplicate ----------

def step_4_deduplicate(documents):
    """Keep one copy of exact and near-duplicate documents."""
    # Check longest documents first, so the most complete version is kept
    docs = sorted(active_docs(documents), key=lambda d: len(d["text_markdown"]), reverse=True)

    seen_hashes = {}   # Content hash -> doc_id of the first copy
    signatures = {}    # doc_id -> MinHash signature
    numbers = {}       # doc_id -> numbers and dates in the document
    lsh = MinHashLSH(threshold=NEAR_DUP_THRESHOLD, num_perm=MINHASH_SIZE)
    exact = near = 0

    for doc in docs:
        # Exact duplicates have identical cleaned text
        content_hash = hashlib.sha256(doc["text_markdown"].encode()).hexdigest()
        if content_hash in seen_hashes:
            mark_duplicate(doc, "duplicate", seen_hashes[content_hash])
            exact += 1
            continue
        seen_hashes[content_hash] = doc["doc_id"]

        words = doc["text_markdown"].lower().split()
        if len(words) < NEAR_DUP_MIN_WORDS:
            continue

        # Near-duplicates share almost all of their word groups
        signature = build_signature(words)
        doc_numbers = extract_numbers(doc["text_markdown"])
        match = find_near_duplicate(signature, doc_numbers, lsh, signatures, numbers)
        if match:
            mark_duplicate(doc, "near_duplicate", match)
            near += 1
            continue

        signatures[doc["doc_id"]] = signature
        numbers[doc["doc_id"]] = doc_numbers
        lsh.insert(doc["doc_id"], signature)

    print_step(4, "Deduplicate", f"removed {exact} exact and {near} near duplicates, "
                                 f"kept {len(active_docs(documents))} documents")


def mark_duplicate(doc, status, original_id):
    """Record that a document is a copy of another one."""
    doc["status"] = status
    doc["duplicate_of"] = original_id


def build_signature(words):
    """Create a MinHash signature from overlapping groups of words."""
    shingles = {" ".join(words[i:i + SHINGLE_WORDS]) for i in range(len(words) - SHINGLE_WORDS + 1)}
    signature = MinHash(num_perm=MINHASH_SIZE)
    signature.update_batch([shingle.encode() for shingle in shingles])
    return signature


def extract_numbers(text):
    """Return every number and date in the text."""
    return set(re.findall(r"\d[\d,./-]*\d|\d", text))


def find_near_duplicate(signature, doc_numbers, lsh, signatures, numbers):
    """Return the doc_id of a near-duplicate, or None."""
    for candidate in lsh.query(signature):
        # LSH only suggests candidates, so confirm the similarity
        is_similar = signature.jaccard(signatures[candidate]) >= NEAR_DUP_THRESHOLD
        # A new date or amount means a different notice, so both are kept
        if is_similar and doc_numbers <= numbers[candidate]:
            return candidate
    return None


# ---------- Step 5: Enrich with metadata ----------

def step_5_enrich(documents):
    """Add metadata used for filtering at search time."""
    docs = active_docs(documents)

    for doc in docs:
        # Path and title, lowercase, with punctuation replaced by spaces
        doc["match_text"] = re.sub(r"[^a-z0-9]+", " ", f"{doc['path']} {doc['title']}".lower())
        doc["category"] = get_category(doc)
        doc["department"] = get_department(doc)
        doc["audience"] = get_audience(doc)
        doc["doc_date"] = get_doc_date(doc)

    categories = Counter(doc["category"] for doc in docs)
    dated = sum(bool(doc["doc_date"]) for doc in docs)
    with_department = sum(bool(doc["department"]) for doc in docs)
    print_step(5, "Enrich", f"{len(categories)} categories, {with_department} with a department, "
                            f"{dated} with a date")


def has_match(pattern, text):
    """Check whether any whole word in the pattern appears in the text."""
    return re.search(rf"\b(?:{pattern})\b", text) is not None


def get_category(doc):
    """Find the document's topic from its path and title."""
    for category, pattern in CATEGORY_RULES:
        if has_match(pattern, doc["match_text"]):
            return category
    return "general"


def get_department(doc):
    """Find the department from the path, title or top of a web page."""
    path_text = re.sub(r"[^a-z0-9]+", " ", remove_file_hash(doc["path"]).lower())

    # Short forms are only reliable inside URLs
    for department, short_form, full_name in DEPARTMENT_RULES:
        if has_match(short_form, path_text) or has_match(full_name, doc["match_text"]):
            return department

    # Web pages about a person name the department near the top
    if doc["file_type"] == "webpage":
        top_of_page = doc["text_markdown"][:DEPARTMENT_SEARCH_CHARS].lower()
        for department, _, full_name in DEPARTMENT_RULES:
            if has_match(full_name, top_of_page):
                return department
    return ""


def get_audience(doc):
    """Find who the document is for: UG, PG, PhD or all."""
    for audience, pattern in AUDIENCE_RULES:
        if has_match(pattern, doc["match_text"]):
            return audience
    return "all"


def get_doc_date(doc):
    """Return YYYY-MM-DD, or YYYY if only the year is known, or an empty string."""
    # 1. A date written in the document, such as "Dated: 12/08/2025"
    match = DATED_PATTERN.search(doc["text_markdown"][:DATE_SEARCH_CHARS])
    if match and parse_date(match.group(1)):
        return parse_date(match.group(1))

    name = f"{remove_file_hash(doc['path'])} {doc['title']}"

    # 2. A date in the file name, such as "25052026" (DDMMYYYY)
    match = re.search(r"(?<!\d)(\d{2})(\d{2})(20\d{2})(?!\d)", name)
    if match and parse_date(" ".join(match.groups())):
        return parse_date(" ".join(match.groups()))

    # 3. A year in the path or title, such as "AR-2019-20"
    for year in re.findall(r"(?<!\d)(20\d{2})(?!\d)", name):
        if MIN_YEAR <= int(year) <= MAX_YEAR:
            return year
    return ""


def parse_date(text):
    """Convert a written date to YYYY-MM-DD, or return an empty string."""
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text)
    text = re.sub(r"[\s,./-]+", " ", text).strip()

    for date_format in DATE_FORMATS:
        try:
            parsed = datetime.strptime(text, date_format)
        except ValueError:
            continue
        if MIN_YEAR <= parsed.year <= MAX_YEAR:
            return parsed.strftime("%Y-%m-%d")
    return ""


# ---------- Step 6: Chunk ----------

def step_6_chunk(documents):
    """Split documents into chunks and save them."""
    total = 0

    with open(CHUNKS_FILE, "w", encoding="utf-8") as file:
        for doc in active_docs(documents):
            chunks = chunk_document(doc)
            for chunk in chunks:
                file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            doc["num_chunks"] = len(chunks)
            doc["status"] = "chunked"
            total += len(chunks)

    print_step(6, "Chunk", f"created {total} chunks")


def chunk_document(doc):
    """Split a document into chunks, keeping tables intact where possible."""
    chunks = []
    section = ""  # Most recent heading seen

    for is_table, block in split_blocks(doc["text_markdown"]):
        pieces = split_table(block) if is_table else splitter.split_text(block)

        for piece in pieces:
            piece = piece.strip()
            headings = re.findall(r"^#{1,6} +(.+)$", piece, flags=re.MULTILINE)
            chunk_section = headings[0] if headings else section
            if headings:
                section = headings[-1]

            if len(piece) < MIN_CHUNK_CHARS:
                continue

            index = len(chunks)
            chunks.append({
                "chunk_id": f"{doc['doc_id']}_{index:04d}",
                "doc_id": doc["doc_id"],
                "chunk_index": index,
                "title": doc["title"],
                "section": chunk_section,
                "source_url": doc["source_url"],
                "path": doc["path"],
                "domain": doc["domain"],
                "file_type": doc["file_type"],
                "is_ocr": "ocr" in doc["flags"],
                "category": doc["category"],
                "department": doc["department"],
                "doc_date": doc["doc_date"],
                "audience": doc["audience"],
                "text": piece,
                "embed_text": build_embed_text(doc, chunk_section, piece),
            })

    return chunks


def split_blocks(text):
    """Separate text into ordered blocks of plain text and tables."""
    blocks = []

    # Group consecutive lines of the same type
    for is_table, group in groupby(text.split("\n"), key=is_table_line):
        lines = list(group)
        # A few pipe lines are page headers or broken rows, not real tables
        is_table = is_table and len(lines) >= MIN_TABLE_ROWS

        # Join neighbouring plain text so it is not fragmented
        if not is_table and blocks and not blocks[-1][0]:
            blocks[-1][1].extend(lines)
        else:
            blocks.append((is_table, lines))

    return [(is_table, "\n".join(lines)) for is_table, lines in blocks]


def is_table_line(line):
    """A table row has at least two pipe characters and is not a heading."""
    return line.count("|") >= 2 and not line.startswith("#")


def split_table(table):
    """Split a large table by rows and repeat the header in each piece."""
    rows = table.split("\n")
    header = rows[0]
    pieces = []
    header_size = count_tokens(header)
    current = [header]
    current_size = header_size

    for row in rows[1:]:
        row_size = count_tokens(row)
        # Start a new piece when this row would not fit
        if current_size + row_size > CHUNK_SIZE and len(current) > 1:
            pieces.append("\n".join(current))
            current = [header]
            current_size = header_size
        current.append(row)
        current_size += row_size

    pieces.append("\n".join(current))

    # A single very long row can still exceed the limit
    result = []
    for piece in pieces:
        result.extend(splitter.split_text(piece) if count_tokens(piece) > CHUNK_SIZE else [piece])
    return result


def build_embed_text(doc, section, text):
    """Add title, section and source to the chunk to improve retrieval."""
    heading = f"{doc['title']} > {section}" if section else doc["title"]
    source = doc["source_url"] or doc["domain"]
    title_line = f"{heading} | Source: {source}"

    # Shorten very long title lines so the whole chunk fits the model limit
    tokens = tokenizer.encode(title_line, add_special_tokens=False)
    if len(tokens) > MAX_TITLE_LINE_TOKENS:
        title_line = tokenizer.decode(tokens[:MAX_TITLE_LINE_TOKENS])

    return f"{title_line}\n\n{text}"


# ---------- Save the manifest ----------

def save_manifest(documents):
    """Save one row per file, showing what happened to it."""
    fields = [
        "doc_id", "path", "source_url", "file_type", "sha256", "last_modified", "crawl_date",
        "title", "category", "department", "doc_date", "audience",
        "status", "duplicate_of", "num_chunks",
    ]
    with open(MANIFEST_FILE, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore", restval="")
        writer.writeheader()
        writer.writerows(documents)


# ---------- Run the pipeline ----------

def run_data_ingestion_pipeline():
    print("starting the data ingestion pipeline")
    OUTPUT_DIR.mkdir(exist_ok=True)

    documents = step_1_build_inventory()
    step_2_parse(documents)
    step_3_clean(documents)
    step_4_deduplicate(documents)
    step_5_enrich(documents)
    step_6_chunk(documents)
    save_manifest(documents)

    print("\nFinal status of all files:")
    for status, count in Counter(doc["status"] for doc in documents).most_common():
        print(f"  {status}: {count}")
    print(f"\nSaved to {OUTPUT_DIR}/")
    print("data ingestion pipeline is completed")

