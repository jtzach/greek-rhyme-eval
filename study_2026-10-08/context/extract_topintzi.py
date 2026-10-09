"""Extract the author's complete PDF with font-aware Unicode recovery.

The source PDF's ToUnicode ranges incorrectly assume contiguous Greek glyph
assignments. Rebuild only Identity-H/Identity CID fonts from their own embedded
TrueType cmap, without changing, OCRing, or resaving the source PDF on disk.
Public portable derivative: install fontTools, pypdf and pdfplumber normally.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path


from fontTools.ttLib import TTFont
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject
import pdfplumber


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    directory = Path(__file__).resolve().parent
    source = directory / "topintzi_etal_2019_author.pdf"
    reader = PdfReader(source)
    visited: set[int] = set()
    font_changes = []
    for page in reader.pages:
        for _, ref in page["/Resources"]["/Font"].items():
            if ref.idnum in visited:
                continue
            visited.add(ref.idnum)
            font = ref.get_object()
            if font.get("/Subtype") != "/Type0" or font.get("/Encoding") != "/Identity-H":
                continue
            descendant = font["/DescendantFonts"][0].get_object()
            if descendant.get("/CIDToGIDMap") != "/Identity":
                continue
            descriptor = descendant["/FontDescriptor"].get_object()
            if "/FontFile2" not in descriptor:
                continue
            raw_ttf = descriptor["/FontFile2"].get_data()
            ttf = TTFont(io.BytesIO(raw_ttf))
            cmap = ttf.getBestCmap()
            if not cmap:
                continue
            glyph_to_unicode = {}
            for codepoint, glyph in sorted(cmap.items()):
                glyph_to_unicode.setdefault(ttf.getGlyphID(glyph), codepoint)
            pairs = [f"<{gid:04X}> <{chr(codepoint).encode('utf-16-be').hex().upper()}>" for gid, codepoint in sorted(glyph_to_unicode.items())]
            lines = ["/CIDInit /ProcSet findresource begin", "12 dict begin", "begincmap", "/CMapName /FontAware-Unicode def", "/CMapType 2 def", "1 begincodespacerange", "<0000> <FFFF>", "endcodespacerange"]
            for offset in range(0, len(pairs), 100):
                chunk = pairs[offset:offset + 100]
                lines += [f"{len(chunk)} beginbfchar", *chunk, "endbfchar"]
            lines += ["endcmap", "CMapName currentdict /CMap defineresource pop", "end", "end"]
            recovered = "\n".join(lines).encode("ascii")
            original = font["/ToUnicode"].get_data()
            stream = DecodedStreamObject()
            stream.set_data(recovered)
            font[NameObject("/ToUnicode")] = stream
            font_changes.append({"font_object": ref.idnum, "base_font": str(font["/BaseFont"]), "glyphs": len(pairs), "embedded_ttf_sha256": sha256(raw_ttf), "original_tounicode_sha256": sha256(original), "recovered_tounicode_sha256": sha256(recovered), "recovered_cid_unicode_mapping": {f"{gid:04X}": f"U+{codepoint:04X}" for gid, codepoint in sorted(glyph_to_unicode.items())}})
    writer = PdfWriter()
    writer.clone_document_from_reader(reader)
    memory_pdf = io.BytesIO()
    writer.write(memory_pdf)
    memory_pdf.seek(0)
    with pdfplumber.open(memory_pdf) as document:
        pages = [page.extract_text(x_tolerance=2, y_tolerance=3) or "" for page in document.pages]
        image_counts = [len(page.images) for page in document.pages]
    # Standard Symbol/Wingdings private-use characters, checked against the
    # rendered source. These are encoding repairs, not rewritten prose.
    symbol_replacements = {"\uf0b7": "\u2022", "\uf0e0": "\u2192", "\uf0a2": "\u2032", "\uf0bb": "\u2248"}
    symbol_counts = {f"U+{ord(old):04X}": sum(page.count(old) for page in pages) for old in symbol_replacements}
    pages = [text.translate(str.maketrans(symbol_replacements)) for text in pages]
    output = "\n\n".join(f"[[PDF PAGE {i + 1}; PRINTED PAGE {429 + i}]]\n{text}" for i, text in enumerate(pages)) + "\n"
    destination = directory / "topintzi_etal_2019_context.txt"
    destination.write_text(output, encoding="utf-8", newline="\n")
    record = {"source_pdf": source.name, "source_pdf_sha256": sha256(source.read_bytes()), "text_file": destination.name, "text_sha256": sha256(destination.read_bytes()), "pages": len(pages), "printed_page_range": [429, 447], "page_character_counts": [len(page) for page in pages], "raster_images_per_page": image_counts, "font_recovery": font_changes, "symbol_replacements": {f"U+{ord(old):04X}": new for old, new in symbol_replacements.items()}, "symbol_replacement_counts": symbol_counts, "procedure": "pdfplumber extraction after rebuilding Identity-H/Identity ToUnicode CMaps from each font's own embedded TrueType Unicode cmap in memory; no OCR or external-language-model rewriting", "limitations": ["Raster figures and labels encoded only in images are not extracted.", "Tables are linearized; reading order must be checked against the source PDF.", "Original header/footer text is retained; explicit page markers are added."]}
    (directory / "topintzi_etal_2019_extraction.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"pages": len(pages), "recovered_fonts": len(font_changes), "characters": len(output), "text_sha256": record["text_sha256"], "raster_images_per_page": image_counts}, indent=2))


if __name__ == "__main__":
    main()
