import re
import xml.etree.ElementTree as ET

from python.models.document import Section
from python.models.result import ParseResult

class XMLParser:
    """
    Parses XML using the stdlib ElementTree.
    Top-level child elements become L1 sections; their children become L2.
    """

    def parse(self, raw: str) -> ParseResult:
        try:
            root = ET.fromstring(raw)
        except ET.ParseError as exc:
            raise ValueError(f"Invalid XML: {exc}") from exc

        sections: list[Section] = []

        def _elem_text(elem: ET.Element) -> str:
            return " ".join(elem.itertext()).strip()

        for child in root:
            tag = re.sub(r"\{[^}]+\}", "", child.tag)  # strip namespace
            body = _elem_text(child)
            if body:
                sections.append(Section(title=tag, level=1, text=body, page_start=1))

            for grandchild in child:
                gtag = re.sub(r"\{[^}]+\}", "", grandchild.tag)
                gbody = _elem_text(grandchild)
                if gbody:
                    sections.append(Section(title=gtag, level=2, text=gbody, page_start=1))

        full_text = _elem_text(root)

        if not sections:
            sections = [Section(title=root.tag, level=1, text=full_text, page_start=1)]

        return ParseResult(
            text=full_text,
            sections=sections,
            extra_meta={"root_tag": root.tag},
        )