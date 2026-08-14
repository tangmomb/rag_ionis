import re


IONIS_STM_TARGET = "Ionis-STM"
IONIS_STM_PATTERN = re.compile(
    r"(?<!\w)(?:(?:l['\u2019]?\s*)?(?:ionis|onis)[\s-]+stm|"
    r"ionis[\s-]+astm|"
    r"(?:l['\u2019]\s*)?(?:yonis|yaunis)[\s-]+stm|"
    r"unisystem|unisstm|ionisstm|unicef-cm|unicef-tm)"
    r"(?!\w)",
    re.IGNORECASE,
)


def normalize_ionis_stm_text(text):
    matched_sources = []

    def replace(match):
        matched_sources.append(match.group(0))
        return IONIS_STM_TARGET

    normalized = IONIS_STM_PATTERN.sub(replace, str(text))
    return normalized, matched_sources
