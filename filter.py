import re
import jieba
import unicodedata

from opencc import OpenCC


def simplified_sentence(text):
    # Create a converter for traditional to simplified Chinese
    cc = OpenCC("t2s")  # 't2s' stands for traditional to simplified

    return cc.convert(text)


# Train the tokenizer on text segmented by Jieba
def segmented_sentences(text):
    segmented = " ".join(jieba.cut(text))
    return segmented


def insert_spaces_between_numbers(text):
    return re.sub(r"(\d)", r" \1 ", text).replace("  ", " ").strip()


def is_chinese_char(char):
    code_point = ord(char)
    return (
        0x4E00 <= code_point <= 0x9FFF  # Basic CJK Unified Ideographs
        or 0x3400 <= code_point <= 0x4DBF  # CJK Unified Ideographs Extension A
        or 0x20000 <= code_point <= 0x2A6DF  # CJK Unified Ideographs Extension B
        or 0x2A700 <= code_point <= 0x2B73F  # CJK Unified Ideographs Extension C
        or 0x2B740 <= code_point <= 0x2B81F  # CJK Unified Ideographs Extension D
        or 0x2B820 <= code_point <= 0x2CEAF  # CJK Unified Ideographs Extension E
        or 0xF900 <= code_point <= 0xFAFF  # CJK Compatibility Ideographs
        or 0x2F800 <= code_point <= 0x2FA1F  # CJK Compatibility Ideographs Supplement
    )


def is_valid_chinese_char(char):
    # If it's a Chinese character, keep it
    if is_chinese_char(char):
        return True

    # Classify the character category using Unicode data
    category = unicodedata.category(char)
    # Unicode categories:
    # 'P' for punctuation (Pc, Pd, Ps, Pe, Pi, Pf, Po)
    # 'N' for numbers (Nd, Nl, No)
    if category[0] in ("P", "N"):
        return True

    return False


def clean_chinese_text(text):
    valid_text = "".join(char for char in text if is_valid_chinese_char(char))
    simplified_text = simplified_sentence(valid_text)
    segmented_text = segmented_sentences(simplified_text)
    spaced_text = insert_spaces_between_numbers(segmented_text)

    # print("seg", segmented_text)
    return spaced_text


def clean_english_text(text):
    # 1. Remove any characters not in the allowed set (letters, digits, spaces, and specified punctuation) and convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"-]', "", text).lower()

    # 2. Separate digits by spaces individually
    spaced_digits_text = re.sub(r"(\d)", r" \1 ", cleaned_text)

    # 3. Ensure there's a space before and after punctuation marks
    spaced_punctuation_text = re.sub(r'([.,!?\'"-])', r" \1 ", spaced_digits_text)

    # 4. Replace multiple spaces with a single space and trim leading/trailing spaces
    final_text = re.sub(r"\s+", " ", spaced_punctuation_text).strip()

    return final_text
