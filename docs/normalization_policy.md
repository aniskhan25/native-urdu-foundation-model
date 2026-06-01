# Urdu Normalization Policy

Version: `custom_urdu_nfkc_then_nfc_v1`

This policy must be used before tokenizer training and before pretraining data tokenization.

## Order

1. Apply Unicode NFKC.
2. Compose common decomposed Urdu forms:
   - `ا + ٓ` to `آ`
   - `و + ٔ` to `ؤ`
   - `ی + ٔ` to `ئ`
   - `ي + ٔ` to `ئ`
3. Map Arabic/Persian variants to Urdu-preferred codepoints.
4. Normalize Eastern Arabic and Urdu digits to ASCII digits.
5. Remove bidi controls.
6. Replace ZWNJ with a space and remove ZWJ.
7. Remove Arabic vowel marks and Quranic annotations by default.
8. Collapse whitespace.
9. Apply final NFC.

## Core Mappings

| Source | Target | Reason |
|---|---|---|
| `ي` | `ی` | Urdu uses Farsi Yeh |
| `ى` | `ی` | Common Yeh variant |
| `ك` | `ک` | Urdu uses Keheh |
| `ه` | `ہ` | Urdu Heh Goal |
| `ة` | `ہ` | Scraped Urdu often contains Teh Marbuta |
| `ۀ` | `ہ` | Normalize Heh variant |
| `إ`, `أ`, `ٱ` | `ا` | Conservative Alef normalization |
| `ـ` | empty | Remove tatweel |

## Notes

The `\u06AA` Swash Kaf mapping to `ک` is acceptable for a strict Urdu corpus. If Sindhi, Punjabi, or Shahmukhi material is intentionally included, revisit this policy before freezing the tokenizer.

