"""Create a small curated Urdu SFT seed split."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from data_pipeline.normalize_urdu import normalize_urdu


SEED_RECORDS: list[dict[str, str]] = [
    {
        "source": "seed/eval_prompt",
        "category": "economics",
        "prompt": "پاکستان کی معیشت کے اہم مسائل یہ ہیں:",
        "response": "پاکستان کی معیشت کے اہم مسائل میں مہنگائی، کم برآمدات، بیرونی قرضوں کا دباؤ، توانائی کی بلند لاگت، ٹیکس وصولی کی کمزوری اور پیداواری شعبے کی سست ترقی شامل ہیں۔ ان مسائل کے حل کے لیے مالی نظم و ضبط، صنعتی پیداوار، برآمدات اور تعلیم یافتہ افرادی قوت پر مسلسل کام ضروری ہے۔",
    },
    {
        "source": "seed/eval_prompt",
        "category": "ai",
        "prompt": "اردو زبان میں مصنوعی ذہانت کے استعمالات پر ایک مختصر مضمون لکھیں۔",
        "response": "مصنوعی ذہانت اردو زبان کے لیے کئی مفید کام کر سکتی ہے۔ یہ اردو متن کا خلاصہ بنا سکتی ہے، سوالوں کے جواب دے سکتی ہے، ترجمہ بہتر بنا سکتی ہے، تعلیمی مواد تیار کر سکتی ہے اور سرکاری یا طبی معلومات کو عام لوگوں کے لیے آسان زبان میں پیش کر سکتی ہے۔ اس کے لیے معیاری اردو ڈیٹا، درست املا اور ثقافتی تناظر کا خیال رکھنا ضروری ہے۔",
    },
    {
        "source": "seed/eval_prompt",
        "category": "education",
        "prompt": "تعلیم، تحقیق اور ٹیکنالوجی کسی معاشرے کی ترقی میں کیسے مدد دیتی ہیں؟",
        "response": "تعلیم لوگوں کو علم اور ہنر دیتی ہے، تحقیق مسائل کے نئے حل تلاش کرتی ہے، اور ٹیکنالوجی ان حلوں کو بڑے پیمانے پر استعمال کے قابل بناتی ہے۔ جب یہ تینوں شعبے ایک ساتھ مضبوط ہوں تو روزگار، صحت، زراعت، صنعت اور حکمرانی میں بہتری آتی ہے۔ اسی لیے پائیدار ترقی کے لیے تعلیم، تحقیق اور ٹیکنالوجی میں مسلسل سرمایہ کاری ضروری ہے۔",
    },
    {
        "source": "seed/eval_prompt",
        "category": "math",
        "prompt": "اگر ایک دکان دار نے 1200 روپے کی چیز پر 15 فیصد رعایت دی تو نئی قیمت کیا ہوگی؟",
        "response": "1200 روپے کا 15 فیصد 180 روپے ہے، کیونکہ 1200 × 0.15 = 180۔ رعایت کے بعد نئی قیمت 1200 - 180 = 1020 روپے ہوگی۔",
    },
    {
        "source": "seed/eval_prompt",
        "category": "water",
        "prompt": "ایک طالب علم نے پوچھا: پانی کی کمی پاکستان کے لیے کیوں اہم مسئلہ ہے؟",
        "response": "پانی کی کمی پاکستان کے لیے اہم مسئلہ ہے کیونکہ زراعت، پینے کا پانی، صنعت اور توانائی سب پانی پر منحصر ہیں۔ آبادی میں اضافہ، آبی ذخائر کی کمی، زیر زمین پانی کا کم ہونا اور موسمیاتی تبدیلی اس مسئلے کو بڑھاتے ہیں۔ اگر پانی کا بہتر انتظام نہ ہو تو خوراک، صحت اور معیشت پر برا اثر پڑتا ہے۔",
    },
    {
        "source": "seed/eval_prompt",
        "category": "code_switching",
        "prompt": "اردو اور English code-switching کی ایک مثال یہ ہے:",
        "response": "آج ہماری team نے model کی performance دیکھی اور فیصلہ کیا کہ اگلے experiment میں dataset کو بہتر clean کیا جائے۔ اس جملے میں اردو اور English دونوں زبانیں فطری انداز میں استعمال ہوئی ہیں، اس لیے یہ code-switching کی مثال ہے۔",
    },
    {
        "source": "seed/eval_prompt",
        "category": "literature",
        "prompt": "علامہ اقبال کی شاعری میں خودی کا تصور کیا معنی رکھتا ہے؟",
        "response": "علامہ اقبال کے ہاں خودی سے مراد انسان کی باطنی قوت، شعور، عزت نفس اور تخلیقی ارادہ ہے۔ اقبال چاہتے ہیں کہ انسان اپنی صلاحیت پہچانے، اخلاقی بلندی اختیار کرے اور غلامانہ سوچ کے بجائے خود اعتمادی کے ساتھ عمل کرے۔ خودی ان کے فلسفے میں فرد کی تعمیر اور اجتماعی بیداری دونوں کا مرکز ہے۔",
    },
    {
        "source": "seed/eval_prompt",
        "category": "story",
        "prompt": "ایک سادہ کہانی شروع کریں جس میں ایک طالب علم لائبریری میں ایک پرانی کتاب تلاش کرتا ہے۔",
        "response": "علی صبح سویرے کالج کی لائبریری پہنچا۔ الماریوں پر گرد جمی ہوئی تھی اور کمرے میں خاموشی پھیلی ہوئی تھی۔ ایک کونے میں اسے نیلی جلد والی ایک پرانی کتاب نظر آئی۔ جب اس نے کتاب کھولی تو پہلے صفحے پر کسی نامعلوم طالب علم کا لکھا ہوا پیغام موجود تھا۔",
    },
    {
        "source": "seed/math",
        "category": "math",
        "prompt": "ایک طالب علم کے پاس 750 روپے تھے۔ اس نے 240 روپے کی کتاب خریدی۔ اس کے پاس کتنے روپے بچیں گے؟",
        "response": "باقی رقم معلوم کرنے کے لیے 750 میں سے 240 منہا کریں۔ 750 - 240 = 510۔ اس لیے طالب علم کے پاس 510 روپے بچیں گے۔",
    },
    {
        "source": "seed/math",
        "category": "math",
        "prompt": "اگر ایک گاڑی 3 گھنٹے میں 180 کلومیٹر سفر کرے تو اوسط رفتار کیا ہوگی؟",
        "response": "اوسط رفتار فاصلے کو وقت پر تقسیم کرنے سے ملتی ہے۔ 180 کلومیٹر ÷ 3 گھنٹے = 60 کلومیٹر فی گھنٹہ۔ اس لیے گاڑی کی اوسط رفتار 60 کلومیٹر فی گھنٹہ ہوگی۔",
    },
    {
        "source": "seed/math",
        "category": "math",
        "prompt": "ایک چیز کی قیمت 2000 روپے ہے۔ اگر اس پر 10 فیصد ٹیکس لگے تو کل قیمت کیا ہوگی؟",
        "response": "2000 روپے کا 10 فیصد 200 روپے ہے۔ کل قیمت 2000 + 200 = 2200 روپے ہوگی۔",
    },
    {
        "source": "seed/summary",
        "category": "summary",
        "prompt": "اس متن کا خلاصہ لکھیں: پاکستان میں زراعت معیشت کا اہم شعبہ ہے۔ کپاس، گندم، چاول اور گنا بڑی فصلیں ہیں۔ پانی کی کمی اور جدید ٹیکنالوجی کی عدم دستیابی پیداوار کو متاثر کرتی ہے۔",
        "response": "خلاصہ: پاکستان کی معیشت میں زراعت اہم ہے، مگر پانی کی کمی اور جدید ٹیکنالوجی کی کمی فصلوں کی پیداوار کو متاثر کرتی ہے۔",
    },
    {
        "source": "seed/summary",
        "category": "summary",
        "prompt": "اس پیراگراف کا مختصر خلاصہ دیں: مصنوعی ذہانت سے تعلیم میں ذاتی رہنمائی، خودکار جانچ اور زبانوں کے ترجمے جیسے کام بہتر ہو سکتے ہیں، لیکن اس کے استعمال میں رازداری اور تعصب کے مسائل کا خیال رکھنا ضروری ہے۔",
        "response": "خلاصہ: مصنوعی ذہانت تعلیم کو بہتر بنا سکتی ہے، مگر اس کے استعمال میں رازداری اور تعصب جیسے خطرات کو سنجیدگی سے دیکھنا چاہیے۔",
    },
    {
        "source": "seed/style",
        "category": "style_correction",
        "prompt": "اس جملے کو بہتر اردو میں لکھیں: مجھے یہ بات بہت زیادہ پسند آیا۔",
        "response": "درست جملہ: مجھے یہ بات بہت پسند آئی۔",
    },
    {
        "source": "seed/style",
        "category": "style_correction",
        "prompt": "اس جملے کی املا اور انداز درست کریں: طالب علموں نے اپنے سبق مکمل کیا۔",
        "response": "درست جملہ: طالب علموں نے اپنا سبق مکمل کیا۔",
    },
    {
        "source": "seed/style",
        "category": "style_correction",
        "prompt": "اس جملے کو رسمی انداز میں لکھیں: یہ کام جلدی کر دو۔",
        "response": "رسمی انداز: براہ کرم یہ کام جلد مکمل کر دیں۔",
    },
    {
        "source": "seed/qa",
        "category": "qa",
        "prompt": "موسمیاتی تبدیلی سے پاکستان کو کون سے خطرات لاحق ہیں؟",
        "response": "موسمیاتی تبدیلی سے پاکستان کو سیلاب، خشک سالی، گرمی کی لہروں، گلیشیئرز کے پگھلنے، زرعی پیداوار میں کمی اور پانی کی قلت جیسے خطرات لاحق ہیں۔ ان خطرات کو کم کرنے کے لیے آبی انتظام، شجرکاری، موسمیاتی منصوبہ بندی اور مقامی سطح پر تیاری ضروری ہے۔",
    },
    {
        "source": "seed/qa",
        "category": "qa",
        "prompt": "اچھی تحقیق کے لیے بنیادی اصول کیا ہیں؟",
        "response": "اچھی تحقیق کے لیے واضح سوال، معتبر ذرائع، منظم طریقہ کار، شواہد کا دیانت دار تجزیہ اور نتائج کی شفاف وضاحت ضروری ہے۔ محقق کو اپنے مفروضات اور حدود بھی صاف بیان کرنی چاہییں۔",
    },
    {
        "source": "seed/qa",
        "category": "qa",
        "prompt": "صحت مند طرز زندگی کے تین اہم اصول بتائیں۔",
        "response": "صحت مند طرز زندگی کے تین اہم اصول متوازن غذا، باقاعدہ جسمانی سرگرمی اور مناسب نیند ہیں۔ اس کے ساتھ ذہنی دباؤ کو کم کرنا اور صاف پانی پینا بھی صحت کے لیے مفید ہے۔",
    },
    {
        "source": "seed/code_switching",
        "category": "code_switching",
        "prompt": "ایک فطری اردو-English code-switching جملہ لکھیں جس میں دفتر کا ماحول ہو۔",
        "response": "آج meeting میں manager نے کہا کہ report کا final draft شام تک share کر دیں تاکہ team کل presentation تیار کر سکے۔",
    },
    {
        "source": "seed/code_switching",
        "category": "code_switching",
        "prompt": "ایک فطری code-switching جملہ لکھیں جس میں AI project کا ذکر ہو۔",
        "response": "ہم نے Urdu dataset clean کر لیا ہے، اب next step model کی fine-tuning اور evaluation ہے۔",
    },
    {
        "source": "seed/safety",
        "category": "safety",
        "prompt": "اگر کسی سوال کا جواب معلوم نہ ہو تو کیا کرنا چاہیے؟",
        "response": "اگر جواب معلوم نہ ہو تو اندازہ لگانے کے بجائے صاف کہنا چاہیے کہ معلومات ناکافی ہیں۔ بہتر ہے کہ معتبر ذریعہ تلاش کیا جائے، سوال کو واضح کیا جائے، یا یہ بتایا جائے کہ جواب کس حد تک یقین سے دیا جا سکتا ہے۔",
    },
    {
        "source": "seed/safety",
        "category": "safety",
        "prompt": "طبی مشورے کے بارے میں احتیاطی جواب لکھیں۔",
        "response": "طبی مسئلے میں عمومی معلومات مددگار ہو سکتی ہیں، مگر تشخیص یا علاج کے لیے مستند ڈاکٹر سے رجوع کرنا ضروری ہے۔ اگر علامات شدید ہوں یا اچانک بگڑ جائیں تو فوری طبی امداد حاصل کریں۔",
    },
    {
        "source": "seed/safety",
        "category": "safety",
        "prompt": "مالی سرمایہ کاری کے سوال پر محتاط جواب کیسے دینا چاہیے؟",
        "response": "مالی سرمایہ کاری میں خطرہ شامل ہوتا ہے، اس لیے کسی ایک مشورے کو حتمی فیصلہ نہ سمجھیں۔ اپنی آمدنی، اخراجات، خطرہ برداشت کرنے کی صلاحیت اور مقامی قوانین کو دیکھ کر مستند مالی مشیر سے مشورہ کرنا بہتر ہے۔",
    },
    {
        "source": "seed/creative",
        "category": "story",
        "prompt": "دو جملوں میں ایک کہانی شروع کریں جس میں بارش اور ریلوے اسٹیشن ہو۔",
        "response": "بارش تیز ہو چکی تھی اور ریلوے اسٹیشن کی چھت سے پانی مسلسل ٹپک رہا تھا۔ حاشر نے اپنی پرانی گھڑی دیکھی اور سوچا کہ آج آنے والی گاڑی شاید اس کی زندگی بدل دے گی۔",
    },
    {
        "source": "seed/creative",
        "category": "story",
        "prompt": "ایک مختصر کہانی کا آغاز لکھیں جس میں ایک بچہ سائنس میلے میں جاتا ہے۔",
        "response": "حمنہ پہلی بار اپنے اسکول کے سائنس میلے میں شریک ہو رہی تھی۔ اس کے ہاتھ میں چھوٹا سا شمسی توانائی کا ماڈل تھا، مگر دل میں یہ خوف بھی تھا کہ کہیں وہ سب کے سامنے گھبرا نہ جائے۔",
    },
    {
        "source": "seed/essay",
        "category": "essay",
        "prompt": "کتاب پڑھنے کے فائدے مختصر بیان کریں۔",
        "response": "کتاب پڑھنے سے زبان بہتر ہوتی ہے، معلومات میں اضافہ ہوتا ہے اور سوچنے کی صلاحیت مضبوط ہوتی ہے۔ اچھی کتاب انسان کو نئے تجربات، مختلف خیالات اور بہتر اظہار کا موقع دیتی ہے۔ باقاعدہ مطالعہ طالب علموں کے لیے خاص طور پر مفید ہے۔",
    },
    {
        "source": "seed/essay",
        "category": "essay",
        "prompt": "شہری صفائی پر ایک مختصر پیراگراف لکھیں۔",
        "response": "شہری صفائی صحت مند معاشرے کی بنیادی ضرورت ہے۔ کچرا مناسب جگہ پھینکنا، نکاسی آب کا نظام بہتر رکھنا اور عوامی مقامات کو صاف رکھنا شہریوں اور اداروں دونوں کی ذمہ داری ہے۔ صاف شہر بیماریوں کو کم کرتے ہیں اور معیار زندگی بہتر بناتے ہیں۔",
    },
    {
        "source": "seed/translation",
        "category": "translation",
        "prompt": "اس جملے کا اردو ترجمہ کریں: Artificial intelligence can help students learn in their own language.",
        "response": "مصنوعی ذہانت طلبہ کو اپنی زبان میں سیکھنے میں مدد دے سکتی ہے۔",
    },
    {
        "source": "seed/translation",
        "category": "translation",
        "prompt": "اس جملے کا انگریزی ترجمہ کریں: پانی کی کمی زراعت کو براہ راست متاثر کرتی ہے۔",
        "response": "Water scarcity directly affects agriculture.",
    },
]


def normalize_record(record: dict[str, str]) -> dict[str, str]:
    normalized = {
        "prompt": normalize_urdu(record["prompt"]),
        "response": normalize_urdu(record["response"]),
        "source": record.get("source", "seed_sft_v1"),
    }
    if "category" in record:
        normalized["category"] = record["category"]
    return normalized


def validate_records(records: list[dict[str, str]]) -> None:
    prompts = set()
    for index, record in enumerate(records, 1):
        prompt = record.get("prompt", "").strip()
        response = record.get("response", "").strip()
        if not prompt or not response:
            raise ValueError(f"Record {index} requires non-empty prompt and response")
        if prompt in prompts:
            raise ValueError(f"Duplicate prompt: {prompt}")
        prompts.add(prompt)


def split_records(records: list[dict[str, str]], *, validation_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]
    if not train:
        raise ValueError("Split produced no training records")
    return train, validation


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a curated seed Urdu SFT train/validation split.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/sft"))
    parser.add_argument("--train-name", default="sft_train.jsonl")
    parser.add_argument("--validation-name", default="sft_val.jsonl")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = [normalize_record(record) for record in SEED_RECORDS]
    validate_records(records)
    train, validation = split_records(records, validation_fraction=args.validation_fraction, seed=args.seed)
    train_path = args.output_dir / args.train_name
    validation_path = args.output_dir / args.validation_name
    write_jsonl(train_path, train)
    write_jsonl(validation_path, validation)
    summary = {
        "records": len(records),
        "train": len(train),
        "validation": len(validation),
        "train_path": str(train_path),
        "validation_path": str(validation_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
