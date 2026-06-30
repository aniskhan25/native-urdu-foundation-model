"""Deterministic, auditable Urdu instruction records for SFT diagnostics."""

from __future__ import annotations

from collections import Counter


def math_records() -> list[dict[str, str]]:
    records = []
    for index in range(40):
        left = 125 + 17 * index
        right = 34 + 7 * index
        total = left + right
        records.append({
            "category": "math",
            "prompt": f"{left} اور {right} کو جمع کریں۔",
            "response": f"{left} + {right} = {total}۔ جواب: {total}",
        })

        starting = 600 + 23 * index
        spent = 120 + 5 * index
        remaining = starting - spent
        records.append({
            "category": "math",
            "prompt": f"ایک شخص کے پاس {starting} روپے تھے۔ اس نے {spent} روپے خرچ کیے۔ کتنے روپے باقی بچے؟",
            "response": f"باقی رقم کے لیے {starting} میں سے {spent} منہا کریں: {starting} - {spent} = {remaining}۔ جواب: {remaining} روپے۔",
        })

        packets = 3 + index % 8
        items = 12 + index
        product = packets * items
        records.append({
            "category": "math",
            "prompt": f"اگر {packets} ڈبوں میں ہر ڈبے کے اندر {items} اشیا ہوں تو کل اشیا کتنی ہوں گی؟",
            "response": f"کل اشیا معلوم کرنے کے لیے {packets} × {items} کریں: {packets} × {items} = {product}۔ جواب: {product} اشیا۔",
        })

        groups = 2 + index % 9
        per_group = 7 + index
        total_items = groups * per_group
        records.append({
            "category": "math",
            "prompt": f"{total_items} کتابیں {groups} برابر گروہوں میں تقسیم کی جائیں تو ہر گروہ میں کتنی کتابیں ہوں گی؟",
            "response": f"{total_items} ÷ {groups} = {per_group}۔ جواب: ہر گروہ میں {per_group} کتابیں ہوں گی۔",
        })

        price = 1000 + 100 * index
        discount = (5, 10, 15, 20, 25)[index % 5]
        reduction = price * discount // 100
        final_price = price - reduction
        records.append({
            "category": "math",
            "prompt": f"ایک دکان میں {price} روپے کی چیز پر {discount} فیصد رعایت ہے۔ ادا کی جانے والی رقم معلوم کریں۔",
            "response": f"{price} روپے کا {discount} فیصد {reduction} روپے ہے۔ {price} - {reduction} = {final_price}۔ جواب: {final_price} روپے۔",
        })

        hours = 2 + index % 5
        speed = 40 + index
        distance = hours * speed
        records.append({
            "category": "math",
            "prompt": f"ایک گاڑی {hours} گھنٹوں میں {distance} کلومیٹر سفر کرتی ہے۔ اس کی اوسط رفتار کیا ہے؟",
            "response": f"اوسط رفتار = فاصلہ ÷ وقت۔ {distance} ÷ {hours} = {speed}۔ جواب: {speed} کلومیٹر فی گھنٹہ۔",
        })
    return records


def translation_records() -> list[dict[str, str]]:
    subjects = (
        ("طالب علم", "The student"),
        ("استاد", "The teacher"),
        ("کسان", "The farmer"),
        ("انجینئر", "The engineer"),
        ("محقق", "The researcher"),
    )
    predicates = (
        ("روزانہ وقت پر پہنچتا ہے۔", "arrives on time every day."),
        ("احتیاط سے اپنا کام مکمل کرتا ہے۔", "completes his work carefully."),
        ("نئی مہارتیں سیکھتا ہے۔", "learns new skills."),
        ("اپنے ساتھیوں کی مدد کرتا ہے۔", "helps his colleagues."),
        ("صبح اخبار پڑھتا ہے۔", "reads the newspaper in the morning."),
        ("مشکل سوالات پر غور کرتا ہے۔", "thinks about difficult questions."),
        ("صاف پانی استعمال کرتا ہے۔", "uses clean water."),
        ("ہر ہفتے ایک کتاب خریدتا ہے۔", "buys a book every week."),
    )
    records = []
    for (urdu_subject, english_subject), (urdu_predicate, english_predicate) in (
        (subject, predicate) for subject in subjects for predicate in predicates
    ):
        urdu = f"{urdu_subject} {urdu_predicate}"
        english = f"{english_subject} {english_predicate}"
        records.extend((
            {
                "category": "translation",
                "prompt": f"اس جملے کا اردو ترجمہ کریں: {english}",
                "response": urdu,
            },
            {
                "category": "translation",
                "prompt": f"اس جملے کا انگریزی ترجمہ کریں: {urdu}",
                "response": english,
            },
        ))
    return records


def summary_records() -> list[dict[str, str]]:
    topics = (
        ("شہری ٹرانسپورٹ", "لوگوں کو کام اور تعلیم تک پہنچاتی ہے", "ٹریفک اور ناقص منصوبہ بندی", "بہتر بس سروس اور مربوط راستوں کی منصوبہ بندی"),
        ("صاف پانی", "صحت اور روزمرہ زندگی کے لیے بنیادی ضرورت ہے", "آلودگی اور ضیاع", "پانی کی صفائی اور ذمہ دارانہ استعمال"),
        ("بنیادی تعلیم", "بچوں کو علم اور ضروری مہارتیں دیتی ہے", "اساتذہ اور سہولتوں کی کمی", "اساتذہ کی تربیت اور اسکولوں میں سرمایہ کاری"),
        ("مقامی زراعت", "خوراک اور دیہی روزگار فراہم کرتی ہے", "پانی کی قلت اور مہنگے بیج", "موثر آب پاشی اور معیاری بیجوں کی فراہمی"),
        ("شمسی توانائی", "صاف بجلی پیدا کر کے ایندھن کی لاگت کم کرتی ہے", "ابتدائی تنصیب اور ذخیرے کی لاگت", "مناسب مالی معاونت اور گرڈ کی منصوبہ بندی"),
        ("عوامی صحت", "بیماریوں کی روک تھام اور علاج میں مدد دیتی ہے", "ادویات اور طبی عملے کی کمی", "بنیادی مراکز صحت کو مضبوط بنانے"),
        ("کتب خانے", "مطالعے اور تحقیق کے مواقع بڑھاتے ہیں", "نئی کتابوں اور ڈیجیٹل رسائی کی کمی", "کتابوں کی فراہمی اور آن لائن سہولتوں میں اضافے"),
        ("شہری صفائی", "بیماریاں کم کر کے ماحول بہتر بناتی ہے", "کچرے کا ناقص انتظام", "باقاعدہ جمع آوری اور عوامی آگاہی"),
        ("فنی تربیت", "نوجوانوں کو روزگار کے قابل ہنر دیتی ہے", "نصاب اور صنعت کی ضروریات میں فرق", "اداروں اور صنعت کے درمیان تعاون"),
        ("درختوں کی حفاظت", "ہوا، مٹی اور مقامی موسم کو بہتر بناتی ہے", "غیر قانونی کٹائی اور کم شجرکاری", "مقامی نگرانی اور مسلسل شجرکاری"),
    )
    contexts = (
        "حالیہ برسوں میں اس کی ضرورت بڑھی ہے۔",
        "عام شہری اس سے روزانہ فائدہ اٹھاتے ہیں۔",
        "اس شعبے میں پائیدار منصوبہ بندی ضروری ہے۔",
        "مقامی ادارے اس کام میں اہم کردار ادا کرتے ہیں۔",
        "بہتر نتائج کے لیے وسائل کا درست استعمال ضروری ہے۔",
        "اس مسئلے کا حل حکومت اور شہریوں کی مشترکہ کوشش چاہتا ہے۔",
    )
    summary_starts = ("مختصراً", "خلاصہ یہ ہے کہ", "مختصر طور پر", "مرکزی نکتہ یہ ہے کہ", "نتیجتاً", "اس عبارت کا خلاصہ یہ ہے کہ")
    records = []
    for topic, context, summary_start in zip(
        (topic for topic in topics for _ in contexts),
        contexts * len(topics),
        summary_starts * len(topics),
    ):
        subject, benefit, problem, action = topic
        text = f"{subject} {benefit}۔ {context} تاہم {problem} کی وجہ سے اس شعبے کی کارکردگی متاثر ہوتی ہے۔ {action} سے صورت حال بہتر ہو سکتی ہے۔"
        summary = f"{summary_start} {subject} کی اہمیت واضح ہے، مگر {problem} بڑی رکاوٹ ہے؛ {action} سے بہتری آ سکتی ہے۔"
        records.append({
            "category": "summary",
            "prompt": f"اس عبارت کا ایک جملے میں خلاصہ کریں: {text}",
            "response": summary,
        })
    return records


def style_correction_records() -> list[dict[str, str]]:
    pairs = (
        ("مجھے یہ کتاب پسند آیا۔", "مجھے یہ کتاب پسند آئی۔"),
        ("بچیوں نے اپنا نئی کتابیں الماری میں رکھ دیا۔", "بچیوں نے اپنی نئی کتابیں الماری میں رکھ دیں۔"),
        ("طلبہ نے اپنے سبق مکمل کیا۔", "طلبہ نے اپنے اسباق مکمل کیے۔"),
        ("وہ کل اسکول جائے گی تھا۔", "وہ کل اسکول گئی تھی۔"),
        ("ہم نے بازار سے تازہ سبزی خریدا۔", "ہم نے بازار سے تازہ سبزی خریدی۔"),
        ("استاد نے بچوں کو کہانی سنائے۔", "استاد نے بچوں کو کہانی سنائی۔"),
        ("میری بہن روز اخبار پڑھتا ہے۔", "میری بہن روز اخبار پڑھتی ہے۔"),
        ("یہ دونوں مسئلہ اہم ہے۔", "یہ دونوں مسائل اہم ہیں۔"),
        ("دروازے کھلا ہوا ہے۔", "دروازہ کھلا ہوا ہے۔"),
        ("تمام کھلاڑی میدان میں موجود تھا۔", "تمام کھلاڑی میدان میں موجود تھے۔"),
        ("اس نے مجھے دو کتاب دیا۔", "اس نے مجھے دو کتابیں دیں۔"),
        ("بچے باغ میں کھیل رہی ہیں۔", "بچے باغ میں کھیل رہے ہیں۔"),
        ("یہ خبریں درست نہیں ہے۔", "یہ خبریں درست نہیں ہیں۔"),
        ("میں نے کل ایک اچھا فلم دیکھی۔", "میں نے کل ایک اچھی فلم دیکھی۔"),
        ("لڑکی نے اپنا بستہ اٹھایا اور چلے گئے۔", "لڑکی نے اپنا بستہ اٹھایا اور چلی گئی۔"),
        ("ہماری ٹیم میچ جیت گیا۔", "ہماری ٹیم میچ جیت گئی۔"),
        ("کمرے میں تین کرسی موجود ہے۔", "کمرے میں تین کرسیاں موجود ہیں۔"),
        ("والدین نے بچے کی حوصلہ افزائی کیا۔", "والدین نے بچے کی حوصلہ افزائی کی۔"),
        ("یہ فیصلہ سب کے لیے فائدہ مند ہوں گے۔", "یہ فیصلہ سب کے لیے فائدہ مند ہوگا۔"),
        ("بارش کی وجہ سے سڑکیں بند ہو گیا۔", "بارش کی وجہ سے سڑکیں بند ہو گئیں۔"),
        ("علی اور زید بازار گیا۔", "علی اور زید بازار گئے۔"),
        ("اس مضمون میں کئی اہم نکتہ بیان ہوئے ہیں۔", "اس مضمون میں کئی اہم نکات بیان ہوئے ہیں۔"),
        ("مجھے آپ کی مشورہ پسند آیا۔", "مجھے آپ کا مشورہ پسند آیا۔"),
        ("وہ اپنے کام مکمل کر چکی ہے۔", "وہ اپنا کام مکمل کر چکی ہے۔"),
        ("کسان نے فصلوں کو پانی دیا اور گھر چلی گئی۔", "کسان نے فصلوں کو پانی دیا اور گھر چلا گیا۔"),
        ("یہ عمارت بہت پرانا ہے۔", "یہ عمارت بہت پرانی ہے۔"),
        ("ہمیں صاف پانی پینا چاہیے ہیں۔", "ہمیں صاف پانی پینا چاہیے۔"),
        ("ہر طالب علموں کو کتاب ملی۔", "ہر طالب علم کو کتاب ملی۔"),
        ("میٹنگ کل دس بجے شروع ہوں گی۔", "میٹنگ کل دس بجے شروع ہوگی۔"),
        ("لوگوں نے مسئلے کا حل تلاش کر لیا ہے تھے۔", "لوگوں نے مسئلے کا حل تلاش کر لیا تھا۔"),
    )
    records = []
    for incorrect, correct in pairs:
        records.extend((
            {
                "category": "style_correction",
                "prompt": f"اس جملے کی املا اور قواعد درست کریں: {incorrect}",
                "response": f"درست جملہ: {correct}",
            },
            {
                "category": "style_correction",
                "prompt": f"اس عبارت کو درست اردو میں لکھیں: {incorrect}",
                "response": f"بہتر صورت: {correct}",
            },
        ))
    return records


def code_switching_records() -> list[dict[str, str]]:
    moments = (
        ("آج کے کام", "آج"),
        ("صبح کے کام", "صبح"),
        ("میٹنگ کے بعد کے کام", "meeting کے بعد"),
        ("دوپہر کے کام", "دوپہر کو"),
        ("کام کے آغاز", "کام شروع کرتے وقت"),
        ("دن کے اختتام", "دن کے اختتام پر"),
    )
    sentences = (
        ("dataset", "ہماری team نے dataset clean کیا اور results کو dashboard پر update کر دیا۔"),
        ("deadline", "project manager نے deadline confirm کی اور سب کو نئی timeline بھیج دی۔"),
        ("model", "data team نے model کی performance check کر کے report share کر دی۔"),
        ("experiment", "research group نے experiment کے نتائج review کیے اور next step طے کیا۔"),
        ("feedback", "design team نے user feedback پڑھ کر interface کا نیا draft بنایا۔"),
        ("ticket", "support team نے customer کی request حل کر کے ticket close کر دیا۔"),
        ("code", "developer نے code test کیا اور تبدیلیاں main branch میں merge کر دیں۔"),
        ("budget", "manager نے budget approve کیا اور procurement process شروع کر دیا۔"),
        ("metrics", "analyst نے weekly metrics دیکھ کر presentation کا final version تیار کیا۔"),
        ("progress", "تمام colleagues نے task list دیکھی اور اپنی progress update کر دی۔"),
    )
    return [
        {
            "category": "code_switching",
            "prompt": f"{keyword} کا ذکر کرتے ہوئے دفتر کا ایک فطری اردو-English code-switching جملہ لکھیں جو {context} سے متعلق ہو۔",
            "response": f"{prefix} {sentence}",
        }
        for context, prefix in moments
        for keyword, sentence in sentences
    ]


def story_records() -> list[dict[str, str]]:
    names = ("علی", "حامد", "زید", "بلال", "عمر")
    settings = (
        ("رات کے وقت بند بازار", "دکانوں کے شٹر گرے ہوئے تھے"),
        ("بارش میں سنسان ریلوے اسٹیشن", "پلیٹ فارم پر پانی جمع تھا"),
        ("صبح سویرے پرانا کتب خانہ", "الماریوں پر گرد جمی ہوئی تھی"),
        ("شام کے وقت خالی اسکول", "راہداری میں خاموشی پھیلی ہوئی تھی"),
    )
    objects = (
        ("ایک گم شدہ خط", "خط پر نام تو تھا مگر پتا مٹ چکا تھا", "ملا"),
        ("ایک بند ڈبہ", "ڈبے کے اندر سے ہلکی سی آواز آ رہی تھی", "ملا"),
        ("ایک پرانی تصویر", "تصویر کے پیچھے آج کی تاریخ لکھی ہوئی تھی", "ملی"),
    )
    return [
        {
            "category": "story",
            "prompt": f"تین جملوں میں ایک کہانی شروع کریں جس کا مرکزی کردار {name} ہو اور جس میں {setting} اور {obj} شامل ہوں۔",
            "response": f"{name} {setting} پہنچا تو {detail}۔ اسے زمین پر {obj} {found}؛ {clue}۔ اس نے وہ چیز سنبھالی اور فیصلہ کیا کہ صبح ہونے سے پہلے اس راز کا سراغ لگائے گا۔",
        }
        for name in names
        for setting, detail in settings
        for obj, clue, found in objects
    ]


def evidence_records() -> list[dict[str, str]]:
    subjects = ("تاریخی", "سائنسی", "طبی", "معاشی", "تعلیمی", "قانونی", "ماحولیاتی", "صحافتی", "ادبی", "سماجی")
    evidence = ("اصل دستاویز", "قابل اعتماد تحقیق", "مستند حوالہ", "آزاد تصدیق")
    return [
        {
            "category": "evidence",
            "prompt": f"اگر کسی {subject} دعوے کے حق میں {proof} دستیاب نہ ہو تو جواب کیسے دینا چاہیے؟",
            "response": f"واضح کریں کہ اس {subject} دعوے کی تصدیق کے لیے {proof} دستیاب نہیں، اس لیے اسے ثابت شدہ حقیقت کے طور پر پیش نہ کریں۔ مزید معتبر شواہد ملنے تک غیر یقینی صورت حال اور معلومات کی حد صاف بیان کریں۔",
        }
        for subject in subjects
        for proof in evidence
    ]


CURATED_TASK_RECORDS = (
    math_records()
    + translation_records()
    + summary_records()
    + style_correction_records()
    + code_switching_records()
    + story_records()
    + evidence_records()
)

EXPECTED_CATEGORY_COUNTS = {
    "math": 240,
    "translation": 80,
    "summary": 60,
    "style_correction": 60,
    "code_switching": 60,
    "story": 60,
    "evidence": 40,
}


def category_counts() -> dict[str, int]:
    return dict(Counter(record["category"] for record in CURATED_TASK_RECORDS))
