"""Build the sk-bench project page in Slovak and English.

Usage: python3 site/build.py
Writes site/dist/index.html (Slovak) and site/dist/en/index.html (English),
plus the overview figure and the leaderboard data as JSON and CSV.
The page is static HTML in the register of an interactive paper. It uses
system fonts only, so a visit makes no third-party requests. JavaScript adds
the quiz marking, the leaderboard controls, the theme switch and BibTeX copy.
"""
import csv
import html
import io
import json
import math
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
SITE_URL = "https://slovak-nlp.github.io/sk-bench/"
PAPER = "https://arxiv.org/abs/2610.09152"
LEADERBOARD = "https://huggingface.co/spaces/slovak-nlp/sk-bench-leaderboard"
CODE = "https://github.com/slovak-nlp/sk-bench"
EMNLP = "https://2026.emnlp.org/"
HF_DATA = "https://huggingface.co/collections/slovak-nlp/sk-bench"
CONTACT = "marek@suppa.sk"

# Paper snapshot, written by site/snapshot_leaderboard.py.
BOARD = json.loads((ROOT / "data" / "leaderboard.json").read_text())
MODELS = BOARD["models"]
TASK_KEYS = [t["key"] for t in BOARD["tasks"]]
BY_KEY = {m["key"]: m for m in MODELS}
SHOW = 15  # leaderboard rows visible before "show all"

e = html.escape


def task_score(model: dict, task: str) -> float:
    return model["tasks"][TASK_KEYS.index(task)]


def best_of(task: str, open_: bool | None = None) -> dict:
    pool = [m for m in MODELS if open_ is None or m["open"] == open_]
    return max(pool, key=lambda m: task_score(m, task))


# ── people ──────────────────────────────────────────────────────────────────
AFFIL = {
    "sk": ["Kempelenov inštitút inteligentných technológií, Bratislava",
           "Univerzita Komenského v Bratislave", "Technická univerzita v Košiciach",
           "Cisco Systems", "Vysoké učení technické v Brně"],
    "en": ["Kempelen Institute of Intelligent Technologies, Bratislava",
           "Comenius University in Bratislava", "Technical University of Košice",
           "Cisco Systems", "Brno University of Technology"],
}
AUTHORS = [
    ("Marek Šuppa", "2,4"), ("Ivan Vykopal", "1,5"), ("Andrej Ridzik", "1"),
    ("Kristián Sopkovič", "3"), ("Natália Kňažeková", "1,2"), ("Jaroslav Kopčan", "1"),
    ("Miroslav Blšták", "1"), ("Viktória Ondrejová", "4"), ("Daniel Hládek", "3"),
    ("Michal Gregor", "1"), ("Martin Tamajka", "1"), ("Marián Šimko", "1"),
]

BIBTEX = """@misc{suppa2026skbench,
  title         = {sk-bench: A Native-First Benchmark for Evaluating Large Language Models in Slovak},
  author        = {Marek Šuppa and Ivan Vykopal and Andrej Ridzik and Kristián Sopkovič and Natália Kňažeková and Jaroslav Kopčan and Miroslav Blšták and Viktória Ondrejová and Daniel Hládek and Michal Gregor and Martin Tamajka and Marián Šimko},
  year          = {2026},
  eprint        = {2610.09152},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.09152}
}"""

# ── task catalogue (paper appendix, Full Dataset Catalogue) ──────────────────
# key: (Slovak name, English name, provenance, new, items, Slovak note, English note)
PROV = {"N": ("pôvodne slovenské", "native Slovak"), "MT": ("preložené", "translated"),
        "MTA": ("preložené a upravené", "translated and adapted"), "G": ("generované", "LLM-generated")}
TASKS = {
    "arc": ("ARC-SK", "ARC-SK", "MT", False, 1169, "Vedecké otázky z ARC Challenge.", "ARC Challenge science questions."),
    "mmlu": ("Okapi-MMLU-SK", "Okapi-MMLU-SK", "MT", False, 13062, "Otázky z 57 odborov.", "Questions across 57 subjects."),
    "globalmmlu_lite": ("Global-MMLU-Lite", "Global-MMLU-Lite", "MT", False, 400, "Slovenská časť viacjazyčného testu znalostí.", "The Slovak part of a multilingual knowledge test."),
    "truthful_qa": ("TruthfulQA-SK", "TruthfulQA-SK", "MT", False, 778, "Otázky, pri ktorých sa ľudia často mýlia.", "Questions built around common misconceptions."),
    "scio": ("Scio", "Scio", "N", True, 1785, "Prijímacie testy z biológie, matematiky, chémie a všeobecného prehľadu.", "Admission tests in biology, maths, chemistry and general knowledge."),
    "financial_exams": ("Finančné skúšky NBS", "NBS financial exams", "N", False, 1334, "Odborná skúška pre finančný sektor.", "Certification exam for the financial sector."),
    "driving_test": ("Autoškola", "Driving test", "N", True, 701, "Otázky z testu na vodičský preukaz.", "Driving-licence exam questions."),
    "slovakcopa": ("SlovakCOPA", "SlovakCOPA", "MT", False, 500, "Príčina a následok v krátkych každodenných situáciách.", "Causes and effects in short everyday scenes."),
    "skj1": ("SKJ1", "SKJ1", "N", True, 100, "Výber gramaticky správneho tvaru.", "Choosing the grammatically correct form."),
    "slovak_trivia": ("Otázky o Slovensku", "Slovak trivia", "N", True, 353, "Otázky o slovenskej kultúre, zemepise a ďalších reáliách.", "Culture, geography and general knowledge about Slovakia."),
    "belebele": ("Belebele-SK", "Belebele-SK", "MT", False, 900, "Úlohy na porozumenie textu, ktoré preložil človek.", "Reading comprehension questions translated by people."),
    "chiby": ("Chiby", "Chiby", "N", True, 3432, "Je veta bez pravopisných chýb a so správnou diakritikou? Vety pochádzajú z opráv na slovenskej Wikipédii.", "Is the sentence free of spelling and diacritics errors? The sentences come from edits to Slovak Wikipedia."),
    "multi_wiki_qa": ("MultiWikiQA-SK", "MultiWikiQA-SK", "G", False, 4997, "Otázky k článkom na Wikipédii, ktoré vygeneroval jazykový model.", "Questions about Wikipedia articles, written by a language model."),
    "e_tests_mqa": ("e-testy, výber odpovede", "e-testy, multiple choice", "N", True, 444, "Stredoškolské testy z viacerých predmetov.", "High-school tests across subjects."),
    "e_tests_qa": ("e-testy, otvorená odpoveď", "e-testy, open answer", "N", True, 444, "Rovnaké otázky bez ponúknutých možností.", "The same questions without answer options."),
    "slovak_tests_mqa": ("Testy zo slovenčiny, výber odpovede", "Slovak-language tests, multiple choice", "N", True, 343, "Školské otázky na gramatiku, pravopis a slovnú zásobu.", "School questions on grammar, spelling and vocabulary."),
    "slovak_tests_qa": ("Testy zo slovenčiny, otvorená odpoveď", "Slovak-language tests, open answer", "N", True, 343, "Rovnaké otázky bez ponúknutých možností.", "The same questions without answer options."),
    "oneruler": ("OneRuler-SK", "OneRuler-SK", "N", True, 9000, "Vyhľadávanie informácií v dlhých slovenských textoch a uvažovanie nad nimi.", "Retrieval and reasoning over long Slovak texts."),
    "sklep_qa": ("skLEP-QA", "skLEP-QA", "N", False, 9583, "Odpoveď treba nájsť v článku z Wikipédie alebo zo spravodajstva.", "Answers found in Wikipedia or news articles."),
    "nli_sk": ("NLI-SK", "NLI-SK", "N", False, 597, "Vyplýva jedna veta z druhej? Odpovede označili ľudia.", "Does one sentence follow from another? Labelled by people."),
    "sklep_nli": ("skLEP-NLI", "skLEP-NLI", "MT", False, 5004, "Vyplýva jedna veta z druhej? Na výber sú tri možnosti.", "Does one sentence follow from another? There are three possible answers."),
    "sklep_rte": ("skLEP-RTE", "skLEP-RTE", "MT", False, 1660, "Vyplýva jedna veta z druhej? Odpoveď je áno alebo nie.", "Does one sentence follow from another? The answer is yes or no."),
    "senti_sk_3class": ("SentiSK, tri triedy", "SentiSK, three classes", "N", False, 4677, "Je text pozitívny, neutrálny alebo negatívny?", "Is the text positive, neutral or negative?"),
    "senti_sk_binary": ("SentiSK, dve triedy", "SentiSK, two classes", "N", False, 4677, "Je text pozitívny alebo negatívny?", "Is the text positive or negative?"),
    "sklep_sentiment_analysis": ("skLEP-sentiment", "skLEP-sentiment", "N", False, 1042, "Menšia kontrolná sada na určovanie sentimentu.", "A smaller second dataset for sentiment."),
    "movie_review": ("Recenzie filmov", "Movie reviews", "N", False, 2008, "Recenzie filmov a seriálov s hodnotením v hviezdičkách.", "Film and TV reviews with star ratings."),
    "toxic_sk": ("ToxicSK", "ToxicSK", "N", False, 881, "Rozpoznanie toxického textu.", "Detecting toxic text."),
    "sklep_hate_speech": ("skLEP-hate", "skLEP-hate", "N", False, 1319, "Rozpoznanie nenávistných prejavov.", "Detecting hate speech."),
    "slovaksum": ("SlovakSum", "SlovakSum", "N", False, 1500, "Zhrnutie správy v porovnaní so zhrnutím od človeka.", "News summaries compared with human-written ones."),
    "smesum": ("SME-Sum", "SME-Sum", "N", False, 1500, "Zhrnutie článkov z denníka SME.", "Summaries of articles from the SME daily."),
    "ifeval_sk": ("IFEval-SK", "IFEval-SK", "MTA", True, 537, "Presné pokyny, napríklad na dĺžku alebo formát. Kontroly sú prispôsobené slovenčine.", "Precise instructions on length or format, with checks adapted to Slovak."),
    "analogies_sk": ("Slovné analógie", "Word analogies", "N", True, 2651, "Doplnenie analógie medzi slovami.", "Completing analogies between words."),
    "slohry": ("Slohry", "Slohry", "N", True, 4004, "Slovné hry na slovnú zásobu a tvaroslovie.", "Word games on vocabulary and morphology."),
}
assert set(TASKS) == set(TASK_KEYS), set(TASKS) ^ set(TASK_KEYS)
METRIC = {"acc": ("presnosť", "accuracy"), "qem": ("zhoda odpovede", "answer match"),
          "rougeL": ("ROUGE-L", "ROUGE-L"), "prompt_level_strict_acc": ("splnené pokyny", "instructions met")}

# API cost of the paper's evaluation runs (paper, appendix on costs).
# few: the run also included the 3-shot setting, so it is not like for like.
COSTS = [("google__gemini-3.1-flash-lite-preview", 102.40, True), ("openai__gpt-5.4-mini", 286.96, True),
         ("anthropic.claude-opus-4-6", 1353.67, False), ("anthropic.claude-sonnet-4-6", 1560.83, True),
         ("openai__gpt-5.5", 2761.67, False), ("google__gemini-3.1-pro-preview", 3833.67, False)]

# The maturita row is not a paper task. It is the Slovak-language subset of
# the school exams, scored on the paper runs (see the press brief).
MATURITA = dict(gem=94.8, gpt=92.9, open=86.5, open_name="Gemma 4 31B")

QUESTIONS = [
    # models=(right, total): how many models answered this item correctly in
    # all three Slovak phrasings. Needs lighteval --save-details for all 55
    # models. None renders a visible placeholder.
    dict(models=None, src="driving_test", a=1,
         q="Musí byť vozidlo stojace za zníženej viditeľnosti na parkovisku osvetlené?",
         q_en="Must a vehicle parked in a car park be lit when visibility is reduced?",
         opts=["Áno, vždy musí byť osvetlené aspoň parkovacími svietidlami.",
               "Nie, nemusí byť osvetlené.",
               "Áno, ale len ak parkovisko nie je verejne osvetlené."],
         opts_en=["Yes, it must always have at least its parking lights on.",
                  "No, it does not need to be lit.",
                  "Yes, but only if the car park has no public lighting."]),
    dict(models=None, src="driving_test", a=0,
         q="Pri uzatvorenom chladiacom systéme chladiaca kvapalina vrie",
         q_en="In a sealed cooling system, the coolant boils",
         opts=["približne pri 110 °C.", "presne pri 100 °C.", "až pri 150 °C."],
         opts_en=["at about 110 °C.", "at exactly 100 °C.", "only at 150 °C."]),
    dict(models=None, src="financial_exams", a=1,
         q="K nadobudnutiu vlastníckeho práva k nehnuteľnosti dochádza",
         q_en="Ownership of real estate is acquired",
         opts=["nadobudnutím platnosti kúpnej alebo inej zmluvy",
               "vkladom do katastra nehnuteľností podľa osobitného predpisu",
               "podpisom kúpnej alebo inej zmluvy",
               "podpisom kúpnej alebo inej zmluvy s overením podpisov kupujúceho aj predávajúceho pred notárom",
               "spísaním notárskej zápisnice o zmluve o prevode nehnuteľnosti"],
         opts_en=["when the purchase or other contract becomes valid",
                  "by registration in the land register under a special regulation",
                  "by signing the purchase or other contract",
                  "by signing the contract with both signatures certified by a notary",
                  "by drawing up a notarial record of the transfer contract"]),
    dict(models=None, src="financial_exams", a=3,
         q="Blízkou osobou v priamom rade je",
         q_en="A close relative in the direct line is",
         opts=["bratranec", "otcov brat", "druh/družka", "syn", "neter"],
         opts_en=["a cousin", "an uncle on the father's side", "an unmarried partner", "a son", "a niece"]),
    dict(models=None, src="maturita", a=2,
         q="Taliančina patrí do rovnakej jazykovej skupiny indoeurópskych jazykov ako",
         q_en="Italian belongs to the same group of Indo-European languages as",
         opts=["dánčina.", "nemčina.", "rumunčina.", "bulharčina."],
         opts_en=["Danish.", "German.", "Romanian.", "Bulgarian."]),
    dict(models=None, src="maturita", a=3,
         q="V ktorej možnosti sa nachádza vokalizovaná predložka?",
         q_en="Which option contains a vocalized preposition (one with an added vowel, such as s becoming so)?",
         opts=["za kladné vybavenie", "na osobnom pohovore",
               "dorozumiem sa aj po maďarsky", "skúsenosti so spoluorganizovaním"],
         opts_en=None),
]

DATASETS = [
    ("https://huggingface.co/datasets/slovak-nlp/driving-test-dataset-slovak", "driving-test-dataset-slovak"),
    ("https://huggingface.co/datasets/TUKE-KEMT/slovak-financial-exam", "slovak-financial-exam"),
    ("https://huggingface.co/datasets/slovak-nlp/slovak-school-exams-test-preprocessed", "slovak-school-exams"),
]

# ── strings ─────────────────────────────────────────────────────────────────
T = {
    "sk": dict(
        lang="sk", other="en", other_href="en/", other_label="English", root="",
        title="Ako dobre rozumie umelá inteligencia slovenčine? | sk-bench",
        description="sk-bench meria, ako jazykové modely ovládajú slovenčinu. Porovnali sme 55 modelov na 30 dátových sadách, od autoškoly a maturity po úlohy, v ktorých AI zaostáva.",
        short="Ako dobre rozumie AI slovenčine",
        nav=[("prehlad", "Prehľad"), ("vyskusajte", "Vyskúšajte"), ("zistenia", "Zistenia"), ("rebricek", "Rebríček"),
             ("ulohy", "Úlohy"), ("metodika", "Metodika"), ("media", "Pre médiá"), ("citovat", "Citovať")],
        theme="Prepnúť medzi svetlým a tmavým režimom",
        h1="Ako dobre rozumie umelá inteligencia slovenčine?",
        og_locale="sk_SK",
        og_alt="sk-bench: Ako dobre rozumie umelá inteligencia slovenčine? Najlepší komerčný model Gemini 3.1 Pro získal 72,6 bodu zo 100, najlepší model s otvorenými váhami gpt-oss 120B 60,0 bodu.",
        subtitle="sk-bench: súbor testov na hodnotenie veľkých jazykových modelov v slovenčine",
        corr="Korešpondenčný autor",
        btn_paper="Prečítať štúdiu", btn_board="Pozrieť rebríček", btn_code="Kód a dáta", btn_cite="Citovať",
        venue="Štúdiu sme predstavili na konferencii {emnlp}.",
        ov_h2="Čo sme zistili",
        ov_lead="Zostavili sme prvý veľký súbor testov, ktorý meria, ako umelá inteligencia ovláda slovenčinu, a otestovali sme na ňom 55 modelov. Najlepšie z nich odpovedajú na otázky z autoškoly či maturity takmer bez chyby. Pri úlohách typických pre slovenčinu však ešte zaostávajú a modely s otvorenými váhami dosahujú výrazne horšie výsledky ako komerčné.",
        cards=[
            ("98 %", "Testy pre ľudí", "Gemini 3.1 Pro odpovedal správne na 98 % otázok z autoškoly, na 96 % otázok z finančnej skúšky NBS a na 95 % otázok z maturity zo slovenčiny.", "z1", "Zistenie 1"),
            ("3 zo 4", "Pravopis a diakritika", "Modely majú určiť, či je slovenská veta napísaná bez chýb. Aj tie najlepšie odpovedajú správne len približne v troch prípadoch zo štyroch.", "z2", "Zistenie 2"),
            ("12,6", "bodu delí najlepší komerčný model a najlepší model s otvorenými váhami", "Najlepší komerčný model získal v celkovom hodnotení 72,6 bodu zo 100, najlepší model s otvorenými váhami 60,0 bodu.", "z3", "Zistenie 3"),
        ],
        stats=[("55", "modelov"), ("30", "dátových sád"), ("33", "hodnotených úloh"), ("11", "nových dátových sád")],
        try_h2="Vyskúšajte si to sami",
        try_intro="Vybrali sme šesť skutočných otázok z testov, na ktorých sme hodnotili modely. Sú len na ukážku, nie na porovnanie človeka s modelom. Kliknite na odpoveď, ktorú považujete za správnu.",
        quiz_more="Zobraziť ďalšie tri otázky", right="Správne.", wrong="Správne je {x}).",
        src=dict(driving_test="Autoškola", financial_exams="Finančná skúška NBS", maturita="Maturita zo slovenčiny, 2024"),
        stat_q="Na túto otázku odpovedalo správne {n} z {total} modelov.",
        stat_todo="Doplniť: koľko z 55 modelov odpovedalo na túto otázku správne.",
        f_h2="Zistenia",
        f_intro="Toto sú tri hlavné zistenia zo štúdie. Pri každom uvádzame aj dáta, z ktorých vychádza.",
        more_h2="Ďalšie zistenia", more_intro="Pozreli sme sa aj na to, ako výsledky súvisia s veľkosťou modelu, cenou, doučením a jazykom zadania. Tieto zistenia platia pre modely z našej štúdie a nemusia platiť všeobecne.",
        brief="V skratke",
        f1_h="Testy pre ľudí zvládajú takmer bez chyby",
        f1_brief="Na otázky z autoškoly, finančnej skúšky NBS a maturity odpovedajú najlepšie modely správne v 95 až 98 % prípadov.",
        f1_p="Najlepšie modely, Gemini 3.1 Pro od Googlu a GPT-5.5 od OpenAI, odpovedali správne aspoň na 96 % otázok z autoškoly aj z finančnej skúšky NBS. Celé skúšky sme nesimulovali, takže tieto percentá nie sú body zo skúšky. Zisťovali sme, na koľko otázok z verejných databáz model odpovie správne. Tieto otázky sa navyše mohli objaviť v dátach, na ktorých sa modely učili.",
        f2_h="Pri úlohách typických pre slovenčinu ešte zaostávajú",
        f2_brief="Keď majú modely posúdiť pravopis a diakritiku slovenskej vety, aj ten najlepší odpovedá správne len v 75 % prípadov.",
        f2_p="Keď modely zvládajú testy pre ľudí takmer bez chyby, tieto testy už neodlíšia dobrý model od výborného. Preto sk-bench obsahuje aj úlohy, v ktorých AI stále zaostáva. V úlohe Chiby majú modely určiť, či je slovenská veta napísaná bez pravopisných chýb a so správnou diakritikou. Pri otázkach o Slovensku je zasa veľký rozdiel medzi komerčnými modelmi a modelmi s otvorenými váhami.",
        legend_open="najlepší model s otvorenými váhami", best="najlepší model",
        open_note="Najlepší model s otvorenými váhami, {name}, dosiahol {v}.", best_any="Spomedzi všetkých modelov bol najlepší {name} s výsledkom {v}.",
        axis_note="Os grafu začína na 50 %. Každý model dostal každú úlohu v troch slovenských zneniach. Uvádzame priemer jeho výsledkov.",
        rows=dict(
            driving_test=("Autoškola", "701 otázok"),
            financial_exams=("Odborná finančná skúška NBS", "1 334 otázok"),
            maturita=("Maturita zo slovenčiny", "369 testových otázok"),
            slovak_trivia=("Otázky o Slovensku", "353 otázok o kultúre, zemepise a reáliách"),
            chiby=("Chyby v pravopise a diakritike", "3 432 viet, pri každej má model určiť, či je bez chýb"),
        ),
        f3_h="Najlepší model s otvorenými váhami zaostáva o 12,6 bodu",
        f3_brief="Najlepší model s otvorenými váhami, gpt-oss 120B, získal 60,0 bodu zo 100. Najlepší komerčný model, Gemini 3.1 Pro, získal 72,6 bodu.",
        f3_p1="Komerčné modely, ako Gemini alebo GPT, bežia len na serveroch svojich výrobcov. Model s otvorenými váhami si firma, nemocnica či úrad môže stiahnuť a spustiť na vlastnom serveri. Váhy sú čísla, v ktorých je uložené všetko, čo sa model naučil. Takýto model však nemusí byť open source v plnom zmysle slova, pretože tréningové dáta ani kód na tréning často zverejnené nie sú.",
        f3_p2="Všetky špičkové modely s otvorenými váhami pochádzajú zo zahraničia. Kto chce AI prevádzkovať na vlastnom hardvéri, napríklad aby údaje neopustili organizáciu, dnes pri slovenčine dosiahne citeľne horšie výsledky. Podmienky použitia týchto modelov sa líšia, preto ich treba overiť pri každom modeli zvlášť.",
        bars1="Celkové skóre, zo 100 bodov", bars2="Otázky o Slovensku, percento správnych odpovedí",
        bar_com="najlepší komerčný", bar_open="najlepší s otvorenými váhami",
        f4_h="Väčší model nie je automaticky lepší",
        f4_brief="Mistral Large 3 so 675 miliardami parametrov získal 45,0 bodu. Oveľa menší Qwen3.5 s 9 miliardami parametrov získal 47,9 bodu.",
        f4_p="Graf ukazuje všetkých 49 modelov s otvorenými váhami podľa veľkosti. Väčšie modely sú v priemere lepšie, no rozdiely medzi nimi sú veľké. Pri rovnakej veľkosti sa výsledky líšia aj o 30 bodov.",
        f4_x="Počet parametrov v miliardách, logaritmická os", f4_y="Celkové skóre",
        f4_line="Gemini 3.1 Pro, najlepší komerčný model",
        f4_note="Pri modeloch typu mixture of experts, ako sú Qwen3.5 397B-A17B alebo Kimi K2.6, uvádzame celkový počet parametrov. Pri jednej odpovedi sa používa len časť z nich.",
        f5_h="Lacnejší model môže stačiť",
        f5_brief="Gemini 3.1 Flash-Lite získal 66,7 bodu a celé hodnotenie stálo 102 dolárov. Gemini 3.1 Pro získal 72,6 bodu za 3 834 dolárov.",
        f5_p="Pri komerčných modeloch sme platili za každú požiadavku. Uvádzame cenu celého hodnotenia v našej štúdii.",
        f5_cols=("Model", "Cena hodnotenia", "Celkové skóre"),
        f5_few="vrátane hodnotenia s ukážkovými príkladmi",
        f5_note="Flash-Lite, GPT-5.4 mini a Claude Sonnet 4.6 sme hodnotili aj s troma ukážkovými príkladmi. Ich cena preto zahŕňa viac hodnotení. Ostatné tri modely sme hodnotili len bez ukážok. Rozdiel v cene za rovnakú prácu je teda ešte väčší.",
        f6_h="Doučenie na slovenských textoch môže model zhoršiť",
        f6_brief="Model Qwen3 14B po doučení na slovenských textoch klesol zo 47,4 na 33,5 bodu. Po doladení na malej sade slovenských pokynov získal späť asi tri štvrtiny stratených bodov.",
        f6_p="Doučiť model s otvorenými váhami na slovenských textoch sa javí ako prirodzený spôsob, ako zlepšiť jeho slovenčinu. V našom pokuse to však nefungovalo hladko. Po ďalšom tréningu na slovenských textoch model horšie plnil zadania. Keď sme ho potom doladili na malej sade slovenských pokynov (Alpaca), celkové skóre stúplo na 43,8 bodu. Výsledok každého takého pokusu treba preto dôkladne overiť.",
        f6_chart="Celkové skóre po doučení kleslo a po doladení opäť čiastočne stúplo",
        f6_rows=("Qwen3 14B, pôvodný model", "po doučení na slovenských textoch", "po doladení na slovenských pokynoch"),
        f7_h="Anglické zadania takmer nepomáhajú",
        f7_brief="Pri anglickom zadaní bol priemerný výsledok v porovnateľných úlohách len o 0,4 bodu zo 100 lepší ako pri slovenskom. Pri porozumení textu stúpol o 1,6 bodu, pri gramatike klesol o 2,6 bodu.",
        f7_p="Každý model dostal tie isté úlohy aj s anglickým zadaním. Texty, otázky a odpovede zostali slovenské. Pri gramatike dosiahli modely s anglickým zadaním dokonca horšie výsledky. Tabuľka uvádza priemer výsledkov všetkých 55 modelov. Úlohy na plnenie pokynov a Slohry sú len po slovensky, preto v tabuľke chýbajú.",
        f7_cols=("Skupina úloh", "Slovenské zadanie", "Anglické zadanie", "Rozdiel"),
        f7_overall="Celkovo",
        lb_h2="Rebríček modelov",
        lb_intro="Rebríček obsahuje všetkých 55 modelov zo štúdie v čase jej vydania. Skóre je zo 100 bodov. Skóre skupiny je priemer výsledkov jej úloh so slovenskými zadaniami. Kliknutím na názov stĺpca zmeníte poradie. Ak chcete vidieť jednotlivé testy, prepnite zobrazenie na všetkých 33 úloh.",
        lb_view="Zobrazenie", lb_v_groups="Skupiny úloh", lb_v_tasks="Všetkých 33 úloh",
        lb_filter="Zobraziť modely", lb_all="Všetky", lb_open="S otvorenými váhami", lb_com="Komerčné",
        lb_search="Hľadať model", lb_leg_open="model s otvorenými váhami", lb_leg_com="komerčný model",
        lb_model="Model", lb_overall="Celkovo",
        lb_groups=[("Znalosti", "Znalosti a uvažovanie"), ("Porozumenie", "Porozumenie textu a odpovede na otázky"),
                   ("Klasifikácia", "Klasifikácia textu"), ("Generovanie", "Generovanie textu a zhrnutia"),
                   ("Pokyny", "Plnenie pokynov"), ("Gramatika", "Gramatika a tvaroslovie")],
        lb_more="Zobraziť všetky ({n})", lb_less="Zobraziť menej", lb_none="Nenašli sa žiadne modely.",
        lb_note="Celkové skóre je priemer výsledkov šiestich skupín úloh. Pri generovaní hodnotíme zhodu so vzorovým zhrnutím (ROUGE-L), preto sú tieto skóre nižšie. Najlepšie skóre v každom stĺpci je zvýraznené. Do <a href=\"" + LEADERBOARD + "\">rebríčka na Hugging Face</a> priebežne pridávame nové modely, preto sa jeho výsledky môžu líšiť od výsledkov v štúdii.",
        lb_dl="Stiahnuť dáta", lb_csv="CSV", lb_json="JSON",
        t_h2="30 dátových sád v šiestich skupinách",
        t_intro="Pri každej úlohe uvádzame pôvod dát, počet položiek, metriku a najlepší výsledok. Pás vpravo ukazuje výsledky všetkých 55 modelov na stupnici od 0 do 100. Niektoré dátové sady hodnotíme dvoma spôsobmi, preto je úloh 33.",
        t_cols=("Úloha", "Pôvod dát", "Položiek", "Metrika", "Najlepší model", "Najlepší s otvorenými váhami", "Všetky modely"),
        t_new="nová", t_new_long="nová dátová sada alebo sada prvýkrát použitá na hodnotenie jazykových modelov",
        m_h2="Ako hodnotíme",
        m_intro="Všetky modely, komerčné aj s otvorenými váhami, dostávajú rovnaké zadania a hodnotíme ich rovnakým postupom v nástroji lighteval. Kde sa dalo, uprednostnili sme pôvodné slovenské dáta pred prekladom.",
        m_row1="Jedna úloha", m_row2="Od úlohy k rebríčku",
        m_steps1=[("Úloha", "napríklad 701 otázok z autoškoly"), ("Tri znenia zadania", "po slovensky, bez ukážkových príkladov"),
                  ("Model", "rovnaký postup pre všetky modely"), ("Odpoveď", "písmeno, slovo, veta alebo celý text"),
                  ("Metrika", "presnosť, zhoda odpovede, ROUGE-L alebo kontrola pokynov")],
        m_steps2=[("Skóre úlohy", "priemer troch znení zadania"), ("Skóre skupiny", "priemer výsledkov jej úloh"),
                  ("Celkové skóre", "priemer výsledkov šiestich skupín")],
        fig_caption="Desať kategórií a 30 dátových sád v sk-bench. Hviezdička označuje nové dátové sady. Štítok pri úlohe ukazuje, či dáta vznikli v slovenčine, prekladom, generovaním alebo úpravou. Obrázok pochádza zo štúdie.",
        p_h2="Pre médiá",
        p_intro="Čísla, ktoré môžete citovať, aj s tým, z čoho vychádzajú.",
        p_facts=[
            ("98,3 %", "otázok z autoškoly zodpovedal Gemini 3.1 Pro správne (701 otázok)"),
            ("96,3 %", "otázok z odbornej finančnej skúšky NBS (1 334 otázok)"),
            ("94,8 %", "testových otázok z maturity zo slovenčiny (369 otázok)"),
            ("75 %", "viet správne posúdil najlepší model pri kontrole pravopisu a diakritiky"),
            ("72,6 a 60,0", "bodu zo 100 získali v celkovom hodnotení najlepší komerčný model a najlepší model s otvorenými váhami"),
            ("94 % a 69 %", "otázok o Slovensku zodpovedali správne Gemini 3.1 Pro a najlepší model s otvorenými váhami"),
            ("55", "modelov sme porovnali na 30 dátových sadách z desiatich oblastí"),
        ],
        p_note_h="Poznámka pre redakcie",
        p_note="Percentá udávajú, na koľko otázok model odpovedal správne. Celé skúšky sme nesimulovali. Celkové skóre aj výsledky pri otázkach o Slovensku pochádzajú zo štúdie, v ktorej sme porovnali 55 modelov. Do rebríčka na Hugging Face pribúdajú nové modely, preto sa jeho výsledky môžu líšiť.",
        p_fig_h="Obrázok na stiahnutie", p_fig="Prehľad desiatich kategórií a 30 dátových sád (SVG)",
        p_links_h="Odkazy",
        c_h2="Citácia", copy="Kopírovať", copied="Skopírované", authors_h3="Autori",
        links=[(PAPER, "Štúdia na arXiv"), (LEADERBOARD, "Rebríček na Hugging Face"), (CODE, "Kód na GitHube"), (HF_DATA, "Dátové sady na Hugging Face")],
        foot1="sk-bench pripravili výskumníci z Kempelenovho inštitútu inteligentných technológií (KInIT), Univerzity Komenského, Technickej univerzity v Košiciach a ďalších partnerských organizácií.",
        foot2="Otázky na tejto stránke pochádzajú z dátových sád {d1} (CC BY 4.0), {d2} od TUKE KEMT (CC BY-SA 4.0) a {d3} (CC BY 4.0). Stránka nepoužíva súbory cookie ani externé písma. Stav k 3. 11. 2026.",
    ),
    "en": dict(
        lang="en", other="sk", other_href="../", other_label="Slovensky", root="../",
        title="How well does AI understand Slovak? | sk-bench",
        description="sk-bench measures how well language models handle Slovak. We tested 55 models on 30 datasets, from driving tests and school exams to tasks where AI falls short.",
        short="How well does AI understand Slovak",
        nav=[("overview", "Overview"), ("try", "Try it"), ("findings", "Findings"), ("leaderboard", "Leaderboard"),
             ("tasks", "Tasks"), ("method", "Method"), ("press", "Press"), ("cite", "Cite")],
        theme="Switch between light and dark mode",
        h1="How well does AI understand Slovak?",
        og_locale="en_US",
        og_alt="sk-bench: How well does AI understand Slovak? The best commercial model, Gemini 3.1 Pro, scored 72.6 out of 100. The best open model, gpt-oss 120B, scored 60.0.",
        subtitle="sk-bench: A Native-First Benchmark for Evaluating Large Language Models in Slovak",
        corr="Corresponding author",
        btn_paper="Read the paper", btn_board="See the leaderboard", btn_code="Code and data", btn_cite="Cite",
        venue="The paper was presented at {emnlp}.",
        ov_h2="What we found",
        ov_lead="We built the first large benchmark that measures how well AI handles Slovak, and tested 55 models on it. The best ones answer driving-test and school-exam questions almost perfectly. They still struggle with tasks specific to Slovak, and open models remain well behind commercial ones.",
        cards=[
            ("98%", "Exam questions", "Gemini 3.1 Pro answered 98% of driving-test questions correctly, 96% of NBS financial exam questions and 95% of Slovak maturita questions.", "f1", "Finding 1"),
            ("3 in 4", "Spelling and diacritics", "When asked whether a Slovak sentence is free of errors, even the best models are right only about three times in four.", "f2", "Finding 2"),
            ("12.6", "point gap between the best commercial and the best open model", "The best commercial model scored 72.6 out of 100 overall, and the best open model scored 60.0.", "f3", "Finding 3"),
        ],
        stats=[("55", "models"), ("30", "datasets"), ("33", "scored tasks"), ("11", "new datasets")],
        try_h2="Try it yourself",
        try_intro="Here are six real questions from the tests we used, in the original Slovak with an English translation. They are only examples and do not compare people with models. Click the answer you think is right.",
        quiz_more="Show three more questions", right="Correct.", wrong="The answer is {x}).",
        src=dict(driving_test="Driving test", financial_exams="NBS financial exam", maturita="Maturita in Slovak, 2024"),
        stat_q="{n} of {total} models answered this question correctly.",
        stat_todo="To fill in: how many of the 55 models answered this question correctly.",
        f_h2="Findings",
        f_intro="These are the three main findings of the paper, each with the data behind it.",
        more_h2="More findings", more_intro="We also looked at model size, cost, further training and the language of the prompt. These results come from one study and one set of models, so they may not hold in general.",
        brief="In brief",
        f1_h="The best models get almost every exam question right",
        f1_brief="On the driving test, the NBS financial exam and the maturita, the best models answer 95 to 98% of questions correctly.",
        f1_p="The best models, Google's Gemini 3.1 Pro and OpenAI's GPT-5.5, answered at least 96% of the driving-test and NBS exam questions correctly. We did not simulate full exams, so these percentages are not exam scores. We measured how many questions from the public question banks each model gets right. These questions may also have appeared in the models' training data.",
        f2_h="Tasks specific to Slovak still trip models up",
        f2_brief="When asked to judge the spelling and diacritics of a Slovak sentence, even the best model is right only 75% of the time.",
        f2_p="Once models answer almost every exam question correctly, those exams no longer tell a good model from a great one. That is why sk-bench also includes tasks where AI still struggles. In Chiby, models have to say whether a Slovak sentence is free of spelling and diacritics errors. Questions about Slovakia show a large gap between commercial and open models.",
        legend_open="best open model", best="best model",
        open_note="The best open model, {name}, scored {v}.", best_any="The best of all models was {name}, with {v}.",
        axis_note="The axis starts at 50%. Each model got every task in three Slovak phrasings, and we report the average.",
        rows=dict(
            driving_test=("Driving test", "701 questions"),
            financial_exams=("NBS financial services exam", "1,334 questions"),
            maturita=("Slovak language maturita", "369 multiple-choice questions"),
            slovak_trivia=("Questions about Slovakia", "353 questions on culture, geography and everyday facts"),
            chiby=("Spelling and diacritics errors", "3,432 sentences, each to be judged error-free or not"),
        ),
        f3_h="The best open model trails by 12.6 points",
        f3_brief="The best open model, gpt-oss 120B, scored 60.0 out of 100. The best commercial one, Gemini 3.1 Pro, scored 72.6.",
        f3_p1="Commercial models such as Gemini or GPT run only on their makers' servers. A company, a hospital or a public office can download an open model and run it on its own server.",
        f3_p2="All of the top open models come from outside Slovakia. Anyone who wants to run AI on their own hardware, for example to keep data inside their organisation, currently gets noticeably worse results in Slovak. Licence terms differ, so they need checking for each open model.",
        bars1="Overall score, out of 100", bars2="Questions about Slovakia, percent correct",
        bar_com="best commercial", bar_open="best open",
        f4_h="Bigger models are not always better",
        f4_brief="Mistral Large 3, with 675 billion parameters, scored 45.0. The much smaller Qwen3.5, with 9 billion, scored 47.9.",
        f4_p="The chart shows all 49 open models by size. Larger models do better on average, but the spread is wide. Models of the same size can differ by 30 points.",
        f4_x="Parameters in billions, log scale", f4_y="Overall score",
        f4_line="Gemini 3.1 Pro, best commercial model",
        f4_note="For mixture-of-experts models such as Qwen3.5 397B-A17B or Kimi K2.6 we show the total parameter count. Only some of those parameters are used for each answer.",
        f5_h="A cheaper model can be enough",
        f5_brief="Gemini 3.1 Flash-Lite scored 66.7 and its whole evaluation cost $102. Gemini 3.1 Pro scored 72.6 for $3,834.",
        f5_p="We paid per call for commercial models. The costs are for the full sk-bench evaluation in the paper.",
        f5_cols=("Model", "Evaluation cost", "Overall score"),
        f5_few="includes the run with examples",
        f5_note="Flash-Lite, GPT-5.4 mini and Claude Sonnet 4.6 were also evaluated with three examples in the prompt, so their cost covers more runs. The other three were evaluated only without examples. The cost gap for the same work is even larger.",
        f6_h="More training on Slovak text can make a model worse",
        f6_brief="Qwen3 14B fell from 47.4 to 33.5 after further training on Slovak text. Tuning on a small set of Slovak instructions won back about three quarters of the lost points.",
        f6_p="Training an open model further on Slovak text seems an obvious way to improve its Slovak. In our experiment it was not that simple. After the extra training the model got worse at following tasks. Tuning it on a small set of Slovak instructions (Alpaca) brought the overall score back up to 43.8. Any such attempt needs careful testing.",
        f6_chart="The overall score fell after further training and partly recovered after instruction tuning",
        f6_rows=("Qwen3 14B, original", "after further training on Slovak text", "after Slovak instruction tuning"),
        f7_h="English prompts barely help",
        f7_brief="Using English instead of Slovak prompts raised the average score on comparable tasks by only 0.4 points out of 100. The reading score rose by 1.6 points, and the grammar score fell by 2.6.",
        f7_p="Each model also got the same tasks with English instructions. The texts, questions and answers stayed in Slovak. On grammar tasks, models scored even lower with English instructions. The table shows the average across all 55 models. Instruction following and Slohry are available only in Slovak, so they are left out.",
        f7_cols=("Task group", "Slovak prompt", "English prompt", "Difference"),
        f7_overall="Overall",
        lb_h2="Model leaderboard",
        lb_intro="The table shows all 55 models in the paper, as of its release. Scores are out of 100. Each task group score is the average of its tasks with Slovak prompts. Click a column name to sort the table. Switch to all 33 tasks to see the individual tests.",
        lb_view="View", lb_v_groups="Task groups", lb_v_tasks="All 33 tasks",
        lb_filter="Show models", lb_all="All", lb_open="Open", lb_com="Commercial",
        lb_search="Find a model", lb_leg_open="open model", lb_leg_com="commercial model",
        lb_model="Model", lb_overall="Overall",
        lb_groups=[("Knowledge", "Knowledge and reasoning"), ("Reading", "Reading comprehension and question answering"),
                   ("Classification", "Text classification"), ("Generation", "Text generation and summarisation"),
                   ("Instructions", "Instruction following"), ("Grammar", "Grammar and morphology")],
        lb_more="Show all ({n})", lb_less="Show fewer", lb_none="No model matches the search.",
        lb_note="The overall score is the average of the six task groups. Generation is scored by overlap with a reference summary (ROUGE-L), so those scores are lower. The best score in each column is highlighted. We keep adding new models to the <a href=\"" + LEADERBOARD + "\">leaderboard on Hugging Face</a>, so its scores can differ from the paper's.",
        lb_dl="Download the data", lb_csv="CSV", lb_json="JSON",
        t_h2="30 datasets in six groups",
        t_intro="For each task we list the data source, its size, the metric and the best result. The strip on the right shows the scores of all 55 models on a scale from 0 to 100. Some datasets are scored in two ways, so there are 33 tasks in total.",
        t_cols=("Task", "Data source", "Items", "Metric", "Best model", "Best open model", "All models"),
        t_new="new", t_new_long="new dataset, or one used to evaluate language models for the first time",
        m_h2="How we tested the models",
        m_intro="Every model gets the same prompts through a single lighteval pipeline, so open and commercial models are scored the same way. Wherever we could, we used data originally written in Slovak rather than translations.",
        m_row1="One task", m_row2="From task to leaderboard",
        m_steps1=[("Task", "for example 701 driving-test questions"), ("Three prompt phrasings", "in Slovak, with no examples"),
                  ("Model", "the same pipeline for every model"), ("Answer", "a letter, a word, a sentence or a full text"),
                  ("Metric", "accuracy, answer match, ROUGE-L or instruction checks")],
        m_steps2=[("Task score", "mean of the three phrasings"), ("Group score", "mean of its tasks"),
                  ("Overall score", "mean of the six groups")],
        fig_caption="The ten skill categories and 30 datasets in sk-bench. A star marks a new dataset. Each task's badge shows whether its data was originally written in Slovak, translated, generated or adapted. The figure is from the paper.",
        p_h2="For the press",
        p_intro="Figures you can quote, with their sources.",
        p_facts=[
            ("98.3%", "of driving-test questions answered correctly by Gemini 3.1 Pro (701 questions)"),
            ("96.3%", "of NBS financial exam questions (1,334 questions)"),
            ("94.8%", "of Slovak language maturita questions (369 questions)"),
            ("75%", "of sentences judged correctly by the best model when checking spelling and diacritics"),
            ("72.6 and 60.0", "points out of 100 overall for the best commercial and the best open model"),
            ("94% and 69%", "of questions about Slovakia answered correctly by Gemini 3.1 Pro and the best open model"),
            ("55", "models compared on 30 datasets across ten areas"),
        ],
        p_note_h="Note for editors",
        p_note="Percentages give the share of questions a model answered correctly. We did not simulate full exams. The overall scores and the results on questions about Slovakia come from the paper, which compares 55 models. New models keep being added to the leaderboard on Hugging Face, so its scores can differ.",
        p_fig_h="Figure to download", p_fig="Overview of the ten categories and 30 datasets (SVG)",
        p_links_h="Links",
        c_h2="Cite", copy="Copy", copied="Copied", authors_h3="Authors",
        links=[(PAPER, "Paper on arXiv"), (LEADERBOARD, "Leaderboard on Hugging Face"), (CODE, "Code on GitHub"), (HF_DATA, "Datasets on Hugging Face")],
        foot1="sk-bench was built by researchers from the Kempelen Institute of Intelligent Technologies (KInIT), Comenius University in Bratislava, the Technical University of Košice and other partners.",
        foot2="The questions on this page come from {d1} (CC BY 4.0), {d2} by TUKE KEMT (CC BY-SA 4.0) and {d3} (CC BY 4.0). The page sets no cookies and loads no external fonts. Last updated 3 November 2026.",
    ),
}

LETTERS = "abcde"
MIN = 50


def fmt(v: float, lang: str, pct: bool = True, nd: int = 1) -> str:
    s = f"{v:,.{nd}f}"
    if lang == "sk":
        s = s.replace(",", " ").replace(".", ",")
        return s + (" %" if pct else "")
    return s + ("%" if pct else "")


def money(v: float, lang: str) -> str:
    return fmt(v, lang, pct=False, nd=0) + " $" if lang == "sk" else "$" + fmt(v, lang, pct=False, nd=0)


def signed(v: float, lang: str) -> str:
    s = fmt(abs(v), lang, pct=False)
    return ("+" if v > 0 else "−" if v < 0 else "") + s


def pos(v: float) -> str:
    return f"{(v - MIN) / (100 - MIN) * 100:.2f}%"


# ── components ──────────────────────────────────────────────────────────────
def question(qd: dict, t: dict) -> str:
    en = t["lang"] == "en"
    opts = []
    for i, o in enumerate(qd["opts"]):
        tr = f'<small lang="en">{e(qd["opts_en"][i])}</small>' if en and qd["opts_en"] else ""
        opts.append(f'<li><button type="button" data-i="{i}"><b>{LETTERS[i]})</b>'
                    f'<span><span lang="sk">{e(o)}</span>{tr}</span></button></li>')
    q_tr = f'<p class="q-tr" lang="en">{e(qd["q_en"])}</p>' if en else ""
    m = qd["models"]
    stat = t["stat_q"].format(n=m[0], total=m[1]) if m else t["stat_todo"]
    todo = "" if m else ' data-todo="1"'
    return (f'<article class="q card" data-a="{qd["a"]}" data-stat="{e(stat)}"{todo}>'
            f'<p class="q-src">{e(t["src"][qd["src"]])}</p><p class="q-text" lang="sk">{e(qd["q"])}</p>{q_tr}'
            f'<ol class="opts">{"".join(opts)}</ol><p class="verdict" aria-live="polite"></p></article>')


def dot_rows(t: dict, ids: list[str]) -> str:
    lang = t["lang"]
    out = []
    for rid in ids:
        if rid == "maturita":
            gem, gpt, opn, open_name = MATURITA["gem"], MATURITA["gpt"], MATURITA["open"], MATURITA["open_name"]
            any_best = None
        else:
            gem = task_score(BY_KEY["google__gemini-3.1-pro-preview"], rid)
            gpt = task_score(BY_KEY["openai__gpt-5.5"], rid)
            bo = best_of(rid, True)
            opn, open_name = task_score(bo, rid), bo["name"]
            ba = best_of(rid)
            any_best = ba if task_score(ba, rid) > max(gem, gpt) + 0.05 else None
        name, n = t["rows"][rid]
        note = t["open_note"].format(name=open_name, v=fmt(opn, lang))
        if any_best:
            note += " " + t["best_any"].format(name=any_best["name"], v=fmt(task_score(any_best, rid), lang))
        top = max([gem, gpt] + ([task_score(any_best, rid)] if any_best else []))
        label = (f'{name}: Gemini 3.1 Pro {fmt(gem, lang)}, GPT-5.5 {fmt(gpt, lang)}, '
                 f'{t["legend_open"]} {open_name} {fmt(opn, lang)}')
        ticks = "".join(f'<i class="tick" style="left:{pos(x)}"></i>' for x in (60, 70, 80, 90))
        out.append(
            f'<div class="row"><div class="row-head"><h4>{e(name)}</h4>'
            f'<span class="rtop">{e(t["best"])} <b>{fmt(top, lang)}</b></span></div>'
            f'<div class="track" role="img" aria-label="{e(label)}">{ticks}'
            f'<i class="dot d-open" style="left:{pos(opn)}"></i>'
            f'<i class="dot d-gpt" style="left:{pos(gpt)}"></i>'
            f'<i class="dot d-gem" style="left:{pos(gem)}"></i></div>'
            f'<p class="note">{e(n)}. {e(note)}</p></div>')
    unit = " %" if lang == "sk" else "%"
    axis = "".join(f'<span style="left:{pos(x)}">{x}{unit}</span>' for x in (50, 60, 70, 80, 90, 100))
    legend = (f'<p class="legend" aria-hidden="true"><span><i class="dot d-gem"></i>Gemini 3.1 Pro</span>'
              f'<span><i class="dot d-gpt"></i>GPT-5.5</span><span><i class="dot d-open"></i>{e(t["legend_open"])}</span>'
              + '</p>')
    return (f'<div class="fig card">{legend}{"".join(out)}<div class="axis" aria-hidden="true">{axis}</div>'
            f'<p class="fig-note">{e(t["axis_note"])}</p></div>')


def hbars(groups: list[tuple[str, list[tuple[str, float, str]], bool]], lang: str, scale: float = 100) -> str:
    out = []
    for title, items, pct in groups:
        rows = "".join(
            f'<div class="bar"><span class="bar-label">{e(lbl)}</span>'
            f'<span class="bar-track"><i class="b-{kind}" style="width:{v / scale * 100:.1f}%"></i></span>'
            f'<span class="bar-val">{fmt(v, lang, pct)}</span></div>'
            for lbl, v, kind in items)
        out.append(f'<div class="bars">{f"<p class=bars-title>{e(title)}</p>" if title else ""}{rows}</div>')
    return "".join(out)


def open_gap(t: dict) -> str:
    lang = t["lang"]
    com = max((m for m in MODELS if not m["open"]), key=lambda m: m["overall"])
    opn = max((m for m in MODELS if m["open"]), key=lambda m: m["overall"])
    tc, to = best_of("slovak_trivia", False), best_of("slovak_trivia", True)
    return hbars([
        (t["bars1"], [(f'{t["bar_com"]}: {com["name"]}', com["overall"], "com"), (f'{t["bar_open"]}: {opn["name"]}', opn["overall"], "open")], False),
        (t["bars2"], [(f'{t["bar_com"]}: {tc["name"]}', task_score(tc, "slovak_trivia"), "com"),
                      (f'{t["bar_open"]}: {to["name"]}', task_score(to, "slovak_trivia"), "open")], True),
    ], lang)


def scatter(t: dict) -> str:
    lang = t["lang"]
    W, H, L, R, TOP, B = 640, 360, 48, 18, 18, 50
    lo, hi = math.log10(0.25), math.log10(1500)
    X = lambda p: L + (math.log10(p) - lo) / (hi - lo) * (W - L - R)  # noqa: E731
    Y = lambda v: TOP + (1 - v / 80) * (H - TOP - B)  # noqa: E731
    parts = []
    for v in (0, 20, 40, 60, 80):
        parts.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                     f'<text class="ax" x="{L - 8}" y="{Y(v) + 4:.1f}" text-anchor="end">{v}</text>')
    for p, lbl in ((1, "1"), (10, "10"), (100, "100"), (1000, "1 000" if lang == "sk" else "1,000")):
        parts.append(f'<line class="grid" x1="{X(p):.1f}" x2="{X(p):.1f}" y1="{TOP}" y2="{H - B}"/>'
                     f'<text class="ax" x="{X(p):.1f}" y="{H - B + 18}" text-anchor="middle">{lbl}</text>')
    parts.append(f'<text class="ax-t" x="{(L + W - R) / 2}" y="{H - 8}" text-anchor="middle">{e(t["f4_x"])}</text>')
    parts.append(f'<text class="ax-t" transform="translate(13 {(TOP + H - B) / 2}) rotate(-90)" text-anchor="middle">{e(t["f4_y"])}</text>')
    gem = BY_KEY["google__gemini-3.1-pro-preview"]["overall"]
    parts.append(f'<line class="ref" x1="{L}" x2="{W - R}" y1="{Y(gem):.1f}" y2="{Y(gem):.1f}"/>'
                 f'<text class="ref-t" x="{L + 6}" y="{Y(gem) - 6:.1f}">{e(t["f4_line"])}, {fmt(gem, lang, pct=False)}</text>')
    marks = {"mistralai__Mistral-Large-3-675B-Instruct-2512": ("end", -10, 4), "Qwen__Qwen3.5-9B": ("end", -10, -8),
             "openai__gpt-oss-120b": ("end", -10, -6)}
    labels = []
    for m in sorted((m for m in MODELS if m["open"]), key=lambda m: m["key"] in marks):
        x, y = X(m["params"]), Y(m["overall"])
        hl = m["key"] in marks
        parts.append(f'<circle class="pt{" hl" if hl else ""}" cx="{x:.1f}" cy="{y:.1f}" r="{6 if hl else 4.5}">'
                     f'<title>{e(m["name"])}: {fmt(m["overall"], lang, pct=False)}</title></circle>')
        if hl:
            anchor, dx, dy = marks[m["key"]]
            labels.append(f'<text class="lbl" x="{x + dx:.1f}" y="{y + dy:.1f}" text-anchor="{anchor}">'
                          f'{e(m["name"])}, {fmt(m["overall"], lang, pct=False)}</text>')
    svg = (f'<svg class="scatter" viewBox="0 0 {W} {H}" role="img" aria-label="{e(t["f4_brief"])}">'
           f'{"".join(parts)}{"".join(labels)}</svg>')
    return f'<figure class="fig card"><div class="fig-scroll">{svg}</div><figcaption class="fig-note">{e(t["f4_note"])}</figcaption></figure>'


def cost_table(t: dict) -> str:
    lang = t["lang"]
    top = max(c for _, c, _ in COSTS)
    rows = []
    for key, cost, few in COSTS:
        m = BY_KEY[key]
        rows.append(f'<tr><th scope="row">{e(m["name"])}{f"<small>{e(t["f5_few"])}</small>" if few else ""}</th>'
                    f'<td><span class="cbar"><i style="width:{cost / top * 100:.1f}%"></i></span>{money(cost, lang)}</td>'
                    f'<td class="num"><b>{fmt(m["overall"], lang, pct=False)}</b></td></tr>')
    c = t["f5_cols"]
    return (f'<figure class="fig card"><div class="fig-scroll"><table class="plain"><thead><tr><th scope="col">{e(c[0])}</th>'
            f'<th scope="col">{e(c[1])}</th><th scope="col" class="num">{e(c[2])}</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div><figcaption class="fig-note">{e(t["f5_note"])}</figcaption></figure>')


def finetune(t: dict) -> str:
    keys = ["Qwen__Qwen3-14B", "slovak-nlp__Qwen3-14B-sk", "mrshu__qwen3-14b-sk-alpaca-slovak-it"]
    items = [(lbl, BY_KEY[k]["overall"], "open" if i == 0 else "muted" if i == 1 else "open")
             for i, (lbl, k) in enumerate(zip(t["f6_rows"], keys))]
    return f'<div class="fig card">{hbars([(t["f6_chart"], items, False)], t["lang"], scale=60)}</div>'


def lang_table(t: dict) -> str:
    lang = t["lang"]
    names = dict(zip(BOARD["groups"], (full for _, full in t["lb_groups"])))
    rows = []
    for r in BOARD["lang"]:
        overall = r["group"] == "__overall__"
        name = t["f7_overall"] if overall else names[r["group"]]
        cls = "pos" if r["delta"] > 0 else "neg" if r["delta"] < 0 else ""
        rows.append(f'<tr{" class=total" if overall else ""}><th scope="row">{e(name)}</th>'
                    f'<td class="num">{fmt(r["sk"], lang, pct=False)}</td><td class="num">{fmt(r["en"], lang, pct=False)}</td>'
                    f'<td class="num {cls}">{signed(r["delta"], lang)}</td></tr>')
    c = t["f7_cols"]
    return (f'<figure class="fig card"><div class="fig-scroll"><table class="plain"><thead><tr><th scope="col">{e(c[0])}</th>'
            + "".join(f'<th scope="col" class="num">{e(x)}</th>' for x in c[1:])
            + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div></figure>')


def finding(n: int, fid: str, t: dict, body: str, fig: str) -> str:
    return (f'<article class="finding" id="{fid}" aria-labelledby="{fid}-h">'
            f'<p class="fnum"><span class="badge">{n}</span></p>'
            f'<h3 id="{fid}-h">{e(t[f"f{n}_h"])}</h3>'
            f'<div class="f-grid"><div class="brief"><p class="brief-h">{e(t["brief"])}</p><p>{e(t[f"f{n}_brief"])}</p></div>'
            f'<div class="f-body">{body}</div></div>{fig}</article>')


def leaderboard(t: dict) -> str:
    lang = t["lang"]
    cols = [[m["overall"]] + m["groups"] + m["tasks"] for m in MODELS]
    best = [max(c[j] for c in cols) for j in range(len(cols[0]))]
    ng = len(BOARD["groups"])
    group_start = {}
    for i, tk in enumerate(BOARD["tasks"]):
        group_start.setdefault(tk["group"], i)
    starts = set(group_start.values())
    num = lambda v: fmt(v, lang, pct=False)  # noqa: E731

    def cell(v: float, j: int, cls: str) -> str:
        return f'<td class="{cls}{" best" if v == best[j] else ""}">{num(v)}</td>'

    rows = []
    for rank, (m, c) in enumerate(zip(MODELS, cols), 1):
        kind = "open" if m["open"] else "com"
        g = "".join(cell(v, j, "c-g") for j, v in enumerate(c[1:1 + ng], 1))
        tk = "".join(cell(v, j, "c-t" + (" gs" if (j - 1 - ng) in starts else "")) for j, v in enumerate(c[1 + ng:], 1 + ng))
        rows.append(
            f'<tr data-open="{int(m["open"])}" data-v="{" ".join(map(str, c))}" data-name="{e(m["name"])}">'
            f'<th scope="row" class="c-name"><span class="rank">{rank}</span><i class="sw sw-{kind}" aria-hidden="true"></i>{e(m["name"])}'
            f'<span class="sr">, {e(t["lb_leg_open"] if m["open"] else t["lb_leg_com"])}</span></th>'
            f'<td class="c-overall{" best" if c[0] == best[0] else ""}"><span class="ov"><span class="ov-bar" aria-hidden="true"><i style="width:{c[0]}%"></i></span>{num(c[0])}</span></td>'
            f'{g}{tk}</tr>')
    btn = lambda col, label, title="": (  # noqa: E731
        f'<button type="button" data-col="{col}"{f" title=\"{e(title)}\"" if title else ""}>{e(label)}</button>')
    tname = (lambda k: TASKS[k][0]) if lang == "sk" else (lambda k: TASKS[k][1])
    head = (f'<th scope="col" class="c-name">{btn("name", t["lb_model"])}</th>'
            f'<th scope="col" class="c-overall" aria-sort="descending">{btn(0, t["lb_overall"])}</th>'
            + "".join(f'<th scope="col" class="c-g">{btn(j, short, full)}</th>' for j, (short, full) in enumerate(t["lb_groups"], 1))
            + "".join(f'<th scope="col" class="c-t{" gs" if i in starts else ""}">{btn(1 + ng + i, tk["label"], tname(tk["key"]))}</th>'
                      for i, tk in enumerate(BOARD["tasks"])))
    seg = lambda name, items, cur: "".join(  # noqa: E731
        f'<button type="button" data-{name}="{v}" aria-pressed="{"true" if v == cur else "false"}">{e(lbl)}</button>' for v, lbl in items)
    return f"""<div class="lb card" data-show="{SHOW}" data-more="{e(t["lb_more"])}" data-less="{e(t["lb_less"])}" data-view="groups">
      <div class="lb-tools">
        <div class="seg" role="group" aria-label="{e(t["lb_view"])}">{seg("view", (("groups", t["lb_v_groups"]), ("tasks", t["lb_v_tasks"])), "groups")}</div>
        <div class="seg" role="group" aria-label="{e(t["lb_filter"])}">{seg("f", (("all", t["lb_all"]), ("open", t["lb_open"]), ("com", t["lb_com"])), "all")}</div>
        <label class="search"><span class="sr">{e(t["lb_search"])}</span><input type="search" placeholder="{e(t["lb_search"])}" autocomplete="off"></label>
      </div>
      <p class="legend" aria-hidden="true"><span><i class="sw sw-open"></i>{e(t["lb_leg_open"])}</span><span><i class="sw sw-com"></i>{e(t["lb_leg_com"])}</span></p>
      <div class="lb-scroll"><table class="lb-table">
        <caption class="sr">{e(t["lb_h2"])}</caption>
        <thead><tr>{head}</tr></thead>
        <tbody>{"".join(rows)}</tbody>
      </table></div>
      <p class="lb-empty" hidden>{e(t["lb_none"])}</p>
      <div class="lb-foot"><button class="btn small lb-more" type="button" hidden></button>
        <p class="dl">{e(t["lb_dl"])}: <a href="{t["root"]}data/sk-bench-leaderboard.csv" download>{e(t["lb_csv"])}</a> <a href="{t["root"]}data/leaderboard.json" download>{e(t["lb_json"])}</a></p></div>
      <p class="fig-note">{t["lb_note"]}</p>
    </div>"""


def tasks_section(t: dict) -> str:
    lang = t["lang"]
    li = 0 if lang == "sk" else 1
    names = dict(zip(BOARD["groups"], t["lb_groups"]))
    c = t["t_cols"]
    out = []
    for grp in BOARD["groups"]:
        keys = [tk["key"] for tk in BOARD["tasks"] if tk["group"] == grp]
        metric_of = {tk["key"]: tk["metric"] for tk in BOARD["tasks"]}
        rows = []
        for k in keys:
            sk_name, en_name, prov, new, n, d_sk, d_en = TASKS[k]
            ba, bo = best_of(k), best_of(k, True)
            strip = "".join(f'<i class="{"o" if m["open"] else "c"}" style="left:{task_score(m, k):.1f}%"></i>' for m in MODELS)
            newb = f' <abbr class="new" title="{e(t["t_new_long"])}">{e(t["t_new"])}</abbr>' if new else ""
            rows.append(
                f'<tr><th scope="row"><span class="t-name">{e(sk_name if li == 0 else en_name)}</span>{newb}'
                f'<small>{e(d_sk if li == 0 else d_en)}</small></th>'
                f'<td><span class="prov prov-{prov}">{e(PROV[prov][li])}</span></td>'
                f'<td class="num">{fmt(n, lang, pct=False, nd=0)}</td>'
                f'<td class="t-metric">{e(METRIC[metric_of[k]][li])}</td>'
                f'<td><b>{fmt(task_score(ba, k), lang, pct=False)}</b><small>{e(ba["name"])}</small></td>'
                f'<td><b>{fmt(task_score(bo, k), lang, pct=False)}</b><small>{e(bo["name"])}</small></td>'
                f'<td class="t-strip"><span class="strip" aria-hidden="true">{strip}</span></td></tr>')
        out.append(
            f'<div class="tgroup card"><h3>{e(names[grp][1])} <span class="count">{len(keys)}</span></h3>'
            f'<div class="t-scroll"><table class="tasks"><thead><tr>'
            + "".join(f'<th scope="col"{" class=num" if i == 2 else ""}{" class=t-metric" if i == 3 else ""}{" class=t-strip" if i == 6 else ""}>{e(x)}</th>'
                      for i, x in enumerate(c))
            + f'</tr></thead><tbody>{"".join(rows)}</tbody></table></div></div>')
    legend = (f'<p class="legend"><span><span class="prov prov-N">{e(PROV["N"][li])}</span></span>'
              f'<span><span class="prov prov-MT">{e(PROV["MT"][li])}</span></span>'
              f'<span><span class="prov prov-MTA">{e(PROV["MTA"][li])}</span></span>'
              f'<span><span class="prov prov-G">{e(PROV["G"][li])}</span></span>'
              f'<span><abbr class="new">{e(t["t_new"])}</abbr> {e(t["t_new_long"])}</span>'
              f'<span><i class="sw sw-open"></i>{e(t["lb_leg_open"])}</span><span><i class="sw sw-com"></i>{e(t["lb_leg_com"])}</span></p>')
    return legend + "".join(out)


def flow(t: dict) -> str:
    def row(title: str, steps: list[tuple[str, str]]) -> str:
        items = "".join(f'<li><b>{e(a)}</b><span>{e(b)}</span></li>' for a, b in steps)
        return f'<div class="flow-row"><p class="flow-h">{e(title)}</p><ol class="steps">{items}</ol></div>'
    return f'<div class="flow card">{row(t["m_row1"], t["m_steps1"])}{row(t["m_row2"], t["m_steps2"])}</div>'


def to_csv() -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["rank", "model", "open", "params_b", "overall"] + BOARD["groups"]
               + [f"{k} (SK)" for k in TASK_KEYS] + [f"{k} (EN)" for k in TASK_KEYS])
    for i, m in enumerate(MODELS, 1):
        w.writerow([i, m["name"], int(m["open"]), m["params"] if m["params"] is not None else "", m["overall"]]
                   + m["groups"] + m["tasks"] + ["" if v is None else v for v in m["tasks_en"]])
    return buf.getvalue()


CSS = r"""
:root {
  --bg: #FAF9F6; --band: #F2F1EC; --card: #FFFFFF; --ink: #1B1E24; --muted: #5A606B; --rule: #E2E0D9;
  --accent: #1F3F68; --accent-ink: #FFFFFF; --accent-soft: #E7EDF5; --open: #2D6CB5; --com: #3B4049;
  --red: #B2322A; --green: #2F7A4B; --hl: #FFF3C4;
  --serif: Charter, "Bitstream Charter", "Sitka Text", Cambria, "Iowan Old Style", Georgia, serif;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  --shadow: 0 1px 2px rgba(20,24,32,.05), 0 8px 24px -14px rgba(20,24,32,.25);
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #121418; --band: #171A1F; --card: #1C2026; --ink: #E7E9ED; --muted: #9BA2AE; --rule: #2D323A;
    --accent: #8DB2E6; --accent-ink: #10151C; --accent-soft: #1F2B3B; --open: #6EA4E8; --com: #C7CCD4;
    --red: #F08A80; --green: #6CC28E; --hl: #3A3520; --shadow: none; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #121418; --band: #171A1F; --card: #1C2026; --ink: #E7E9ED; --muted: #9BA2AE; --rule: #2D323A;
  --accent: #8DB2E6; --accent-ink: #10151C; --accent-soft: #1F2B3B; --open: #6EA4E8; --com: #C7CCD4;
  --red: #F08A80; --green: #6CC28E; --hl: #3A3520; --shadow: none; color-scheme: dark;
}
* { box-sizing: border-box; }
html { -webkit-text-size-adjust: 100%; scroll-behavior: smooth; scroll-padding-top: 4.5rem; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 400 1.0625rem/1.62 var(--serif); -webkit-font-smoothing: antialiased; }
a { color: var(--accent); text-underline-offset: .18em; text-decoration-thickness: 1px; }
a:hover { text-decoration-thickness: 2px; }
:focus-visible { outline: 3px solid var(--accent); outline-offset: 2px; border-radius: 3px; }
.wrap { max-width: 72rem; margin: 0 auto; padding: 0 1.25rem; }
p { margin: .8rem 0; max-width: 70ch; }
h1, h2, h3, h4 { font-family: var(--serif); margin: 0; line-height: 1.15; letter-spacing: -.01em; font-weight: 700; }
h2 { font-size: clamp(1.75rem, 3.6vw, 2.35rem); }
h3 { font-size: 1.45rem; }
h4 { font-size: 1.05rem; }
.sr { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
.card { background: var(--card); border: 1px solid var(--rule); border-radius: 10px; box-shadow: var(--shadow); }
.num { text-align: right; font-variant-numeric: tabular-nums; }
.ui, .btn, .seg button, .lb-table, table.plain, table.tasks, .legend, .fig-note, nav, .badge, .prov, .new, .q-src { font-family: var(--sans); }

/* Top bar */
.top { position: sticky; top: 0; z-index: 20; background: color-mix(in srgb, var(--bg) 88%, transparent); backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); border-bottom: 1px solid var(--rule); }
.top .wrap { display: flex; align-items: center; gap: 1rem; height: 3.5rem; }
.brand { display: flex; align-items: center; gap: .75rem; color: var(--ink); text-decoration: none; min-width: 0; }
.brand b { font: 700 1rem var(--sans); letter-spacing: -.01em; white-space: nowrap; }
.brand span { font-size: .95rem; color: var(--muted); border-left: 1px solid var(--rule); padding-left: .75rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.top nav { margin-left: auto; display: flex; align-items: center; gap: .15rem; font-size: .88rem; }
.top nav a { color: var(--muted); text-decoration: none; padding: .3rem .55rem; border-radius: 6px; white-space: nowrap; }
.top nav a:hover { color: var(--ink); background: var(--band); }
.top nav a.on { color: var(--accent); background: var(--accent-soft); }
.lang-sw { font: 600 .85rem var(--sans); color: var(--ink); text-decoration: none; border: 1px solid var(--rule); border-radius: 999px; padding: .2rem .7rem; white-space: nowrap; }
.theme { flex: none; width: 2rem; height: 2rem; border-radius: 50%; border: 1px solid var(--rule); background: var(--card); color: var(--ink); cursor: pointer; display: grid; place-items: center; padding: 0; }
.theme svg { width: 1rem; height: 1rem; }
@media (max-width: 68rem) { .brand span { display: none; } }
@media (max-width: 56rem) { .top nav .nl { display: none; } }

/* Hero */
.hero { text-align: center; padding-top: 3.5rem; padding-bottom: 3rem; }
.hero h1 { font-size: clamp(2.3rem, 5.4vw, 3.6rem); line-height: 1.08; max-width: 18ch; margin: 0 auto; letter-spacing: -.02em; }
.subtitle { font-style: italic; color: var(--muted); font-size: 1.15rem; margin: 1rem auto 0; max-width: 46ch; }
.authors { list-style: none; padding: 0; margin: 1.75rem auto 0; max-width: 50rem; display: flex; flex-wrap: wrap; justify-content: center; gap: .2rem 1.1rem; font-size: 1rem; }
.authors sup, .affil sup { font: 500 .68em var(--sans); color: var(--muted); margin-left: .1em; }
.affil { font: .86rem/1.6 var(--sans); color: var(--muted); max-width: 52rem; margin: .8rem auto 0; display: flex; flex-wrap: wrap; justify-content: center; gap: 0 1.1rem; }
.corr { font: .84rem var(--sans); color: var(--muted); margin: .5rem auto 0; }
.actions { display: flex; flex-wrap: wrap; justify-content: center; gap: .55rem; margin-top: 1.6rem; }
.btn { display: inline-flex; align-items: center; gap: .45rem; padding: .55rem 1rem; border-radius: 999px; font-weight: 600; font-size: .93rem; text-decoration: none; border: 1px solid var(--rule); background: var(--card); color: var(--ink); cursor: pointer; }
.btn:hover { border-color: var(--ink); }
.btn.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.btn.primary:hover { filter: brightness(1.1); }
.btn.small { padding: .35rem .85rem; font-size: .86rem; }
.btn svg { width: 1em; height: 1em; }
.venue { font: .9rem var(--sans); color: var(--muted); margin: 1.1rem auto 0; }
.venue a { color: inherit; text-decoration-thickness: 1px; text-underline-offset: 2px; }

/* Sections */
section.block { padding: 4rem 0; border-top: 1px solid var(--rule); }
section.band { background: var(--band); }
.intro { color: var(--muted); font-size: 1.1rem; margin-top: .9rem; }
.sec-head { max-width: 50rem; margin-bottom: 2rem; }

/* Overview cards */
.cards { display: grid; gap: 1rem; }
@media (min-width: 52rem) { .cards { grid-template-columns: repeat(3, 1fr); } }
.ocard { padding: 1.4rem 1.5rem 1.2rem; display: flex; flex-direction: column; }
.ocard .big { font: 700 2.8rem/1 var(--serif); color: var(--accent); letter-spacing: -.02em; margin: 0; }
.ocard h3 { font-size: 1.12rem; margin-top: .5rem; }
.ocard p { font-size: .98rem; color: var(--muted); margin: .6rem 0 1rem; }
.ocard a { margin-top: auto; font: 600 .88rem var(--sans); border-top: 1px solid var(--rule); padding-top: .8rem; text-decoration: none; }
.ocard a:hover { text-decoration: underline; }
.stats { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 2rem 0 0; padding: 0; }
@media (min-width: 40rem) { .stats { grid-template-columns: repeat(4, 1fr); } }
.stats div { border-left: 2px solid var(--accent); padding: .1rem 0 .1rem .9rem; }
.stats dt { font: 700 1.7rem/1.1 var(--serif); }
.stats dd { margin: 0; font: .9rem var(--sans); color: var(--muted); }

/* Quiz */
.quiz { display: grid; gap: 1rem; align-items: start; }
@media (min-width: 52rem) { .quiz { grid-template-columns: 1fr 1fr; } }
@media (min-width: 72rem) { .quiz { grid-template-columns: 1fr 1fr 1fr; } }
.q { padding: 1.2rem 1.3rem 1.2rem; }
.q-src { font-size: .82rem; color: var(--muted); margin: 0; font-weight: 600; }
.q-text { font-size: 1.15rem; line-height: 1.4; font-weight: 700; margin: .45rem 0 0; }
.q-tr { font-style: italic; color: var(--muted); margin: .25rem 0 0; font-size: .96rem; }
.opts { list-style: none; padding: 0; margin: .9rem 0 0; display: grid; gap: .4rem; }
.opts button { position: relative; width: 100%; text-align: left; font: 400 1rem/1.4 var(--serif); color: var(--ink); cursor: pointer; background: var(--bg); border: 1px solid var(--rule); border-radius: 7px; padding: .55rem .75rem; display: flex; gap: .6rem; align-items: baseline; }
.opts button:hover { border-color: var(--ink); }
.opts button b { font: 650 .9rem var(--sans); min-width: 1.1rem; }
.opts small { display: block; font-style: italic; color: var(--muted); font-size: .88rem; margin-top: .15rem; }
.q[data-done] button { cursor: default; }
.q[data-done] button:not(.is-right):not(.is-wrong):hover { border-color: var(--rule); }
.opts button.is-right { border-color: var(--green); background: color-mix(in srgb, var(--green) 10%, var(--card)); }
.opts button.is-right b { color: var(--green); }
.opts button.is-wrong { border-color: var(--red); }
.opts button.is-wrong > span { text-decoration: line-through; text-decoration-color: var(--red); color: var(--muted); }
.verdict { margin: .8rem 0 0; font: .9rem/1.5 var(--sans); color: var(--muted); }
.verdict:empty { display: none; }
.quiz-more { text-align: center; max-width: none; margin-top: 1.2rem; }
.js-quiz .q:nth-child(n+4) { display: none; }
.verdict strong { color: var(--ink); margin-right: .35rem; }
.verdict .todo { display: block; margin-top: .4rem; padding: .3rem .55rem; border: 1px dashed var(--muted); border-radius: 4px; font-style: italic; }

/* Findings */
.findings { display: grid; gap: 3.5rem; }
.findings > *, .f-grid > *, .cards > *, .quiz > *, .press > *, .cite-grid > *, .split > * { min-width: 0; }
.finding { scroll-margin-top: 4.5rem; }
.fnum { margin: 0 0 .6rem; }
.badge { display: inline-grid; place-items: center; min-width: 1.9rem; height: 1.9rem; padding: 0 .5rem; border-radius: 6px; background: var(--accent); color: var(--accent-ink); font-weight: 700; font-size: .95rem; }
.finding h3 { font-size: clamp(1.4rem, 2.6vw, 1.8rem); }
.f-grid { display: grid; gap: 1rem 2rem; margin-top: 1rem; }
@media (min-width: 60rem) { .f-grid { grid-template-columns: 1fr 1.5fr; align-items: start; } }
.brief { background: var(--accent-soft); border-left: 3px solid var(--accent); border-radius: 0 8px 8px 0; padding: .9rem 1.1rem; }
.brief-h { font: 700 .85rem var(--sans); color: var(--accent); margin: 0; }
.brief p:last-child { margin: .3rem 0 0; font-size: 1.02rem; }
.f-body p:first-child { margin-top: 0; }
.fig { margin: 1.4rem 0 0; padding: 1.3rem 1.4rem 1.1rem; }
.fig-note { font-size: .84rem; color: var(--muted); margin: .9rem 0 0; max-width: 90ch; }
.fig-scroll { overflow-x: auto; }

/* Dot chart */
.legend { display: flex; flex-wrap: wrap; gap: .45rem 1.3rem; font-size: .86rem; color: var(--muted); margin: 0 0 .4rem; }
.legend > span { display: inline-flex; align-items: center; gap: .45rem; }
.legend .dot { position: static; transform: none; }
.row { padding: 1rem 0 .9rem; border-top: 1px solid var(--rule); }
.legend + .row { border-top: 0; }
.row-head { display: flex; justify-content: space-between; align-items: baseline; gap: 1rem; }
.rtop { font: .88rem var(--sans); color: var(--muted); white-space: nowrap; }
.rtop b { color: var(--ink); font-size: 1rem; }
.track { position: relative; height: 2.4rem; margin: .9rem 0 .2rem; }
.track::before { content: ""; position: absolute; left: 0; right: 0; top: 50%; border-top: 1px solid var(--rule); }
.tick { position: absolute; top: 50%; height: .55rem; border-left: 1px solid var(--rule); transform: translateY(-50%); }
.dot { position: absolute; top: 50%; width: .9rem; height: .9rem; transform: translate(-50%, -50%); display: inline-block; }
.d-gem { background: var(--com); border-radius: 50%; }
.d-gpt { border: 2.5px solid var(--com); border-radius: 50%; z-index: 1; width: 1.2rem; height: 1.2rem; }
.legend .d-gpt { width: .9rem; height: .9rem; }
.d-open { background: var(--open); border-radius: 2px; }
.note { font: .86rem var(--sans); color: var(--muted); margin: 0; max-width: none; }
.axis { position: relative; height: 1.3rem; font: .78rem var(--sans); color: var(--muted); font-variant-numeric: tabular-nums; margin-top: .3rem; }
.axis span { position: absolute; transform: translateX(-50%); white-space: nowrap; }
.axis span:first-child { transform: none; }
.axis span:last-child { transform: translateX(-100%); }

/* Bars */
.bars { margin-bottom: 1.4rem; font-family: var(--sans); }
.bars:last-child { margin-bottom: 0; }
.bars-title { font-weight: 650; margin: 0 0 .7rem; font-size: .95rem; }
.bar { display: grid; grid-template-columns: 1fr auto; gap: .2rem .8rem; margin-bottom: .7rem; }
.bar-label { font-size: .88rem; color: var(--muted); }
.bar-val { grid-row: 1 / 3; grid-column: 2; align-self: end; font-weight: 700; font-size: 1.15rem; min-width: 4rem; text-align: right; font-variant-numeric: tabular-nums; }
.bar-track { height: .75rem; background: var(--band); border-radius: 3px; overflow: hidden; }
.bar-track i { display: block; height: 100%; }
.b-com { background: var(--com); }
.b-open { background: var(--open); }
.b-muted { background: color-mix(in srgb, var(--open) 40%, var(--band)); }
.split { display: grid; gap: 1rem 2rem; }
@media (min-width: 60rem) { .split { grid-template-columns: 1fr 1fr; } }

/* Scatter */
.scatter { display: block; width: 100%; min-width: 520px; height: auto; font-family: var(--sans); }
.scatter .grid { stroke: var(--rule); stroke-width: 1; }
.scatter .ax { fill: var(--muted); font-size: 12px; }
.scatter .ax-t { fill: var(--muted); font-size: 12.5px; }
.scatter .pt { fill: var(--open); fill-opacity: .55; stroke: var(--card); stroke-width: 1; }
.scatter .pt.hl { fill-opacity: 1; stroke: var(--ink); stroke-width: 1.5; }
.scatter .ref { stroke: var(--com); stroke-dasharray: 5 4; stroke-width: 1.5; }
.scatter .ref-t { fill: var(--muted); font-size: 12px; }
.scatter .lbl { fill: var(--ink); font-size: 12.5px; font-weight: 600; paint-order: stroke; stroke: var(--card); stroke-width: 4px; stroke-linejoin: round; }

/* Plain tables */
table.plain { width: 100%; border-collapse: collapse; font-size: .93rem; }
table.plain th, table.plain td { padding: .55rem .6rem; border-top: 1px solid var(--rule); text-align: left; vertical-align: top; }
table.plain thead th { border-top: 0; border-bottom: 1.5px solid var(--ink); font-size: .82rem; color: var(--muted); font-weight: 600; vertical-align: bottom; }
table.plain th.num, table.plain td.num { text-align: right; }
table.plain tbody th { font-weight: 600; }
table.plain small { display: block; font-weight: 400; color: var(--muted); font-size: .8rem; }
table.plain tr.total th, table.plain tr.total td { border-top: 1.5px solid var(--ink); font-weight: 700; }
.pos { color: var(--green); }
.neg { color: var(--red); }
.cbar { display: inline-block; vertical-align: middle; width: 7rem; height: .55rem; background: var(--band); border-radius: 3px; overflow: hidden; margin-right: .6rem; }
.cbar i { display: block; height: 100%; background: var(--com); }

/* Leaderboard */
.lb { padding: 1.2rem 1.3rem; }
.lb-tools { display: flex; flex-wrap: wrap; align-items: center; gap: .6rem; margin-bottom: .8rem; }
.seg { display: inline-flex; border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; background: var(--bg); }
.seg button { font: 600 .86rem var(--sans); color: var(--muted); background: transparent; border: 0; padding: .4rem .8rem; cursor: pointer; }
.seg button + button { border-left: 1px solid var(--rule); }
.seg button:hover { color: var(--ink); }
.seg button[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); }
.search { margin-left: auto; }
.search input { font: .9rem var(--sans); color: var(--ink); background: var(--bg); border: 1px solid var(--rule); border-radius: 8px; padding: .42rem .75rem; width: 13rem; max-width: 100%; }
.sw { display: inline-block; width: .7rem; height: .7rem; border-radius: 2px; flex: none; }
.sw-open { background: var(--open); }
.sw-com { background: var(--com); border-radius: 50%; }
.lb-scroll { overflow-x: auto; margin: 0 -.3rem; }
.lb-table { width: 100%; border-collapse: separate; border-spacing: 0; font-size: .9rem; }
.lb-table th, .lb-table td { padding: .45rem .55rem; text-align: right; border-top: 1px solid var(--rule); white-space: nowrap; font-weight: 400; font-variant-numeric: tabular-nums; }
.lb-table thead th { border-top: 0; border-bottom: 1.5px solid var(--ink); font-size: .8rem; font-weight: 600; color: var(--muted); vertical-align: bottom; }
.lb-table thead button { font: inherit; color: inherit; background: none; border: 0; padding: .15rem 0; cursor: pointer; white-space: nowrap; }
.lb-table thead button:hover, .lb-table th[aria-sort] button { color: var(--ink); }
.lb-table th[aria-sort="descending"] button::after { content: " \25BE"; }
.lb-table th[aria-sort="ascending"] button::after { content: " \25B4"; }
.lb-table .c-name { text-align: left; font-weight: 600; position: sticky; left: 0; background: var(--card); z-index: 1; min-width: 12rem; }
.c-name .rank { display: inline-block; width: 1.7rem; color: var(--muted); font-weight: 400; font-size: .82rem; }
.c-name .sw { margin-right: .5rem; vertical-align: baseline; }
.lb-table .c-overall { min-width: 9.5rem; font-weight: 700; }
.ov { display: flex; align-items: center; gap: .6rem; justify-content: flex-end; }
.ov-bar { flex: 1; height: .5rem; background: var(--band); border-radius: 3px; overflow: hidden; min-width: 3rem; }
.ov-bar i { display: block; height: 100%; background: var(--com); }
tr[data-open="1"] .ov-bar i { background: var(--open); }
.lb-table td.best { background: var(--hl); font-weight: 700; }
.lb-table .gs { border-left: 1px solid var(--rule); }
.lb[data-view="groups"] .c-t { display: none; }
.lb[data-view="tasks"] .c-g { display: none; }
.lb[data-view="tasks"] .ov-bar { display: none; }
.lb[data-view="tasks"] .c-overall { min-width: 0; }
.lb-table tbody tr:hover td, .lb-table tbody tr:hover th { background: color-mix(in srgb, var(--accent-soft) 70%, var(--card)); }
.lb-table tbody tr:hover td.best { background: var(--hl); }
.lb-empty { font: .9rem var(--sans); color: var(--muted); }
.lb-foot { display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: .6rem 1rem; margin-top: 1rem; }
.dl { font: .88rem var(--sans); color: var(--muted); margin: 0 0 0 auto; }
.dl a { margin-left: .5rem; font-weight: 600; }

/* Tasks */
.tgroup { padding: 1.2rem 1.3rem; margin-top: 1rem; }
.tgroup h3 { font-size: 1.2rem; }
.count { font: 600 .8rem var(--sans); color: var(--muted); background: var(--band); border-radius: 999px; padding: .1rem .5rem; vertical-align: middle; }
.t-scroll { overflow-x: auto; }
table.tasks { width: 100%; border-collapse: collapse; font-size: .9rem; margin-top: .6rem; }
table.tasks th, table.tasks td { padding: .6rem .55rem; border-top: 1px solid var(--rule); text-align: left; vertical-align: top; }
table.tasks thead th { border-top: 0; border-bottom: 1.5px solid var(--ink); font-size: .8rem; color: var(--muted); font-weight: 600; white-space: nowrap; }
table.tasks thead th.num { text-align: right; }
table.tasks tbody th { font-weight: 600; min-width: 13rem; }
table.tasks small { display: block; color: var(--muted); font-weight: 400; font-size: .8rem; line-height: 1.4; margin-top: .1rem; }
table.tasks td b { font-variant-numeric: tabular-nums; }
.prov { display: inline-block; font-size: .76rem; font-weight: 600; border-radius: 999px; padding: .1rem .55rem; white-space: nowrap; border: 1px solid transparent; }
.prov-N { background: var(--accent-soft); color: var(--accent); }
.prov-MT { background: var(--band); color: var(--muted); border-color: var(--rule); }
.prov-MTA { background: var(--band); color: var(--ink); border-color: var(--muted); }
.prov-G { background: transparent; color: var(--muted); border: 1px dashed var(--muted); }
.new { font-size: .72rem; font-weight: 700; color: var(--red); border: 1px solid var(--red); border-radius: 4px; padding: 0 .3rem; text-decoration: none; vertical-align: .1em; }
.t-strip { width: 11rem; min-width: 9rem; }
.strip { position: relative; display: block; height: 1.3rem; margin-top: .15rem; border-bottom: 1px solid var(--rule); background: linear-gradient(90deg, var(--rule) 1px, transparent 1px) 0 0 / 25% 100%; }
.strip i { position: absolute; bottom: .15rem; width: 2px; height: .9rem; transform: translateX(-1px); border-radius: 1px; }
.strip i.o { background: var(--open); opacity: .6; }
.strip i.c { background: var(--com); height: 1.1rem; }

/* Method flow */
.flow { padding: 1.4rem; display: grid; gap: 1.4rem; }
.flow-h { font: 700 .9rem var(--sans); margin: 0 0 .6rem; }
.steps { list-style: none; margin: 0; padding: 0; display: grid; gap: 1.6rem; }
@media (min-width: 60rem) { .steps { grid-auto-flow: column; grid-auto-columns: 1fr; gap: 2rem; } }
.steps li { position: relative; border: 1px solid var(--rule); border-radius: 8px; padding: .7rem .85rem; background: var(--bg); font-family: var(--sans); }
.steps li b { display: block; font-size: .95rem; }
.steps li span { display: block; font-size: .82rem; color: var(--muted); line-height: 1.4; margin-top: .15rem; }
.steps li + li::before { content: ""; position: absolute; left: 50%; top: -1.25rem; width: 0; height: 0; border: .4rem solid transparent; border-top-color: var(--muted); transform: translateX(-50%); }
@media (min-width: 60rem) {
  .steps li + li::before { left: -1.4rem; top: 50%; border-top-color: transparent; border-left-color: var(--muted); transform: translateY(-50%); }
  .flow-row:last-child .steps li:last-child { border-color: var(--accent); background: var(--accent-soft); }
}
.overview { margin: 1.4rem 0 0; padding: .8rem; background: #fff; }
.overview img { display: block; width: 100%; height: auto; }
.overview figcaption { font: .84rem var(--sans); color: #5A606B; margin: .7rem .2rem .1rem; }
.ov-scroll { overflow-x: auto; }
.ov-scroll img { min-width: 640px; }

/* Press */
.press { display: grid; gap: 2rem; }
@media (min-width: 60rem) { .press { grid-template-columns: 1.3fr 1fr; } }
.facts { margin: 0; padding: 0; }
.facts div { display: grid; grid-template-columns: 8.5rem 1fr; gap: 1rem; padding: .7rem 0; border-top: 1px solid var(--rule); }
.facts dt { font: 700 1.2rem/1.3 var(--serif); color: var(--accent); }
.facts dd { margin: 0; font-size: .98rem; }
.press h3 { font-size: 1.1rem; margin-top: 1.4rem; }
.press h3:first-child { margin-top: 0; }
.press ul { padding-left: 1.1rem; margin: .5rem 0 0; }

/* Cite */
.cite-grid { display: grid; gap: 2rem; }
@media (min-width: 60rem) { .cite-grid { grid-template-columns: 1.6fr 1fr; } }
.cite-head { display: flex; justify-content: space-between; align-items: center; gap: 1rem; }
pre.bib { margin: .8rem 0 0; background: var(--card); border: 1px solid var(--rule); border-radius: 8px; padding: 1rem 1.1rem; font: .78rem/1.6 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-x: auto; white-space: pre; }
.links { list-style: none; padding: 0; margin: .8rem 0 0; display: grid; gap: .35rem; font-family: var(--sans); font-size: .95rem; }
.alist { list-style: none; padding: 0; margin: .8rem 0 0; columns: 2; column-gap: 1.5rem; font-size: .96rem; }

footer { border-top: 1px solid var(--rule); padding: 2rem 0 3rem; font: .86rem/1.6 var(--sans); color: var(--muted); }
footer p { max-width: 90ch; }

@media (max-width: 40rem) {
  section.block { padding: 3rem 0; }
  .hero { padding-top: 2.5rem; }
  .row-head { flex-direction: column; gap: .1rem; }
  .fig, .lb, .tgroup, .flow { padding: 1rem .9rem; }
  .lb-table .c-name { min-width: 10rem; }
  .search { margin-left: 0; width: 100%; }
  .search input { width: 100%; }
  .facts div { grid-template-columns: 1fr; gap: .1rem; }
  .alist { columns: 1; }
  .cbar { width: 3.5rem; }
  .t-metric { display: none; }
}
@media (prefers-reduced-motion: reduce) { html { scroll-behavior: auto; } }
"""

JS = r"""
const root = document.documentElement, body = document.body;
const themeBtn = document.querySelector('.theme');
themeBtn.addEventListener('click', () => {
  const dark = root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches;
  root.dataset.theme = dark ? 'light' : 'dark';
  try { localStorage.setItem('skb-theme', root.dataset.theme); } catch (err) {}
});
document.querySelectorAll('.q').forEach(box => {
  box.addEventListener('click', ev => {
    const btn = ev.target.closest('button');
    if (!btn || box.dataset.done) return;
    box.dataset.done = '1';
    const a = +box.dataset.a, i = +btn.dataset.i;
    const buttons = box.querySelectorAll('button');
    buttons[a].classList.add('is-right');
    if (i !== a) btn.classList.add('is-wrong');
    buttons.forEach(b => b.setAttribute('aria-disabled', 'true'));
    const v = box.querySelector('.verdict');
    const s = document.createElement('strong');
    s.textContent = i === a ? body.dataset.right : body.dataset.wrong.replace('{x}', 'abcde'[a]);
    const st = document.createElement('span');
    st.textContent = box.dataset.stat || '';
    if (box.dataset.todo) st.className = 'todo';
    v.append(s, st);
  });
});
const quiz = document.querySelector('.quiz'), qm = document.querySelector('.quiz-more');
if (quiz && qm) {
  quiz.classList.add('js-quiz'); qm.hidden = false;
  qm.querySelector('button').addEventListener('click', () => { quiz.classList.remove('js-quiz'); qm.hidden = true; });
}
const lb = document.querySelector('.lb');
if (lb) {
  const tbody = lb.querySelector('tbody');
  const rows = [...tbody.rows];
  const more = lb.querySelector('.lb-more');
  const empty = lb.querySelector('.lb-empty');
  const input = lb.querySelector('input[type=search]');
  const show = +lb.dataset.show;
  let filter = 'all', expanded = false, q = '';
  const render = () => {
    const vis = rows.filter(r => (filter === 'all' || (r.dataset.open === '1') === (filter === 'open'))
      && (!q || r.dataset.name.toLowerCase().includes(q)));
    rows.forEach(r => { r.hidden = true; });
    const cap = expanded || q ? Infinity : show;
    vis.forEach((r, i) => { r.hidden = i >= cap; });
    more.hidden = !!q || vis.length <= show;
    more.textContent = expanded ? lb.dataset.less : lb.dataset.more.replace('{n}', vis.length);
    empty.hidden = vis.length > 0;
  };
  lb.querySelectorAll('.seg').forEach(seg => seg.addEventListener('click', ev => {
    const b = ev.target.closest('button'); if (!b) return;
    seg.querySelectorAll('button').forEach(x => x.setAttribute('aria-pressed', x === b));
    if (b.dataset.view) lb.dataset.view = b.dataset.view;
    if (b.dataset.f) filter = b.dataset.f;
    render();
  }));
  input.addEventListener('input', () => { q = input.value.trim().toLowerCase(); render(); });
  lb.querySelector('thead').addEventListener('click', ev => {
    const b = ev.target.closest('button'); if (!b) return;
    const th = b.parentElement, col = b.dataset.col;
    const asc = col === 'name' ? th.getAttribute('aria-sort') !== 'ascending' : th.getAttribute('aria-sort') === 'descending';
    lb.querySelectorAll('thead th').forEach(x => x.removeAttribute('aria-sort'));
    th.setAttribute('aria-sort', asc ? 'ascending' : 'descending');
    const key = r => col === 'name' ? r.dataset.name : +r.dataset.v.split(' ')[col];
    rows.sort((a, b) => {
      const x = key(a), y = key(b);
      const d = typeof x === 'string' ? x.localeCompare(y, root.lang) : x - y;
      return asc ? d : -d;
    });
    tbody.append(...rows);
    render();
  });
  more.addEventListener('click', () => { expanded = !expanded; render(); });
  render();
}
const links = [...document.querySelectorAll('.top nav a.nl')];
const secs = links.map(a => document.querySelector(a.getAttribute('href')));
if ('IntersectionObserver' in window) {
  const io = new IntersectionObserver(entries => {
    entries.forEach(en => {
      if (en.isIntersecting) links.forEach((a, i) => a.classList.toggle('on', secs[i] === en.target));
    });
  }, { rootMargin: '-40% 0px -55% 0px' });
  secs.forEach(s => s && io.observe(s));
}
const copy = document.querySelector('.copy');
if (copy) copy.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(document.querySelector('pre.bib').textContent);
    const label = copy.textContent;
    copy.textContent = copy.dataset.done;
    setTimeout(() => { copy.textContent = label; }, 1600);
  } catch (err) {
    const r = document.createRange(); r.selectNodeContents(document.querySelector('pre.bib'));
    const s = getSelection(); s.removeAllRanges(); s.addRange(r);
  }
});
"""

THEME_ICON = ('<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" stroke-width="2"/>'
              '<path d="M12 3a9 9 0 0 1 0 18z" fill="currentColor"/></svg>')
DOC_ICON = ('<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round">'
            '<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 16h6"/></svg>')
PRE_THEME = "try{var t=localStorage.getItem('skb-theme');if(t)document.documentElement.dataset.theme=t}catch(e){}"


def page(lang: str) -> str:
    t = T[lang]
    ids = [i for i, _ in t["nav"]]
    ov_id, try_id, f_id, lb_id, t_id, m_id, p_id, c_id = ids
    fids = [f"{'z' if lang == 'sk' else 'f'}{n}" for n in range(1, 8)]
    nav = "".join(f'<a class="nl" href="#{i}">{e(lbl)}</a>' for i, lbl in t["nav"])
    nav += f'<a class="lang-sw" href="{t["other_href"]}" hreflang="{t["other"]}" lang="{t["other"]}">{t["other_label"]}</a>'
    ds = [f'<a href="{u}">{e(n)}</a>' for u, n in DATASETS]
    url = SITE_URL + ("en/" if lang == "en" else "")
    og_img = f"{SITE_URL}og-{lang}.png"
    authors = "".join(f"<li>{e(a)}<sup>{s}</sup></li>" for a, s in AUTHORS)
    affil = "".join(f"<span><sup>{i}</sup>{e(a)}</span>" for i, a in enumerate(AFFIL[lang], 1))
    cards = "".join(f'<article class="ocard card"><p class="big">{e(big)}</p><h3>{e(h)}</h3><p>{e(p)}</p>'
                    f'<a href="#{fids[int(link[1:]) - 1]}">{e(lbl)}</a></article>' for big, h, p, link, lbl in t["cards"])
    stats = "".join(f"<div><dt>{e(a)}</dt><dd>{e(b)}</dd></div>" for a, b in t["stats"])
    facts = "".join(f"<div><dt>{e(a)}</dt><dd>{e(b)}</dd></div>" for a, b in t["p_facts"])
    links = "".join(f'<li><a href="{u}">{e(lbl)}</a></li>' for u, lbl in t["links"])
    alist = "".join(f"<li>{e(a)}</li>" for a, _ in AUTHORS)
    p = lambda k: f"<p>{e(t[k])}</p>"  # noqa: E731
    findings = [
        finding(1, fids[0], t, p("f1_p"), dot_rows(t, ["driving_test", "financial_exams", "maturita"])),
        finding(2, fids[1], t, p("f2_p"), dot_rows(t, ["chiby", "slovak_trivia"])),
        finding(3, fids[2], t, p("f3_p1") + p("f3_p2"), f'<div class="fig card">{open_gap(t)}</div>'),
        finding(4, fids[3], t, p("f4_p"), scatter(t)),
        finding(5, fids[4], t, p("f5_p"), cost_table(t)),
        finding(6, fids[5], t, p("f6_p"), finetune(t)),
        finding(7, fids[6], t, p("f7_p"), lang_table(t)),
    ]
    more_id = "dalsie" if lang == "sk" else "more"
    return f"""<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(t["title"])}</title>
<meta name="description" content="{e(t["description"])}">
<link rel="canonical" href="{url}">
<link rel="alternate" hreflang="sk" href="{SITE_URL}">
<link rel="alternate" hreflang="en" href="{SITE_URL}en/">
<link rel="alternate" hreflang="x-default" href="{SITE_URL}en/">
<meta name="theme-color" content="#FAF9F6" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#121418" media="(prefers-color-scheme: dark)">
<meta property="og:type" content="website">
<meta property="og:site_name" content="sk-bench">
<meta property="og:locale" content="{t["og_locale"]}">
<meta property="og:locale:alternate" content="{T[t["other"]]["og_locale"]}">
<meta property="og:title" content="{e(t["h1"])}">
<meta property="og:description" content="{e(t["description"])}">
<meta property="og:url" content="{url}">
<meta property="og:image" content="{og_img}">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="{e(t["og_alt"])}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{e(t["h1"])}">
<meta name="twitter:description" content="{e(t["description"])}">
<meta name="twitter:image" content="{og_img}">
<meta name="twitter:image:alt" content="{e(t["og_alt"])}">
<meta name="citation_title" content="sk-bench: A Native-First Benchmark for Evaluating Large Language Models in Slovak">
{"".join(f'<meta name="citation_author" content="{e(a)}">' for a, _ in AUTHORS)}
<meta name="citation_publication_date" content="2026">
<meta name="citation_arxiv_id" content="2610.09152">
<meta name="citation_pdf_url" content="https://arxiv.org/pdf/2610.09152">
<link rel="icon" href="{t["root"]}favicon.svg" type="image/svg+xml">
<link rel="apple-touch-icon" href="{t["root"]}apple-touch-icon.png">
<script type="application/ld+json">{json_ld(t, url)}</script>
<script>{PRE_THEME}</script>
<style>{CSS}</style>
</head>
<body data-right="{e(t["right"])}" data-wrong="{e(t["wrong"])}">
<header class="top"><div class="wrap">
  <a class="brand" href="#"><b>sk-bench</b><span>{e(t["short"])}</span></a>
  <nav aria-label="{e(t["short"])}">{nav}</nav>
  <button class="theme" type="button" aria-label="{e(t["theme"])}" title="{e(t["theme"])}">{THEME_ICON}</button>
</div></header>

<main>
<div class="wrap hero">
  <h1>{e(t["h1"])}</h1>
  <p class="subtitle">{e(t["subtitle"])}</p>
  <ul class="authors">{authors}</ul>
  <p class="affil">{affil}</p>
  <p class="corr">{e(t["corr"])}: <a href="mailto:{CONTACT}">{CONTACT}</a></p>
  <div class="actions">
    <a class="btn primary" href="{PAPER}">{DOC_ICON}{e(t["btn_paper"])}</a>
    <a class="btn" href="#{lb_id}">{e(t["btn_board"])}</a>
    <a class="btn" href="{CODE}">{e(t["btn_code"])}</a>
    <a class="btn" href="#{c_id}">{e(t["btn_cite"])}</a>
  </div>
  <p class="venue">{e(t["venue"]).replace("{emnlp}", f'<a href="{EMNLP}">EMNLP 2026</a>')}</p>
</div>

<section class="block band" id="{ov_id}" aria-labelledby="{ov_id}-h"><div class="wrap">
  <div class="sec-head"><h2 id="{ov_id}-h">{e(t["ov_h2"])}</h2><p class="intro">{e(t["ov_lead"])}</p></div>
  <div class="cards">{cards}</div>
  <dl class="stats">{stats}</dl>
</div></section>

<section class="block" id="{try_id}" aria-labelledby="{try_id}-h"><div class="wrap">
  <div class="sec-head"><h2 id="{try_id}-h">{e(t["try_h2"])}</h2><p class="intro">{e(t["try_intro"])}</p></div>
  <div class="quiz">{"".join(question(q, t) for q in QUESTIONS)}</div>
  <p class="quiz-more" hidden><button class="btn small" type="button">{e(t["quiz_more"])}</button></p>
</div></section>

<section class="block" id="{f_id}" aria-labelledby="{f_id}-h"><div class="wrap">
  <div class="sec-head"><h2 id="{f_id}-h">{e(t["f_h2"])}</h2><p class="intro">{e(t["f_intro"])}</p></div>
  <div class="findings">{"".join(findings[:3])}</div>
</div></section>

<section class="block band" id="{lb_id}" aria-labelledby="{lb_id}-h"><div class="wrap">
  <div class="sec-head"><h2 id="{lb_id}-h">{e(t["lb_h2"])}</h2><p class="intro">{e(t["lb_intro"])}</p></div>
  {leaderboard(t)}
</div></section>

<section class="block" id="{more_id}" aria-labelledby="{more_id}-h"><div class="wrap">
  <div class="sec-head"><h2 id="{more_id}-h">{e(t["more_h2"])}</h2><p class="intro">{e(t["more_intro"])}</p></div>
  <div class="findings">{"".join(findings[3:])}</div>
</div></section>

<section class="block" id="{t_id}" aria-labelledby="{t_id}-h"><div class="wrap">
  <div class="sec-head"><h2 id="{t_id}-h">{e(t["t_h2"])}</h2><p class="intro">{e(t["t_intro"])}</p></div>
  {tasks_section(t)}
</div></section>

<section class="block band" id="{m_id}" aria-labelledby="{m_id}-h"><div class="wrap">
  <div class="sec-head"><h2 id="{m_id}-h">{e(t["m_h2"])}</h2><p class="intro">{e(t["m_intro"])}</p></div>
  {flow(t)}
  <figure class="overview card">
    <div class="ov-scroll"><img src="{t["root"]}sk-bench-overview.svg" alt="{e(t["fig_caption"])}" width="885" height="288" loading="lazy"></div>
    <figcaption>{e(t["fig_caption"])}</figcaption>
  </figure>
</div></section>

<section class="block" id="{p_id}" aria-labelledby="{p_id}-h"><div class="wrap">
  <div class="sec-head"><h2 id="{p_id}-h">{e(t["p_h2"])}</h2><p class="intro">{e(t["p_intro"])}</p></div>
  <div class="press">
    <dl class="facts">{facts}</dl>
    <div>
      <h3>{e(t["p_note_h"])}</h3><p>{e(t["p_note"])}</p>
      <h3>{e(t["p_fig_h"])}</h3><p><a href="{t["root"]}sk-bench-overview.svg" download>{e(t["p_fig"])}</a></p>
      <h3>{e(t["p_links_h"])}</h3><ul>{links}</ul>
    </div>
  </div>
</div></section>

<section class="block band" id="{c_id}" aria-labelledby="{c_id}-h"><div class="wrap">
  <div class="cite-grid">
    <div>
      <div class="cite-head"><h2 id="{c_id}-h">{e(t["c_h2"])}</h2><button class="btn small copy" type="button" data-done="{e(t["copied"])}">{e(t["copy"])}</button></div>
      <pre class="bib">{e(BIBTEX)}</pre>
    </div>
    <div>
      <h3>{e(t["authors_h3"])}</h3>
      <ul class="alist">{alist}</ul>
      <ul class="links">{links}</ul>
    </div>
  </div>
</div></section>
</main>

<footer><div class="wrap">
  <p>{e(t["foot1"])}</p>
  <p>{t["foot2"].format(d1=ds[0], d2=ds[1], d3=ds[2])}</p>
</div></footer>
<script>{JS}</script>
</body>
</html>
"""


SITEMAP = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">
""" + "".join(f"""  <url><loc>{SITE_URL}{p}</loc>
    <xhtml:link rel="alternate" hreflang="sk" href="{SITE_URL}"/>
    <xhtml:link rel="alternate" hreflang="en" href="{SITE_URL}en/"/>
  </url>
""" for p in ("", "en/")) + "</urlset>\n"


def json_ld(t: dict, url: str) -> str:
    """schema.org data for Google: the paper and the benchmark as a dataset."""
    people = [{"@type": "Person", "name": a} for a, _ in AUTHORS]
    data = {"@context": "https://schema.org", "@graph": [
        {"@type": "ScholarlyArticle", "@id": "https://arxiv.org/abs/2610.09152",
         "headline": "sk-bench: A Native-First Benchmark for Evaluating Large Language Models in Slovak",
         "author": people, "datePublished": "2026", "inLanguage": "en",
         "url": "https://arxiv.org/abs/2610.09152", "sameAs": ["https://arxiv.org/pdf/2610.09152"],
         "about": {"@id": SITE_URL + "#dataset"}},
        {"@type": "Dataset", "@id": SITE_URL + "#dataset", "name": "sk-bench",
         "description": t["description"], "url": url, "inLanguage": "sk", "creator": people,
         "isAccessibleForFree": True, "keywords": ["Slovak", "large language models", "benchmark", "evaluation"],
         "sameAs": ["https://github.com/slovak-nlp/sk-bench", "https://huggingface.co/collections/slovak-nlp/sk-bench"],
         "citation": {"@id": "https://arxiv.org/abs/2610.09152"}},
    ]}
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def main() -> None:
    if DIST.exists():
        shutil.rmtree(DIST)
    (DIST / "en").mkdir(parents=True)
    (DIST / "data").mkdir()
    (DIST / "index.html").write_text(page("sk"))
    (DIST / "en" / "index.html").write_text(page("en"))
    (DIST / ".nojekyll").write_text("")
    shutil.copy(ROOT / "data" / "leaderboard.json", DIST / "data" / "leaderboard.json")
    (DIST / "data" / "sk-bench-leaderboard.csv").write_text(to_csv())
    shutil.copy(ROOT.parent / "images" / "sk-bench-overview.svg", DIST / "sk-bench-overview.svg")
    for f in ("og-sk.png", "og-en.png", "favicon.svg", "apple-touch-icon.png"):
        shutil.copy(ROOT / "og" / f, DIST / f)
    (DIST / "sitemap.xml").write_text(SITEMAP)
    print("built", sorted(str(p.relative_to(DIST)) for p in DIST.rglob("*") if p.is_file()))


if __name__ == "__main__":
    main()
