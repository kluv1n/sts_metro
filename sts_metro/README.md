# sts_metro

Локальный пайплайн для СТС: препроцесс изображения → OCR → извлечение ФИО (эвристики) → NER по полям владельца (опционально). Ниже — **в каком порядку что запускать** и куда смотреть артефакты.

## Окружение

Из корня **`sts_metro`**:

```bash
cd sts_metro
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Дополнительно по задачам:

| Задача | Установка |
|--------|-----------|
| EasyOCR | `pip install -r requirements-ocr-easyocr.txt` (первый запуск качает веса; на macOS при SSL-сбоях помогает certifi из этого файла или «Install Certificates.command» в установке Python) |
| Обучение NER | `pip install -r requirements-ner.txt` |

Скрипт **`scripts/train_ner.sh`** сам подхватывает **`.venv/bin/python`**, если он есть; иначе `python3` или интерпретатор из **`PYTHON`**. Зависимости NER ставит через **`python -m pip`**.

Во всех командах ниже предполагается: **`cd sts_metro`** и при необходимости **`source .venv/bin/activate`**.

---

## 1. Куда класть фото

Исходники — в **`data/input/`** (в репозиторий обычно не коммитят, только локально):

```text
sts_metro/data/input/sts_01.jpg
sts_metro/data/input/sts_02.jpg
```

---

## 2. Только CV (отладка шагов)

Один файл → папка с шагами (`01_raw` … **`06_for_ocr_default.png`** и т.д.):

```bash
PYTHONPATH=src python scripts/run_cv_debug.py -i data/input/sts_01.jpg -o debug/sts_01
```

Вся папка:

```bash
PYTHONPATH=src python scripts/run_cv_debug.py -d data/input -o debug/batch_$(date +%Y%m%d)
```

Полезные флаги (кратко): **`--profile default|minimal|aggressive`**, **`--no-warp`**, **`--no-deskew`**, **`--no-orient`**, **`--bbox-crop`**, **`--force-180`**, **`--no-upright-180`**. Подробная логика deskew / upright — в **`src/sts_cv/pipeline.py`** и в логе `run_cv_debug.py` (`tb_desk`, `ud_sig`, `res180`, `tb_out`).

---

## 3. CV + OCR (EasyOCR или Tesseract)

**Папка с фото** → подпапки на каждый файл: шаги CV, **`09_ocr.txt`**, **`09_ocr_meta.txt`**.

EasyOCR (рекомендуется для грязных фото):

```bash
PYTHONPATH=src python scripts/run_ocr_debug.py -d data/input -o debug/ocr_easy --ocr-engine easyocr
```

Один файл:

```bash
PYTHONPATH=src python scripts/run_ocr_debug.py -i data/input/sts_01.jpg -o debug/ocr_one --ocr-engine easyocr
```

Флаг **`--easyocr-gpu`** — если в PyTorch доступен GPU.

Tesseract (нужен системный `tesseract`, языки rus+eng):

```bash
brew install tesseract tesseract-lang
PYTHONPATH=src python scripts/run_ocr_debug.py -d data/input -o debug/ocr_tess --ocr-engine tesseract
```

Явный путь к бинарнику: **`--tesseract /path/to/tesseract`** (или переменные **`TESSERACT_CMD`** / **`TESSERACT_PATH`**). Для Tesseract в коде перебираются PSM **3 / 4 / 6** (без 11, чтобы не срываться в сотни фрагментов на шуме). В `run_ocr_debug.py` подавляется предупреждение PyTorch про **`pin_memory` / MPS**.

---

## 4. ФИО из уже готового OCR (эвристики, без NER)

По корню с подпапками, где лежит **`09_ocr.txt`** (как после `run_ocr_debug.py`):

```bash
PYTHONPATH=src python scripts/extract_fio.py --ocr-root debug/ocr_easy
```

В каждой подпапке появится **`10_fio.json`**: **`fio`**, **`fio_cyrillic`** / **`parts_cyrillic`**, **`fio_latin`** / **`parts_latin`**, **`confidence`**, **`note`**, **`primary_language`**, **`primary_reason`**, плюс поля **`surname_ru`**, **`name_ru`**, … (см. **`src/sts_fio/extract.py`**).

---

## 5. Полный путь «картинка → OCR → NER» (один скрипт)

Нужны зависимости **`requirements.txt`** + **`requirements-ner.txt`** и (по умолчанию) **`requirements-ocr-easyocr.txt`** для EasyOCR.

```bash
PYTHONPATH=src python scripts/run_sts_pipeline.py -i data/input/sts_01.jpg --out-json out/pipeline_sts_01.json
```

Папка с картинками: **`--input-dir`**, каталог с одним JSON на файл или один общий **`--out-json`**. Модель NER по умолчанию: **`source_model_train/ner_model`** (**`--ner-model`**).

---

## 6. NER: разметка, сплиты, обучение, оценка

1. Разметка лежит в **`source_model_train/file*.json`** (и т.п.), исключения документов — **`source_model_train/holdout_doc_ids.txt`** (строки с **`#`** — комментарии).

2. Сбор сплитов:

```bash
PYTHONPATH=src python scripts/prepare_ner_splits.py
```

Появится **`source_model_train/splits/`** (`train.json`, `val.json`, `test.json`, `holdout.json`, `split_meta.json`). Список **`doc_id`**: **`PYTHONPATH=src python scripts/prepare_ner_splits.py --list-doc-ids`**.

3. Обучение + eval одной командой:

```bash
bash scripts/train_ner.sh
```

Или вручную: **`train_sts_ner.py`** (чекпоинт в **`source_model_train/ner_model`**) затем **`eval_sts_ner.py`** на **`splits/test.json`**. Отчёты: **`ner_model/test_metrics.json`**, **`test_classification_report.txt`**.

4. Инференс по одному тексту или файлу:

```bash
PYTHONPATH=src python scripts/infer_sts_ner.py -f path/to/09_ocr.txt
```

Токенизатор: **`use_fast=True`**, **`fix_mistral_regex=False`** (как в обучении).

---

## 7. Сравнение профилей OCR в `debug/ocr_compare` (несколько вариантов на одну STS)

Удобная раскладка (пример): **`debug/ocr_compare/<профиль>/sts_XX/`** в каждой папке **`09_ocr.txt`**, **`10_fio.json`** (после **`extract_fio.py`** по соответствующему корню). Профилей может быть несколько (например **`default`**, **`bw`**, **`bw_inv`**).

**Пакетный NER** по всем `*.json` под корнем (по умолчанию **`debug/ocr_compare`**): для каждого JSON берётся текст из **`09_ocr.txt`** или первого **`*_ocr.txt`** в той же папке.

```bash
PYTHONPATH=src python scripts/batch_infer_debug_ner.py
```

Результат: **`debug/ner_batch_output.json`** (список **`items`**: путь к json, **`ocr_source`**, **`entities`**, при ошибке — **`error`**).

**Сводка «правила + NER» по каждой STS** (слияние **`10_fio.json`** и строк из **`ner_batch_output.json`**):

```bash
PYTHONPATH=src python scripts/merge_debug_sts_summary.py
```

Появляются:

- **`debug/sts_fio_ner_merged.json`** — все **`sts_XX`** и все варианты в одном файле;
- **`debug/sts_merged/sts_01.json`** … — по одному файлу на документ (отключить второе: **`--no-per-sts-files`**).

---

## 8. Импорт из кода

- CV: **`from sts_cv import preprocess, load_image, PreprocessConfig`**
- OCR: **`from sts_ocr import …`** (EasyOCR / Tesseract, см. **`src/sts_ocr`**)
- NER: **`from sts_ner import load_ner, predict_entities`** (или датасет **`StsNerDataset`**)

---

## Краткий порядок «с нуля до отчёта по дебагу»

1. Положить фото в **`data/input/`**.  
2. **`run_ocr_debug.py`** → **`debug/...`** с **`09_ocr.txt`**.  
3. **`extract_fio.py`** → **`10_fio.json`**.  
4. (Опционально) несколько профилей собрать под **`debug/ocr_compare/...`**.  
5. **`batch_infer_debug_ner.py`** → **`debug/ner_batch_output.json`**.  
6. **`merge_debug_sts_summary.py`** → **`debug/sts_fio_ner_merged.json`** и **`debug/sts_merged/*.json`**.

Для обучения NER на своей разметке: пункты из раздела **6** вместо шагов 4–6.
