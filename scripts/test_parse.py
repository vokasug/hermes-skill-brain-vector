#!/usr/bin/env python3
"""Самопроверка parse() из brain_index.py — без фреймворков.

Запуск:
  ~/.local/share/uv/tools/mlx-embeddings/bin/python scripts/test_parse.py
Ожидаемый вывод: SELFTEST OK (при расхождении — AssertionError с сутью).
Гонять после любой правки parse() или правил нарезки на абзацы.
"""
import importlib.util
import pathlib
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("brain_index", HERE / "brain_index.py")
bi = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bi)

HEAD = ("- **Название:** Название материала\n- **Автор:** Автор\n"
        "- **Дата публикации:** 2026-01-01\n- **Ссылка:** https://example.com\n")

BODY_BULLETS = """
# Главная мысль
Одна мысль про важное.

# Ключевые тезисы
- Первый тезис достаточно длинный.
- Второй тезис тоже длинный.

# Вывод
Короткий вывод по материалу.
"""

# тот же материал нормальным текстом: пустая строка между пунктами — по правилу скилла
BODY_PLAIN = """
# Главная мысль
Одна мысль про важное.

# Ключевые тезисы
Первый тезис достаточно длинный.

Второй тезис тоже длинный.

# Вывод
Короткий вывод по материалу.
"""

# нумерованный тесный список: пункты разделяются маркером, без пустых строк
BODY_NUMBERED = """
# Главная мысль
Одна мысль про важное.

# Ключевые тезисы
1. Первый тезис достаточно длинный.
2) Второй тезис тоже длинный.

# Вывод
Короткий вывод по материалу.
"""

# тесный список без маркеров и без пустых строк неразделим — это ожидаемое поведение
BODY_TIGHT_PLAIN = """
# Главная мысль
Одна мысль про важное.

# Ключевые тезисы
Первый тезис достаточно длинный.
Второй тезис тоже длинный.

# Вывод
Короткий вывод по материалу.
"""

BODY_SHORT = """
# Главная мысль
Мысль.

# Ключевые тезисы
Очень коротко.

# Вывод
Итог.
"""


def parse_text(text):
    with tempfile.TemporaryDirectory() as d:
        p = pathlib.Path(d) / "probe_sum.md"
        p.write_text(text, encoding="utf-8")
        return bi.parse(p)


def sections_of(parsed):
    return [s for s, _ in parsed[1]]


def texts_of(parsed):
    return [t for _, t in parsed[1]]


# 1. базовая нарезка: мысль / каждый тезис / вывод, маркеры срезаны
ref = parse_text(HEAD + BODY_BULLETS)
assert sections_of(ref) == ["мысль", "тезис", "тезис", "вывод"], sections_of(ref)
assert not any(t.startswith(("- ", "* ", "1.")) for t in texts_of(ref)), "маркеры не срезаны"

# 2–3. маркер не важен: без дефисов (с пустыми строками) и нумерованный тесный — то же самое
assert parse_text(HEAD + BODY_PLAIN) == ref, "абзацы без дефисов дают другой результат"
assert texts_of(parse_text(HEAD + BODY_NUMBERED)) == texts_of(ref), "нумерованный список даёт другой результат"

# 4. тесный список без маркеров неразделим — две строки остаются одной записью
assert sections_of(parse_text(HEAD + BODY_TIGHT_PLAIN)) == ["мысль", "тезис", "вывод"], \
    "тесный список без маркеров должен оставаться одной записью"

# 5. лишний заголовок перед шапкой не роняет файл
pre = parse_text("# Заголовок\n\n" + HEAD + BODY_BULLETS)
assert pre is not None, "файл с заголовком перед шапкой не распарсился"
assert pre[0]["author"] == "Автор", pre[0]

# 6. короткие абзацы не отбрасываются (фильтра по длине нет)
short = parse_text(HEAD + BODY_SHORT)
assert short is not None and sections_of(short) == ["мысль", "тезис", "вывод"], short
assert texts_of(short) == ["Мысль.", "Очень коротко.", "Итог."], texts_of(short)

# 7. файл без шапки метаданных — None (сигнал «!! не распарсился»)
assert parse_text("# Главная мысль\nМысль без шапки.\n") is None, "нет шапки, а parse не вернул None"

print("SELFTEST OK")
