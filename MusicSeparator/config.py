from pathlib import Path
import torch

# Находим папку с текущим файлом, от неё будем строить все пути
BASE_DIR = Path(__file__).parent.absolute()

# Папки для хранения результатов, кэша и датасета
OUTPUT_DIR = BASE_DIR / "output"
CACHE_DIR = BASE_DIR / "cache"
MUSDB_PATH = BASE_DIR / "musdb18"

# Создаём папки, если их нет
OUTPUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
MUSDB_PATH.mkdir(exist_ok=True)

# Название модели для разделения
MODEL_NAME = "htdemucs"

# Автоматически выбираем устройство: видеокарта если есть, иначе процессор
if torch.cuda.is_available():
    DEVICE = "cuda"
    print(f"CUDA доступен, использую GPU: {torch.cuda.get_device_name(0)}")
else:
    DEVICE = "cpu"
    print("CUDA не доступен, использую CPU (обработка будет медленнее)")

# Частота дискретизации для обработки аудио
SAMPLE_RATE = 44100

# Длина одного кусочка и перекрытие между ними (для длинных треков)
SEGMENT_DURATION = 10.0
OVERLAP_DURATION = 1.0

# Ограничения для загружаемых файлов
MAX_FILE_SIZE_MB = 50
MAX_DURATION_SECONDS = 600

# Какие форматы поддерживаем
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a"}

# Имена стемов в том порядке, в котором их выдаёт модель
STEM_NAMES = ["drums", "bass", "other", "vocals"]

# Ограничения для отображения на веб-странице
MAX_FILE_SIZE_MB_DISPLAY = 50
MAX_DURATION_MINUTES = 10