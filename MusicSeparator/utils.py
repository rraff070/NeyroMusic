import hashlib
from pathlib import Path
from typing import Tuple, Optional

import librosa
import numpy as np
import torch
import soundfile as sf
from pydub import AudioSegment, effects
from config import SAMPLE_RATE, CACHE_DIR
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Считаем уникальный отпечаток файла по его содержимому
# Это нужно, чтобы понять — обрабатывали мы уже такой файл или нет
def compute_file_hash(file_path: Path) -> str:
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        # Читаем файл кусочками по 4 КБ, чтобы не загружать огромные файлы в память целиком
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


# По хешу файла и имени стема получаем путь, где лежит (или будет лежать) кэшированный результат
def get_cached_path(file_hash: str, stem_name: str) -> Path:
    return CACHE_DIR / f"{file_hash}_{stem_name}.wav"


# Проверяем, есть ли в кэше все четыре стема для этого файла
def is_cached(file_hash: str, stem_names: list) -> bool:
    for stem in stem_names:
        if not get_cached_path(file_hash, stem).exists():
            return False
    return True


# Загружаем аудиофайл и приводим его к нужной частоте
# Если один способ не сработал, пробуем другой
def load_audio(file_path: Path, target_sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    # Пробуем сначала librosa — она умеет много форматов
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, mono=False)
    except Exception as e:
        # Если librosa не справилась (например, с mp4), используем pydub
        audio_segment = AudioSegment.from_file(file_path)
        audio_segment = effects.normalize(audio_segment)  # Выравниваем громкость
        audio_segment = audio_segment.set_frame_rate(target_sr)

        # Превращаем pydub-объект в массив numpy
        audio = np.array(audio_segment.get_array_of_samples())
        if audio_segment.channels == 2:
            audio = audio.reshape(-1, 2).T  # Стерео -> два канала
        else:
            audio = audio.reshape(1, -1)    # Моно -> один канал
        audio = audio.astype(np.float32) / (2**15)  # Приводим к диапазону от -1 до 1
        sr = target_sr

    # Если получился одномерный массив (моно), добавляем измерение канала
    if audio.ndim == 1:
        audio = audio.reshape(1, -1)

    return audio, sr


# Сохраняем аудио в WAV-файл
def save_audio(file_path: Path, audio: torch.Tensor, sample_rate: int):
    # Если пришёл тензор из PyTorch, превращаем его в обычный массив numpy
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    # Soundfile ожидает формат (сэмплы, каналы), а у нас обычно (каналы, сэмплы)
    # Если два канала — меняем местами
    if audio.ndim == 2 and audio.shape[0] == 2:
        audio = audio.T

    sf.write(file_path, audio, sample_rate)


# Отрезаем тишину в начале и конце трека
# Это ускоряет обработку и не влияет на качество
def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    top_db: int = 20,
    margin_ms: int = 100
) -> np.ndarray:
    # Смешиваем все каналы в один моно, чтобы искать тишину
    if audio.shape[0] > 1:
        mono = audio.mean(axis=0)
    else:
        mono = audio[0]

    # Находим участки, где громкость выше порога top_db
    non_silent = librosa.effects.split(
        mono, top_db=top_db, frame_length=2048, hop_length=512
    )

    # Если весь файл оказался тишиной — возвращаем как есть
    if len(non_silent) == 0:
        return audio

    # Добавляем небольшой отступ, чтобы не обрезать прямо в плотную
    margin_samples = int(margin_ms * sample_rate / 1000)

    start = max(0, non_silent[0][0] - margin_samples)
    end = min(audio.shape[1], non_silent[-1][1] + margin_samples)

    return audio[:, start:end]


# Простая проверка файла перед загрузкой: подходит ли формат и размер
def validate_audio_file(file_path: Path, max_size_mb: int = 50) -> bool:
    # Проверяем расширение
    if file_path.suffix.lower() not in {".wav", ".mp3", ".flac", ".m4a"}:
        return False

    # Проверяем размер
    if file_path.stat().st_size > max_size_mb * 1024 * 1024:
        return False

    return True


# Узнаём длительность трека, не загружая его полностью
def get_audio_duration(file_path: Path) -> float:
    try:
        # Пробуем через librosa (быстро, потому что sr=None не передискретизирует)
        audio, sr = librosa.load(file_path, sr=None, mono=True)
        return len(audio) / sr
    except Exception:
        # Если не получилось — через pydub
        audio = AudioSegment.from_file(file_path)
        return len(audio) / 1000  # pydub возвращает длительность в миллисекундах