import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torchaudio
from tqdm import tqdm

from config import (
    MODEL_NAME, DEVICE, SAMPLE_RATE,
    SEGMENT_DURATION, OVERLAP_DURATION,
    STEM_NAMES, OUTPUT_DIR
)
from utils import (
    load_audio, save_audio, trim_silence,
    compute_file_hash, get_cached_path, is_cached
)


class MusicSeparator:
    """
    Класс для разделения музыки на инструментальные дорожки
    Использует предобученную модель Hybrid Demucs из torchaudio
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str = None,
        sample_rate: int = SAMPLE_RATE,
        segment_duration: float = SEGMENT_DURATION,
        overlap_duration: float = OVERLAP_DURATION
    ):
        """
        Инициализация разделителя
        """
        self.device = device if device else DEVICE
        self.sample_rate = sample_rate
        self.segment_duration = segment_duration
        self.overlap_duration = overlap_duration
        self.stem_names = STEM_NAMES

        print(f"Инициализация разделителя на устройстве: {self.device}")

        # Загружаем модель один раз при создании объекта
        self.model = self._load_model(model_name)

        # Сюда будем записывать статистику по каждому файлу
        self.stats = {}

    def _load_model(self, model_name: str):
        """
        Загрузка предобученной модели Hybrid Demucs
        Пробуем разные варианты, если один не загружается
        """
        print(f"Загрузка модели {model_name}...")

        # Сначала пробуем самую качественную версию модели
        try:
            bundle = torchaudio.pipelines.HDEMUCS_HIGH_MUSDB_PLUS
            model = bundle.get_model()
            print("Загружена HDEMUCS_HIGH_MUSDB_PLUS")
        except Exception as e:
            print(f"Не удалось загрузить HDEMUCS_HIGH_MUSDB_PLUS: {e}")
            print("Пробуем загрузить HDEMUCS_BASIC_MUSDB...")
            try:
                bundle = torchaudio.pipelines.HDEMUCS_BASIC_MUSDB
                model = bundle.get_model()
                print("Загружена HDEMUCS_BASIC_MUSDB")
            except:
                # Последний вариант — грузим с торча хаба
                print("Загружаем Demucs из torch.hub...")
                model = torch.hub.load('facebookresearch/demucs', 'demucs', pretrained=True)

        # Перекладываем модель на нужное устройство (видеокарта или процессор)
        model = model.to(self.device)
        model.eval()  # Переключаем в режим предсказания (не обучения)

        print(f"Модель загружена на {self.device}. Параметров: {sum(p.numel() for p in model.parameters()):,}")
        return model

    def _chunk_audio(
        self,
        audio: torch.Tensor
    ) -> Tuple[List[torch.Tensor], int, int, float]:
        """
        Разрезаем длинное аудио на маленькие кусочки (чанки)
        Кусочки перекрываются, чтобы на стыках не было щелчков
        """
        channels, length = audio.shape

        # Переводим секунды в количество семплов
        chunk_len = int(self.sample_rate * self.segment_duration)
        overlap_len = int(self.sample_rate * self.overlap_duration)
        hop_len = chunk_len - overlap_len  # На сколько сдвигаемся при каждом шаге

        chunks = []
        start = 0

        while start < length:
            end = min(start + chunk_len, length)
            chunk = audio[:, start:end]

            # Если последний кусочек короче остальных — добиваем тишиной до нужной длины
            if chunk.shape[1] < chunk_len:
                pad_len = chunk_len - chunk.shape[1]
                chunk = torch.nn.functional.pad(chunk, (0, pad_len))

            chunks.append(chunk)
            start += hop_len

        return chunks, chunk_len, hop_len, overlap_len

    def _merge_chunks(
        self,
        chunks: List[torch.Tensor],
        original_length: int,
        hop_len: int,
        overlap_len: int
    ) -> torch.Tensor:
        """
        Склеиваем обработанные кусочки обратно в одно целое аудио
        Используем плавное затухание на стыках, чтобы не было щелчков
        """
        n_sources = chunks[0].shape[0] if len(chunks) > 0 else 0
        result_length = original_length
        result = torch.zeros(n_sources, result_length, device=self.device)
        weights = torch.zeros(result_length, device=self.device)

        for i, chunk in enumerate(chunks):
            start = i * hop_len
            end = start + chunk.shape[1]
            end = min(end, result_length)

            chunk_end = min(chunk.shape[1], result_length - start)

            # Создаём веса для плавного смешивания
            chunk_weights = torch.ones(chunk.shape[1], device=self.device)

            # Начало файла — плавно появляемся
            if i > 0 and overlap_len > 0:
                fade_len = min(overlap_len, chunk.shape[1])
                fade_in = torch.linspace(0, 1, fade_len, device=self.device)
                chunk_weights[:fade_len] = fade_in

            # Конец файла — плавно затухаем
            if i < len(chunks) - 1 and overlap_len > 0:
                fade_len = min(overlap_len, chunk.shape[1])
                fade_out = torch.linspace(1, 0, fade_len, device=self.device)
                chunk_weights[-fade_len:] = fade_out

            # Добавляем взвешенный кусочек в общий результат
            result[:, start:start + chunk_end] += chunk[:, :chunk_end] * chunk_weights[:chunk_end]
            weights[start:start + chunk_end] += chunk_weights[:chunk_end]

        # Делим на сумму весов, чтобы нормализовать громкость
        weights = torch.clamp(weights, min=1e-8)
        result = result / weights

        return result

    @torch.no_grad()  # Отключаем вычисление градиентов — нам не нужно обучать модель, только предсказывать
    def separate(
        self,
        audio_path: Path,
        output_dir: Optional[Path] = None,
        use_cache: bool = True,
        trim_silence_before: bool = True
    ) -> Dict[str, Path]:
        """
        Основной метод для разделения аудио на стемы
        Здесь происходит вся магия
        """
        start_time = time.time()

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Файл не найден: {audio_path}")

        # Проверяем, нет ли уже результата в кэше (чтобы не обрабатывать тот же файл дважды)
        file_hash = compute_file_hash(audio_path)
        if use_cache and is_cached(file_hash, self.stem_names):
            print(f"Использование кэшированных результатов для {audio_path.name}")
            result_paths = {}
            for stem in self.stem_names:
                result_paths[stem] = get_cached_path(file_hash, stem)
            return result_paths

        print(f"Загрузка аудио: {audio_path.name}")
        audio, sr = load_audio(audio_path, self.sample_rate)

        # Вырезаем тишину в начале и конце — это ускоряет обработку
        if trim_silence_before:
            original_len = audio.shape[1]
            audio = trim_silence(audio, sr)
            trimmed_len = audio.shape[1]
            if original_len > trimmed_len:
                print(f"Обрезано тишины: {(original_len - trimmed_len) / sr:.1f} сек")

        # Превращаем массив numpy в тензор torch и отправляем на нужное устройство
        audio_tensor = torch.from_numpy(audio).float().to(self.device)

        # Добавляем размерность для батча (модель ожидает 3 измерения: batch, channels, samples)
        if audio_tensor.dim() == 2:
            audio_tensor = audio_tensor.unsqueeze(0)

        total_duration = audio_tensor.shape[2] / self.sample_rate
        print(f"Длительность аудио: {total_duration:.1f} секунд")

        # Если трек короткий — обрабатываем целиком, не режем на кусочки
        if total_duration <= self.segment_duration * 2:
            print("Аудио короткое, обрабатываем целиком...")
            sources = self.model(audio_tensor)[0]  # Модель возвращает все стемы сразу

            # Раскладываем результат по именам стемов
            results = {}
            for stem_idx, stem in enumerate(self.stem_names):
                if stem_idx < sources.shape[0]:
                    results[stem] = sources[stem_idx].cpu()  # Перекладываем обратно в оперативную память
                else:
                    results[stem] = torch.zeros_like(audio_tensor[0])
        else:
            # Для длинных треков — режем на кусочки и обрабатываем каждый отдельно
            print(f"Разделение на чанки (длительность: {self.segment_duration}с, перекрытие: {self.overlap_duration}с)...")

            chunks, chunk_len, hop_len, overlap_len = self._chunk_audio(audio_tensor[0])
            print(f"Создано {len(chunks)} чанков")

            # Будем хранить обработанные кусочки для каждого стема отдельно
            separated_chunks = {stem: [] for stem in self.stem_names}

            for idx, chunk in enumerate(tqdm(chunks, desc="Обработка чанков")):
                chunk = chunk.unsqueeze(0)  # Добавляем batch dimension

                # Запускаем модель на одном кусочке
                sources = self.model(chunk)[0]

                # Сохраняем каждый стем из этого кусочка
                for stem_idx, stem in enumerate(self.stem_names):
                    if stem_idx < sources.shape[0]:
                        separated_chunks[stem].append(sources[stem_idx].cpu())
                    else:
                        separated_chunks[stem].append(torch.zeros_like(chunk[0]))

                # Чистим память видеокарты, чтобы она не переполнялась
                if self.device == "cuda":
                    torch.cuda.empty_cache()

            # Склеиваем кусочки обратно в целые стемы
            results = {}
            for stem in self.stem_names:
                merged = self._merge_chunks(
                    separated_chunks[stem],
                    audio_tensor.shape[2],
                    hop_len,
                    overlap_len
                )
                results[stem] = merged

        # Сохраняем результаты в WAV файлы
        if output_dir is None:
            output_dir = OUTPUT_DIR / audio_path.stem

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        result_paths = {}
        for stem, audio_data in results.items():
            stem_path = output_dir / f"{audio_path.stem}_{stem}.wav"
            save_audio(stem_path, audio_data, self.sample_rate)
            result_paths[stem] = stem_path
            print(f"  ✓ {stem}: {stem_path.name}")

            # Сохраняем также в кэш, чтобы при повторном запросе не обрабатывать снова
            if use_cache:
                cache_path = get_cached_path(file_hash, stem)
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                save_audio(cache_path, audio_data, self.sample_rate)

        # Собираем статистику
        elapsed = time.time() - start_time
        duration_seconds = audio_tensor.shape[2] / self.sample_rate
        self.stats[audio_path.name] = {
            "duration_seconds": duration_seconds,
            "processing_seconds": elapsed,
            "rtf": elapsed / duration_seconds if duration_seconds > 0 else 0  # Real Time Factor
        }

        print(f"\nОбработка завершена за {elapsed:.2f} секунд")
        print(f"RTF (Real Time Factor): {self.stats[audio_path.name]['rtf']:.2f}")

        return result_paths

    def get_stats(self) -> Dict:
        """Возвращает статистику обработки"""
        return self.stats

    def clear_cache(self):
        """Удаляет все сохранённые кэши"""
        from config import CACHE_DIR
        import shutil
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
            CACHE_DIR.mkdir()
            print("Кэш очищен")


# ДЛЯ ПРЯМОГО ЗАПУСКА ИЗ КОМАНДНОЙ СТРОКИ

def main():
    parser = argparse.ArgumentParser(
        description="Разделение музыки на инструментальные дорожки (вокал, барабаны, бас, остальное)"
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Путь к аудиофайлу (MP3, WAV, FLAC, M4A, MP4)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Директория для сохранения результатов (по умолчанию: output/имя_файла)"
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Не использовать кэширование"
    )
    parser.add_argument(
        "--no-trim",
        action="store_true",
        help="Не обрезать тишину на границах"
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="Устройство для вычислений (cuda/cpu)"
    )

    args = parser.parse_args()

    # Проверяем, существует ли файл
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"\n❌ Ошибка: Файл '{args.input_file}' не найден!")
        print("\nПример использования:")
        print("  python separator.py test.mp3")
        print("  python separator.py music.wav -o my_output")
        print("  python separator.py song.flac --no-cache")
        print("  python separator.py audio.mp4")
        return

    # Проверяем формат файла
    allowed_extensions = {".mp3", ".wav", ".flac", ".m4a", ".mp4", ".mpeg", ".ogg"}
    if input_path.suffix.lower() not in allowed_extensions:
        print(f"\n❌ Ошибка: Неподдерживаемый формат '{input_path.suffix}'")
        print(f"Поддерживаемые форматы: {', '.join(allowed_extensions)}")
        print("\nПримечание: MP4 файлы обрабатываются как аудио (извлекается звуковая дорожка)")
        return

    print("\n" + "="*60)
    print("🎵 MUSIC SEPARATION SYSTEM - Hybrid Demucs")
    print("="*60)
    print(f"Входной файл: {input_path}")
    print(f"Размер файла: {input_path.stat().st_size / 1024 / 1024:.2f} MB")

    if input_path.suffix.lower() == ".mp4":
        print("⚠️  Файл MP4: будет использована звуковая дорожка")

    print("="*60 + "\n")

    # Создаём разделитель и запускаем обработку
    separator = MusicSeparator(device=args.device)

    try:
        result_paths = separator.separate(
            audio_path=input_path,
            output_dir=Path(args.output) if args.output else None,
            use_cache=not args.no_cache,
            trim_silence_before=not args.no_trim
        )

        print("\n" + "="*60)
        print("✅ РАЗДЕЛЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
        print("="*60)
        print("\n📁 Результаты сохранены:")
        for stem, path in result_paths.items():
            stem_name = {
                "vocals": "🎤 Вокал",
                "drums": "🥁 Барабаны",
                "bass": "🎸 Бас",
                "other": "🎹 Остальное"
            }.get(stem, stem)
            if path.exists():
                file_size = path.stat().st_size / 1024 / 1024
                print(f"  {stem_name}: {path}")
                print(f"    Размер: {file_size:.2f} MB")
            else:
                print(f"  {stem_name}: {path} (файл не создан)")

        # Показываем статистику
        stats = separator.get_stats()
        for filename, stat in stats.items():
            print(f"\n📊 Статистика обработки:")
            print(f"  Длительность: {stat['duration_seconds']:.1f} сек ({stat['duration_seconds']/60:.1f} мин)")
            print(f"  Время обработки: {stat['processing_seconds']:.2f} сек")
            print(f"  RTF: {stat['rtf']:.2f} (x{stat['rtf']:.1f} медленнее реального времени)")

        print("\n" + "="*60)

    except Exception as e:
        print(f"\n❌ Ошибка при разделении: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()