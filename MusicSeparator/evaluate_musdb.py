import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torchaudio
import soundfile as sf
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent))

from separator import MusicSeparator
from config import STEM_NAMES, SAMPLE_RATE, DEVICE

# Пробуем подключить библиотеку для графиков, если не встанет — работаем без неё
try:
    import matplotlib

    matplotlib.use('Agg')  # Отключаем всплывающее окно с графиком, сохраняем сразу в файл
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
    print("✅ Matplotlib готов")
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️ Установите matplotlib: pip install matplotlib")

# Пробуем подключить stempeg — он нужен, чтобы читать эталонные стемы из MUSDB18
try:
    import stempeg

    STEMPEG_AVAILABLE = True
except ImportError:
    STEMPEG_AVAILABLE = False
    print("⚠️ Установите stempeg: pip install stempeg")


class MUSDBEvaluator:
    def __init__(self, musdb_path: Path, sample_rate: int = 44100):
        self.musdb_path = Path(musdb_path)
        self.sample_rate = sample_rate
        self.tracks = []
        self._find_tracks()

    # Ищем все файлы с расширением .stem.mp4 в папке датасета
    def _find_tracks(self):
        mp4_files = list(self.musdb_path.rglob("*.stem.mp4"))
        for mp4_file in mp4_files:
            # Убираем .stem из имени, оставляем чистый название трека
            track_name = mp4_file.stem.replace(".stem", "")
            self.tracks.append({"name": track_name, "mixture": mp4_file})

        print(f"📁 Найдено треков: {len(self.tracks)}")
        if self.tracks:
            print(f"📋 Пример: {self.tracks[0]['name']}")

    # Загружаем аудиофайл и приводим его к нужной частоте
    def load_audio(self, file_path: Path) -> np.ndarray:
        try:
            waveform, sr = torchaudio.load(str(file_path))
            # Если частота не совпадает с нужной — пересчитываем
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
            audio = waveform.numpy()
        except:
            # Если torchaudio не справился, пробуем soundfile
            audio, sr = sf.read(str(file_path))

        # Если аудио стерео (два канала) — превращаем в моно, просто усредняя каналы
        if audio.ndim == 2:
            audio = np.mean(audio, axis=0)
        elif audio.ndim > 2:
            audio = audio.flatten()

        return audio.astype(np.float32)

    # Из MP4-файла MUSDB18 достаём эталонные стемы (правильные ответы)
    def load_stems_from_mp4(self, mp4_path: Path) -> Dict[str, np.ndarray]:
        if not STEMPEG_AVAILABLE:
            return {}

        try:
            stems, sr = stempeg.read_stems(str(mp4_path), sample_rate=self.sample_rate)
            if stems is None or len(stems) < 5:
                return {}

            result = {}
            # В MUSDB18 стемы лежат в определённом порядке: 0 — микс, 1 — барабаны, 2 — бас, 3 — остальное, 4 — вокал
            stem_indices = {"drums": 1, "bass": 2, "other": 3, "vocals": 4}

            for stem_name, idx in stem_indices.items():
                if idx < len(stems):
                    audio = stems[idx]
                    if audio.ndim == 2:
                        audio = np.mean(audio, axis=0)  # Переводим в моно
                    result[stem_name] = audio.astype(np.float32)
            return result
        except Exception as e:
            return {}

    # Считаем метрики качества: SDR, SIR, SAR
    # Это главная математическая часть — сравниваем то, что предсказала модель, с правильным ответом
    def compute_metrics(self, target: np.ndarray, estimated: np.ndarray,
                        all_stems: Dict[str, np.ndarray], stem_name: str) -> Dict:
        # Обрезаем до одинаковой длины, чтобы можно было сравнивать
        min_len = min(len(target), len(estimated))
        target = target[:min_len]
        estimated = estimated[:min_len]

        # Нормализуем громкость, чтобы честно сравнивать
        target = target / (np.sqrt(np.mean(target ** 2)) + 1e-8)
        estimated = estimated / (np.sqrt(np.mean(estimated ** 2)) + 1e-8)

        error = target - estimated  # Разница между правильным и предсказанным

        # Собираем помехи от других инструментов (интерференцию)
        interference = np.zeros_like(target)
        for other_stem, other_audio in all_stems.items():
            if other_stem != stem_name:
                other_len = min(len(other_audio), min_len)
                interference[:other_len] += other_audio[:other_len]

        # Тоже нормализуем помехи
        if np.sum(interference ** 2) > 0:
            interference = interference / (np.sqrt(np.mean(interference ** 2)) + 1e-8)
            interference = interference * (np.sqrt(np.mean(target ** 2)) / (np.sqrt(np.mean(interference ** 2)) + 1e-8))

        # Раскладываем ошибку на то, что вызвано помехами, и то, что вызвано артефактами
        if np.sum(interference ** 2) > 0:
            alpha = np.sum(error * interference) / (np.sum(interference ** 2) + 1e-8)
            interference_component = alpha * interference
        else:
            interference_component = 0

        artifact_component = error - interference_component

        # Считаем мощности сигналов
        target_power = np.sum(target ** 2)
        interference_power = np.sum(interference_component ** 2)
        artifact_power = np.sum(artifact_component ** 2)
        error_power = np.sum(error ** 2)

        # Формулы метрик в децибелах
        sdr = 10 * np.log10(target_power / (error_power + 1e-8))
        sir = 10 * np.log10(target_power / (interference_power + 1e-8)) if interference_power > 0 else 20
        sar = 10 * np.log10(target_power / (artifact_power + 1e-8)) if artifact_power > 0 else sdr

        # Ограничиваем значения, чтобы не вылезали за разумные пределы
        return {
            "SDR": float(np.clip(sdr, -10, 20)),
            "SIR": float(np.clip(sir, -10, 30)),
            "SAR": float(np.clip(sar, -10, 20))
        }

    # Оцениваем один трек: применяем модель и сравниваем с эталоном
    def evaluate_track(self, separator: MusicSeparator, track: Dict) -> Dict:
        track_name = track["name"]
        mixture_path = track["mixture"]

        print(f"\n🎵 {track_name[:45]}...")

        # Загружаем правильные ответы
        ground_truth = self.load_stems_from_mp4(mixture_path)
        if not ground_truth:
            return {}

        print(f"  ✓ Загружено {len(ground_truth)} эталонных стемов")

        # Запускаем разделение нашей моделью
        try:
            separated = separator.separate(mixture_path, output_dir=None, use_cache=True)
        except Exception as e:
            print(f"  ❌ Ошибка: {e}")
            return {}

        # Загружаем предсказанные стемы
        predicted = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            if stem in separated:
                predicted[stem] = self.load_audio(separated[stem])

        # Сравниваем и считаем метрики для каждого стема
        metrics = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            if stem in ground_truth and stem in predicted:
                m = self.compute_metrics(ground_truth[stem], predicted[stem], ground_truth, stem)
                metrics[stem] = m

                # Простая оценка качества на глаз
                quality = "✅ Отлично" if m["SDR"] > 6 else "👍 Хорошо" if m["SDR"] > 3 else "⚠️ Средне"
                print(f"  {stem:6}: SDR={m['SDR']:5.1f} dB | SIR={m['SIR']:5.1f} dB | SAR={m['SAR']:5.1f} dB {quality}")

        return metrics

    # Запускаем оценку на нескольких треках
    def evaluate_all(self, max_tracks: int = 5) -> pd.DataFrame:
        if not self.tracks:
            return pd.DataFrame()

        tracks_to_eval = self.tracks[:max_tracks]

        print("\n🚀 Загрузка модели Hybrid Demucs...")
        separator = MusicSeparator(device=DEVICE, sample_rate=self.sample_rate)

        print(f"\n📊 Оценка {len(tracks_to_eval)} треков\n")

        all_metrics = []
        for track in tqdm(tracks_to_eval, desc="Прогресс"):
            metrics = self.evaluate_track(separator, track)
            for stem, m in metrics.items():
                all_metrics.append({
                    "track": track["name"][:35],
                    "stem": stem,
                    "SDR": m["SDR"],
                    "SIR": m["SIR"],
                    "SAR": m["SAR"]
                })

        return pd.DataFrame(all_metrics) if all_metrics else pd.DataFrame()

    # Печатаем красивый отчёт и сохраняем график
    def generate_report(self, df: pd.DataFrame):
        if df.empty:
            print("❌ Нет данных")
            return

        print("\n" + "=" * 70)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ КАЧЕСТВА РАЗДЕЛЕНИЯ")
        print("=" * 70)
        print(f"Модель: Hybrid Demucs")
        print(f"Датасет: MUSDB18 ({df['track'].nunique()} треков)")
        print(f"Устройство: {DEVICE.upper()}")
        print("=" * 70)

        print("\n📊 СРЕДНИЕ МЕТРИКИ:")
        print("-" * 70)

        results = {}
        for stem in ["vocals", "drums", "bass", "other"]:
            stem_df = df[df["stem"] == stem]
            if len(stem_df) > 0:
                sdr_m, sdr_s = stem_df['SDR'].mean(), stem_df['SDR'].std()
                sir_m, sir_s = stem_df['SIR'].mean(), stem_df['SIR'].std()
                sar_m, sar_s = stem_df['SAR'].mean(), stem_df['SAR'].std()

                results[stem] = (sdr_m, sdr_s, sir_m, sir_s, sar_m, sar_s)

                # Звёздочки для наглядности
                rating = "⭐⭐⭐" if sdr_m > 6 else "⭐⭐" if sdr_m > 3 else "⭐"
                print(f"\n{stem.upper()}:")
                print(f"  SDR: {sdr_m:.2f} ± {sdr_s:.2f} dB {rating}")
                print(f"  SIR: {sir_m:.2f} ± {sir_s:.2f} dB")
                print(f"  SAR: {sar_m:.2f} ± {sar_s:.2f} dB")

        print("-" * 70)
        print(f"\n📈 ОБЩИЕ СРЕДНИЕ:")
        print(f"  SDR: {df['SDR'].mean():.2f} dB")
        print(f"  SIR: {df['SIR'].mean():.2f} dB")
        print(f"  SAR: {df['SAR'].mean():.2f} dB")

        # Сохраняем таблицу результатов
        df.to_csv("evaluation_results.csv", index=False)
        print(f"\n💾 Сохранено: evaluation_results.csv")

        # Рисуем график
        self.create_plot(df, results)

    # Рисуем столбчатый график с метриками
    def create_plot(self, df: pd.DataFrame, results: dict):
        if not MATPLOTLIB_AVAILABLE:
            print("⚠️ matplotlib не установлен. Установите: pip install matplotlib")
            return

        try:
            # Создаём три графика рядом для SDR, SIR, SAR
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            stems = ['vocals', 'drums', 'bass', 'other']
            metrics = ['SDR', 'SIR', 'SAR']
            colors = ['#2ecc71', '#e74c3c', '#3498db', '#9b59b6']
            titles = [
                'SDR (Signal-to-Distortion Ratio)',
                'SIR (Signal-to-Interference Ratio)',
                'SAR (Signal-to-Artifact Ratio)'
            ]

            for idx, (metric, title) in enumerate(zip(metrics, titles)):
                means = []
                stds = []
                labels = []
                color_list = []

                # Собираем средние значения и стандартные отклонения
                for i, stem in enumerate(stems):
                    if stem in results:
                        means.append(results[stem][idx * 2])
                        stds.append(results[stem][idx * 2 + 1])
                        labels.append(stem.upper())
                        color_list.append(colors[i])

                if len(means) == 0:
                    continue

                # Рисуем столбцы
                bars = axes[idx].bar(range(len(means)), means, color=color_list, alpha=0.7, edgecolor='black',
                                     linewidth=1.5)
                axes[idx].errorbar(range(len(means)), means, yerr=stds, fmt='none', color='black', capsize=5,
                                   capthick=2)
                axes[idx].set_xticks(range(len(means)))
                axes[idx].set_xticklabels(labels, fontsize=11)
                axes[idx].set_title(title, fontsize=12, fontweight='bold')
                axes[idx].set_ylabel('dB', fontsize=11)
                axes[idx].set_ylim(0, 22)
                axes[idx].grid(True, alpha=0.3, axis='y')

                # Добавляем линии-ориентиры: 5 dB — хорошо, 10 dB — отлично
                axes[idx].axhline(y=5, color='orange', linestyle='--', alpha=0.7, linewidth=1.5)
                axes[idx].axhline(y=10, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
                axes[idx].text(len(means) - 0.5, 5.5, 'Хорошо (5 dB)', fontsize=8, color='orange')
                axes[idx].text(len(means) - 0.5, 10.5, 'Отлично (10 dB)', fontsize=8, color='green')

                # Подписываем значения на столбцах
                for bar, val in zip(bars, means):
                    axes[idx].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                                   f'{val:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

            plt.suptitle(f'Оценка качества разделения Hybrid Demucs на MUSDB18\n'
                         f'({df["track"].nunique()} треков, {DEVICE.upper()})',
                         fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig("evaluation_plot.png", dpi=200, bbox_inches='tight', facecolor='white')
            print(f"📊 Сохранено: evaluation_plot.png")
            plt.close()

        except Exception as e:
            print(f"⚠️ Ошибка графика: {e}")
            print("Проверьте: pip install matplotlib")


def main():
    import argparse

    # Настройка параметров командной строки
    parser = argparse.ArgumentParser()
    parser.add_argument("--musdb-path", default="musdb18")  # Путь к папке с датасетом
    parser.add_argument("--max-tracks", type=int, default=5)  # Сколько треков обработать
    args = parser.parse_args()

    musdb_path = Path(args.musdb_path)
    if not musdb_path.exists():
        print(f"\n❌ MUSDB18 не найден: {musdb_path}")
        return

    print("\n" + "=" * 70)
    print("ОЦЕНКА КАЧЕСТВА РАЗДЕЛЕНИЯ MUSDB18")
    print("=" * 70)

    evaluator = MUSDBEvaluator(musdb_path)

    if not evaluator.tracks:
        print("\n❌ Файлы .stem.mp4 не найдены!")
        return

    max_tracks = min(args.max_tracks, len(evaluator.tracks))
    print(f"\nОценка на {max_tracks} треках...")

    df = evaluator.evaluate_all(max_tracks=max_tracks)

    if not df.empty:
        evaluator.generate_report(df)
        print("\n" + "=" * 70)
        print("✅ ГОТОВО!")
        print("=" * 70)
        print("\n📁 Результаты:")
        print("  📊 evaluation_plot.png - график метрик")
        print("  📄 evaluation_results.csv - детальные результаты")


if __name__ == "__main__":
    main()