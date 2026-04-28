import shutil
import tempfile
import zipfile
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE_MB, MAX_DURATION_SECONDS, OUTPUT_DIR
from separator import MusicSeparator
from utils import get_audio_duration

# Здесь будем хранить загруженную модель, чтобы использовать её в разных местах
separator = None


# Эта штука запускается до старта сервера и закрывается после остановки
# Нужна, чтобы загрузить модель один раз, а не при каждом обращении
@asynccontextmanager
async def lifespan(app: FastAPI):
    global separator
    print("=" * 60)
    print("Загрузка модели Hybrid Demucs...")
    print("=" * 60)
    separator = MusicSeparator()  # Создаём объект с моделью внутри
    print("=" * 60)
    print("Сервер готов к работе!")
    print(f"Доступ по адресу: http://localhost:8000")
    print("=" * 60)
    yield  # Здесь сервер работает и обрабатывает запросы
    # Когда сервер выключается, попадаем сюда
    print("Завершение работы сервера...")


# Создаём само приложение с названием и описанием
app = FastAPI(
    title="Music Separation API",
    description="API для разделения музыки на инструментальные дорожки (вокал, барабаны, бас, остальное)",
    version="1.0.0",
    lifespan=lifespan
)

# Разрешаем другим сайтам обращаться к нашему серверу
# Без этого веб-страница не смогла бы отправлять файлы
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Папка со статикой — там лежит наша веб-страница (index.html)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)  # Создаём папку, если её нет
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    # Когда пользователь заходит на главную страницу, отдаём ему нашу HTML-ку
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    else:
        # Если файла нет — сообщаем об ошибке и показываем доступные адреса
        return JSONResponse(content={
            "message": "Веб-интерфейс не найден. Создайте файл static/index.html",
            "endpoints": {
                "/separate": "POST - загрузить аудиофайл для разделения",
                "/health": "GET - проверка статуса сервера",
                "/stats": "GET - статистика обработки"
            }
        })


@app.get("/health")
async def health():
    # Простая проверка: работает ли сервер и загружена ли модель
    return {
        "status": "healthy" if separator else "loading",  # healthy или loading
        "model_loaded": separator is not None,
        "device": separator.device if separator else None,  # cpu или cuda
        "sample_rate": 44100
    }


@app.get("/stats")
async def get_stats():
    # Отдаём статистику по обработанным файлам (время обработки и т.д.)
    if separator and separator.get_stats():
        return JSONResponse(content=separator.get_stats())
    return JSONResponse(content={"message": "Нет данных"})


@app.get("/test")
async def test():
    # Простой тестовый адрес, чтобы проверить, что сервер вообще запустился
    return {
        "status": "ok",
        "message": "API работает",
        "device": separator.device if separator else None,
        "output_dir": str(OUTPUT_DIR)
    }


@app.post("/separate")
async def separate_audio(
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = None
):
    # Это главная функция — сюда прилетает аудиофайл от пользователя

    # Сначала убеждаемся, что модель загружена
    if separator is None:
        raise HTTPException(status_code=503, detail="Модель не загружена. Подождите немного.")

    # Проверяем расширение файла (mp3, wav и т.д.)
    file_extension = Path(file.filename).suffix.lower()
    if file_extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат. Поддерживаются: MP3, WAV, FLAC, M4A"
        )

    # Сохраняем загруженный файл во временную папку
    # delete=False означает, что файл не удалится сам при закрытии
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)  # Копируем содержимое
        temp_path = Path(tmp_file.name)  # Запоминаем путь к временному файлу

    try:
        # Проверяем размер файла (в мегабайтах)
        file_size_mb = temp_path.stat().st_size / 1024 / 1024
        if file_size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"Файл превышает максимальный размер ({MAX_FILE_SIZE_MB} MB). Ваш файл: {file_size_mb:.1f} MB"
            )

        # Проверяем длительность трека
        duration = get_audio_duration(temp_path)
        if duration > MAX_DURATION_SECONDS:
            minutes = duration / 60
            raise HTTPException(
                status_code=400,
                detail=f"Файл слишком длинный. Максимальная длительность: {MAX_DURATION_SECONDS // 60} минут. Ваш файл: {minutes:.1f} минут"
            )

        # Отсекаем совсем короткие или пустые файлы
        if duration < 1.0:
            raise HTTPException(
                status_code=400,
                detail="Файл слишком короткий (менее 1 секунды)"
            )

        # Всё проверено, можно начинать разделение
        print(f"\n{'=' * 60}")
        print(f"Обработка файла: {file.filename}")
        print(f"Длительность: {duration:.1f} секунд ({duration / 60:.2f} мин)")
        print(f"Размер: {file_size_mb:.2f} MB")
        print(f"{'=' * 60}\n")

        # Запускаем основную логику — модель разделяет трек на стемы
        result_paths = separator.separate(
            temp_path,
            output_dir=OUTPUT_DIR / Path(file.filename).stem,
            use_cache=True,  # Включаем кэш, чтобы повторно не обрабатывать тот же файл
            trim_silence_before=True  # Обрезаем тишину до обработки
        )

        # Упаковываем полученные файлы в ZIP-архив
        zip_filename = f"{Path(file.filename).stem}_stems.zip"
        zip_path = OUTPUT_DIR / zip_filename

        with zipfile.ZipFile(zip_path, 'w') as zipf:
            for stem, stem_path in result_paths.items():
                # Даём файлам понятные имена: например, песня_vocals.wav
                zipf.write(stem_path, f"{Path(file.filename).stem}_{stem}.wav")

        print(f"\nZIP архив создан: {zip_path}")
        print(f"Размер ZIP: {zip_path.stat().st_size / 1024 / 1024:.2f} MB\n")

        # Достаём статистику обработки (сколько времени заняло)
        processing_stats = separator.get_stats()
        if file.filename in processing_stats:
            print(f"Время обработки: {processing_stats[file.filename]['processing_seconds']:.1f} сек")
            print(f"RTF: {processing_stats[file.filename]['rtf']:.2f}")

        # Вспомогательная функция, которая удаляет временные файлы
        # Нужна, чтобы на диске не копился мусор
        def cleanup_files():
            try:
                temp_path.unlink(missing_ok=True)  # Удаляем временный загруженный файл
                # Ждём 5 секунд, чтобы ZIP успел улететь пользователю
                import time
                time.sleep(5)
                zip_path.unlink(missing_ok=True)  # Удаляем ZIP-архив
                for path in result_paths.values():
                    path.unlink(missing_ok=True)  # Удаляем отдельные стемы
                print(f"Временные файлы для {file.filename} удалены")
            except Exception as e:
                print(f"Ошибка при удалении файлов: {e}")

        # Запускаем очистку в фоновом режиме, чтобы пользователю не ждать
        if background_tasks:
            background_tasks.add_task(cleanup_files)
        else:
            # Если фоновые задачи не поддерживаются, используем простой таймер
            import threading
            threading.Timer(10.0, cleanup_files).start()

        # Отдаём пользователю ZIP-архив со всеми стемами
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename=zip_filename
        )

    except HTTPException:
        # Если ошибка связана с пользовательским вводом, просто удаляем временный файл
        temp_path.unlink(missing_ok=True)
        raise
    except Exception as e:
        # Любая другая ошибка — тоже чистим за собой и сообщаем пользователю
        temp_path.unlink(missing_ok=True)
        import traceback
        traceback.print_exc()  # Печатаем подробности в консоль для отладки
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


# Запускаем сервер, если файл запущен напрямую (а не импортирован как библиотека)
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",  # Слушаем все сетевые интерфейсы (можно заходить с других устройств)
        port=8000,  # Порт, на котором работает сервер
        reload=False,  # Не перезагружаемся при изменении кода (чтобы не грузить модель заново)
        workers=1  # Один рабочий процесс (модель тяжёлая, больше не нужно)
    )