# ml-service

Yandex GPT, эмбеддинги, Unlimited-OCR.

## Разворот

Через compose в server-service (рекомендуется):

```bash
cd ../server-service
cp .env.example .env
task up
```

http://localhost:8001/health

Переменные Yandex берутся из `.env` server-service.

## Локально без Docker

```bash
cp .env.example .env
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8001
```

OCR на CPU — первый запрос долгий (скачивание модели `baidu/Unlimited-OCR`).

## Эндпоинты

```
GET  /health
POST /embed
POST /query/synthesize
POST /ocr/parse          multipart PDF
GET  /ocr/health
```

## Образ

```bash
docker build -t ml-service .
```

Память: до 16G на Mac с OCR.
