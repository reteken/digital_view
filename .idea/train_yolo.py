from ultralytics import YOLO

# Загружаем предобученную модель YOLOv8 (выберите подходящий размер: n, s, m, l, x)
model = YOLO('yolov8n.pt')

# Запускаем обучение модели
model.train(
    data=r'C:\Users\alexs\Desktop\python_yolo_light\model\model\dataset\data.yaml',   # Путь к конфигурационному файлу
    epochs=100,           # Количество эпох
    batch=16,             # Размер батча
    imgsz=640,             # Размер изображений (по умолчанию 640)
    device='cpu'
)
