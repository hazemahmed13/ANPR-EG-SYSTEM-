from ultralytics import YOLO

def main():
    # Load a pretrained YOLO model
    model = YOLO("yolo11n.pt")

    # Train the model
    train_results = model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        device=0
    )

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # مهم لو بتستخدم Windows
    main()
