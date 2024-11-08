import ultralytics
from ultralytics import YOLO
import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt

def main():

    print(torch.cuda.is_available())
    # def load_images_and_labels(image_folder, label_folder):
    #     images = []
    #     labels = []
    #     for image_file in os.listdir(image_folder):
    #         if image_file.endswith('.jpg'):
    #             image_path = os.path.join(image_folder, image_file)
    #             label_path = os.path.join(label_folder, image_file.replace('.jpg', '.txt'))
    #             if os.path.exists(label_path):
    #                 images.append(image_path)
    #                 with open(label_path, 'r') as file:
    #                     label_data = file.read().strip()
    #                     labels.append(label_data)
    #     return images, labels

    image_folder = './BoneFractureYolo8/train/images'
    label_folder = './BoneFractureYolo8/train/labels'

    # label_folder = './data/train/labels'

    # train_images, train_labels = load_images_and_labels(image_folder, label_folder)

    # Set the model to YOLOv8
    model = YOLO('yolov8n.pt')

    # Set up the configuration for training
    data_config = """
    train: ./BoneFractureYolo8/train/images
    val: ./BoneFractureYolo8/valid/images
    nc: 6  # Number of classes
    names: ['class0', 'class1', 'class2', 'class3', 'class4', 'class5']  # Update with actual class names

    """

    # Save the configuration to a file
    with open('yolov8_config.yaml', 'w') as f:
        f.write(data_config)

    # Train the model
    results = model.train(
        data='yolov8_config.yaml',
        epochs=50,
        batch=32,
        imgsz=640,
        name='fracture_detector',
        lr0=0.01,
        device=0
    )

    # Save the trained model
    model.save('fracture_detector.pt')

    # Load the trained model
    model = YOLO('fracture_detector.pt')

    # Load a test image
    # Had issues so I had to get the EXACT path, you will need to alter this for your computer
    test_image_folder = 'C:/Users/blake/Documents/MVison/datasets/BoneFractureYolo8/test/images' 


    # test_image_path = os.path.join(test_image_folder, 'test_image.jpg')
    # image = cv2.imread(test_image_path)

    # # Predict using the model
    # results = model.predict(image)

    # Display results
    for image_file in os.listdir(test_image_folder):
        test_image_path = os.path.join(test_image_folder, image_file)
        image = cv2.imread(test_image_path)

        # Ensure the file is an image and not another type of file
        if image is not None:
            # Predict using the model
            results = model.predict(image)

            # Display results on the image
            for result in results:
                boxes = result.boxes  # Detection bounding boxes
                class_ids = boxes.cls  # Class IDs for each detected box

                for box, class_id in zip(boxes.xyxy, class_ids):
                    x1, y1, x2, y2 = map(int, box)  # Convert to integer coordinates
                    label_text = model.names[int(class_id)]  # Get class name
                    cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(image, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            # Display the result using matplotlib
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
        else:
            print(f"Unable to load image: {test_image_path}")


if __name__ == '__main__':
    main()