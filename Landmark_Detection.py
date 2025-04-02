
import cv2
import mediapipe as mp
import os
import matplotlib.pyplot as plt

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, 
                                  max_num_faces=2, 
                                  refine_landmarks=True, 
                                  min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1, color=(0,255,0))

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image, image_rgb

def detect_and_draw_landmarks(image_path, save_path=None):
    original_image, rgb_image = load_image(image_path)
    if original_image is None:
        return
    
    results = face_mesh.process(rgb_image)
    
    if not results.multi_face_landmarks:
        print(f"No face landmarks detected in {image_path}")
        return
    
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=original_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1)
        )
        mp_drawing.draw_landmarks(
            image=original_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,0,255), thickness=1, circle_radius=1)
        )
    
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title('Facial Landmarks')
    plt.show()
    
    if save_path:
        cv2.imwrite(save_path, original_image)
        print(f"Saved annotated image to {save_path}")

def process_directory(input_dir, output_dir=None):
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            save_path = os.path.join(output_dir, filename) if output_dir else None
            detect_and_draw_landmarks(input_path, save_path)

image_path = 'path_to_image.jpg' 
detect_and_draw_landmarks(image_path, save_path='annotated_image.jpg')

