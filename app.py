from flask import Flask, render_template,Blueprint, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import tensorflow as tf
import mediapipe as mp
from scipy.interpolate import splprep, splev

app = Flask(__name__)

# Route for the main page
@app.route('/')
def index():
    return render_template('index.html')

# Route for the upload image page
@app.route('/upload_image')
def upload_image():
    return render_template('upload_image.html')

@app.route('/check_hair_loss')
def check_hair_loss():
    return render_template('heatmap_image.html')

@app.route('/hair_cvg_estimation')
def hair_cvg_estimation():
    return render_template('cvg.html')

model = YOLO(r'best.pt')  # Load your YOLO model

# Function to detect and process masks
def detect_and_overlay(image_data):
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    # Image dimensions
    image_width, image_height = image.shape[1], image.shape[0]

    # Run predictions using the YOLO model
    results = model.predict(
        source=image,
        conf=0.3,
        iou=0.3,
        show_boxes=False,
        imgsz=1024,
        retina_masks=True,
        agnostic_nms=False,
        task='segment'
    )

    mask_coordinates = []
    total_hair_width_percentage = 0
    count = 0

    # Process results
    for result in results:
        if len(result.masks) > 0:
            for idx, mask in enumerate(result.masks.data):
                binary_mask = (mask.cpu().numpy() > 0.5).astype(np.uint8)

                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    min_area_rect = cv2.minAreaRect(largest_contour)
                    box = cv2.boxPoints(min_area_rect)
                    box = np.array(box, dtype=np.int32)

                    box_sorted = box[box[:, 1].argsort()]
                    top_points = box_sorted[:2]
                    bottom_points = box_sorted[2:]

                    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
                    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]

                    width = min(min_area_rect[1])
                    width_percentage = (width / image_width) * 100
                    total_hair_width_percentage += width_percentage
                    count += 1

                    min_height = width * 4
                    max_height = width * 5

                    height = max(min_area_rect[1])
                    within_range = min_height <= height <= max_height

                    topmost_point_percentage = (
                        float(top_right[0] / image_width) * 100,
                        float(top_right[1] / image_height) * 100
                    )
                    bottommost_point_percentage = (
                        float(bottom_right[0] / image_width) * 100,
                        float(bottom_right[1] / image_height) * 100
                    )

                    mask_coordinates.append({
                        "p1": {"x": topmost_point_percentage[0], "y": topmost_point_percentage[1]},
                        "p2": {"x": bottommost_point_percentage[0], "y": bottommost_point_percentage[1]},
                        "width_percentage": f"{width_percentage:.2f}",
                        "within_range": within_range
                    })

    average_hair_width_percentage = (total_hair_width_percentage / count) if count > 0 else 0
    return mask_coordinates, average_hair_width_percentage

@app.route("/detect", methods=["POST"])
def classify():
    try:
        buf = request.files["image_file"]
        file_content = buf.read()
        mask_coordinates, avg_width_percentage = detect_and_overlay(file_content)

        response = {
            "mask_coordinates": mask_coordinates,
            "average_width_percentage": f"{avg_width_percentage:.2f}"
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Detection with overlay image generation
def detect_and_overlay_image(image_data):
    image_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    image_width, image_height = image.shape[1], image.shape[0]
    overlay_image = image.copy()

    results = model.predict(
        source=image,
        conf=0.3,
        iou=0.3,
        show_boxes=False,
        imgsz=640,
        retina_masks=True,
        agnostic_nms=False,
        task='segment',
        max_det=4500,
        half=True,
        device='0',
        augment=False,
        verbose=False
    )

    mask_coordinates = []
    total_hair_width_percentage = 0
    count = 0

    for result in results:
        if len(result.masks) > 0:
            for idx, mask in enumerate(result.masks.data):
                binary_mask = (mask.cpu().numpy() > 0.5).astype(np.uint8)
                contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    min_area_rect = cv2.minAreaRect(largest_contour)
                    box = cv2.boxPoints(min_area_rect)
                    box = np.array(box, dtype=np.int32)

                    box_sorted = box[box[:, 1].argsort()]
                    top_points = box_sorted[:2]
                    bottom_points = box_sorted[2:]

                    top_left, top_right = top_points[np.argsort(top_points[:, 0])]
                    bottom_left, bottom_right = bottom_points[np.argsort(bottom_points[:, 0])]

                    cv2.polylines(overlay_image, [box], isClosed=True, color=(0, 255, 0), thickness=2)

                    width = min(min_area_rect[1])
                    width_percentage = (width / image_width) * 100
                    total_hair_width_percentage += width_percentage
                    count += 1

                    min_height = width * 4
                    max_height = width * 5

                    height = max(min_area_rect[1])
                    within_range = min_height <= height <= max_height

                    topmost_point_percentage = (
                        float(top_right[0] / image_width) * 100,
                        float(top_right[1] / image_height) * 100
                    )
                    bottommost_point_percentage = (
                        float(bottom_right[0] / image_width) * 100,
                        float(bottom_right[1] / image_height) * 100
                    )

                    mask_coordinates.append({
                        "p1": {"x": topmost_point_percentage[0], "y": topmost_point_percentage[1]},
                        "p2": {"x": bottommost_point_percentage[0], "y": bottommost_point_percentage[1]},
                        "width_percentage": f"{width_percentage:.2f}",
                        "within_range": within_range
                    })

    average_hair_width_percentage = (total_hair_width_percentage / count) if count > 0 else 0
    _, img_encoded = cv2.imencode('.png', overlay_image)
    img_bytes = img_encoded.tobytes()

    return img_bytes, mask_coordinates, average_hair_width_percentage

@app.route("/gen-image", methods=["POST"])
def classify_image():
    try:
        buf = request.files["image_file"]
        file_content = buf.read()
        overlay_img_bytes, mask_coordinates, avg_width_percentage = detect_and_overlay_image(file_content)

        img_base64 = base64.b64encode(overlay_img_bytes).decode('utf-8')

        response = {
            "mask_coordinates": mask_coordinates,
            "average_width_percentage": f"{avg_width_percentage:.2f}",
            "image": img_base64
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
def extract_points_from_image(image_path, model_path="selfie_multiclass_256x256.tflite"):
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, refine_landmarks=True
        )

        hair_indices = [54, 103, 67, 109, 10, 338, 297, 332, 284]

        frame = cv2.imread(image_path)
        if frame is None:
            raise ValueError("Error: Unable to load the image.")

        points_dict = {"hair": [], "forehead": []}

        frame_resized = cv2.resize(
            frame, (input_details[0]['shape'][2], input_details[0]['shape'][1])
        )
        input_data = np.expand_dims(frame_resized, axis=0).astype(np.float32) / 255.0

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        segmentation_map = interpreter.get_tensor(output_details[0]['index'])[0]
        segmentation_map_resized = cv2.resize(segmentation_map, (frame.shape[1], frame.shape[0]))

        hair_mask = (segmentation_map_resized[:, :, 1] > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(hair_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
            points_dict["hair"] = [tuple(point[0].tolist()) for point in approx_contour]

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for idx in hair_indices:
                landmark = results.multi_face_landmarks[0].landmark[idx]
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                points_dict["forehead"].append((x, y))

        return points_dict, hair_mask

    except Exception as e:
        print(f"Error in processing image: {str(e)}")
        return None, None

def smooth_curve(points, smoothing_factor=2):
    if len(points) < 3:
        return np.array(points)
    points = np.array(points)
    x, y = points[:, 0], points[:, 1]
    tck, _ = splprep([x, y], s=smoothing_factor)
    x_smooth, y_smooth = splev(np.linspace(0, 1, 100), tck)
    return np.stack((x_smooth, y_smooth), axis=-1)

def calculate_density_areas_from_output_heatmap(output_heatmap_path, binary_mask):
    try:
        heatmap = cv2.imread(output_heatmap_path)
        if heatmap is None:
            raise ValueError("Error: Unable to load the output heatmap image.")

        hsv_heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2HSV)
        blue_range = ((100, 150, 50), (140, 255, 255))
        orange_range = ((5, 150, 50), (20, 255, 255))

        blue_mask = cv2.inRange(hsv_heatmap, blue_range[0], blue_range[1])
        orange_mask = cv2.inRange(hsv_heatmap, orange_range[0], orange_range[1])
        medium_mask = cv2.bitwise_not(cv2.bitwise_or(blue_mask, orange_mask))

        total_pixels = np.count_nonzero(binary_mask)
        high_density_pixels = cv2.countNonZero(cv2.bitwise_and(blue_mask, blue_mask, mask=binary_mask))
        low_density_pixels = cv2.countNonZero(cv2.bitwise_and(orange_mask, orange_mask, mask=binary_mask))
        medium_density_pixels = cv2.countNonZero(cv2.bitwise_and(medium_mask, medium_mask, mask=binary_mask))

        high_density_percentage = (high_density_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        medium_density_percentage = (medium_density_pixels / total_pixels) * 100 if total_pixels > 0 else 0
        low_density_percentage = (low_density_pixels / total_pixels) * 100 if total_pixels > 0 else 0

        return high_density_percentage, medium_density_percentage, low_density_percentage

    except Exception as e:
        print(f"Error in calculating density areas from output heatmap: {str(e)}")
        return 0, 0, 0

def calculate_hair_coverage(hair_mask, output_hair_coverage="hair_coverage.jpg"):
    total_pixels = hair_mask.size
    hair_pixels = np.count_nonzero(hair_mask)
    coverage_percentage = (hair_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    coverage_image = (hair_mask * 255).astype(np.uint8)
    cv2.imwrite(output_hair_coverage, coverage_image)

    return coverage_percentage

def create_heatmaps(image_path, points_dict, hair_mask,
                    output_whole="whole_image_heatmap.jpg",
                    output_points="points_image.jpg",
                    output_masked="output_heatmap.jpg"):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Error: Unable to load the image.")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
        whole_heatmap = cv2.applyColorMap(blurred_image, cv2.COLORMAP_JET)
        cv2.imwrite(output_whole, whole_heatmap)

        points_image = image.copy()
        if points_dict and points_dict["hair"]:
            for point in points_dict["hair"]:
                cv2.circle(points_image, point, 5, (0, 255, 0), -1)
        if points_dict and points_dict["forehead"]:
            for point in points_dict["forehead"]:
                cv2.circle(points_image, point, 5, (0, 0, 255), -1)
        cv2.imwrite(output_points, points_image)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if points_dict and points_dict["hair"]:
            smooth_points = smooth_curve(points_dict["hair"])
            cv2.fillPoly(mask, [np.array(smooth_points, dtype=np.int32)], 255)

        mask_blurred = cv2.GaussianBlur(mask, (15, 15), 0)
        heatmap_masked = cv2.bitwise_and(whole_heatmap, whole_heatmap, mask=mask_blurred)
        blended_heatmap = cv2.addWeighted(image, 0.1, heatmap_masked, 0.9, 0)
        cv2.imwrite(output_masked, blended_heatmap)

    except Exception as e:
        print(f"Error in creating heatmaps: {str(e)}")

@app.route("/heatmap-coverage", methods=["POST"])
def heatmap_coverage():
    try:
        # Get the uploaded file and model path from the form
        image_file = request.files["image_file"]
        model_path = request.form.get("model_path", "selfie_multiclass_256x256.tflite")

        # Save the uploaded image to a temporary location
        image_path = "uploaded_image.jpg"
        with open(image_path, "wb") as f:
            f.write(image_file.read())

        # Process the image with the model path
        points_dict, hair_mask = extract_points_from_image(image_path, model_path)

        if points_dict and hair_mask is not None:
            # Create the heatmaps
            create_heatmaps(image_path, points_dict, hair_mask)

            # Calculate density and hair coverage
            output_heatmap_path = "output_heatmap.jpg"
            high_density, medium_density, low_density = calculate_density_areas_from_output_heatmap(output_heatmap_path, hair_mask)
            hair_coverage = calculate_hair_coverage(hair_mask)

            # Prepare the response with base64-encoded images
            def image_to_base64(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")

            response = {
                # "hair_coverage": hair_coverage,
                "density_areas": {
                    "high": high_density,
                    "medium": medium_density,
                    "low": low_density
                },
                "output_images": {
                    "points_image": image_to_base64("points_image.jpg"),
                    "output_heatmap": image_to_base64("output_heatmap.jpg"),
                    # "hair_coverage_mask": image_to_base64("hair_coverage.jpg")
                }
            }

            return jsonify(response)

        return jsonify({"error": "Failed to process the image."})

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"})
    
    
@app.route("/hair-coverage", methods=["POST"])
def hair_coverage():
    try:
        # Ensure the file exists in the request
        if "image_file" not in request.files:
            return jsonify({"error": "No file part in the request."}), 400

        # Get the uploaded file and model path
        image_file = request.files["image_file"]
        if image_file.filename == "":
            return jsonify({"error": "No selected file."}), 400

        model_path = request.form.get("model_path", "selfie_multiclass_256x256.tflite")

        # Save the uploaded image temporarily
        image_path = "uploaded_image.jpg"
        image_file.save(image_path)

        # Process the image with the model
        points_dict, hair_mask = extract_points_from_image(image_path, model_path)

        if points_dict and hair_mask is not None:
            # Calculate hair coverage percentage
            def calculate_hair_coverage(hair_mask, output_hair_coverage="hair_coverage.jpg"):
                total_pixels = hair_mask.size
                hair_pixels = np.count_nonzero(hair_mask)
                coverage_percentage = (hair_pixels / total_pixels) * 100 if total_pixels > 0 else 0

                # Save the hair coverage mask as an image
                coverage_image = (hair_mask * 255).astype(np.uint8)
                cv2.imwrite(output_hair_coverage, coverage_image)

                return coverage_percentage

            hair_coverage = calculate_hair_coverage(hair_mask)

            # Convert images to base64 for frontend display
            def image_to_base64(image_path):
                with open(image_path, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode("utf-8")

            # Prepare response with hair coverage and output image
            response = {
                "hair_coverage": hair_coverage,
                "output_images": {
                    "hair_coverage_mask": image_to_base64("hair_coverage.jpg")
                }
            }

            return jsonify(response)

        return jsonify({"error": "Failed to process the image."}), 500

    except Exception as e:
        # Handle unexpected errors
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)