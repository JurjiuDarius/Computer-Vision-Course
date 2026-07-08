import cv2
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
logo_with_alpha = cv2.imread(os.path.join(script_dir, "images", "opencv_logo.png"), cv2.IMREAD_UNCHANGED)
logo_with_alpha = cv2.resize(logo_with_alpha, (200, 260))

logo_color = logo_with_alpha[:, :, :3]
logo_transparency = logo_with_alpha[:, :, 3]

_, mask_where_logo_visible = cv2.threshold(logo_transparency, 10, 255, cv2.THRESH_BINARY)
mask_where_logo_transparent = cv2.bitwise_not(mask_where_logo_visible)

webcam = cv2.VideoCapture(0)

while True:
    ret, frame = webcam.read()
    if not ret:
        break

    logo_height, logo_width = logo_color.shape[:2]
    frame_region_under_logo = frame[20:20+logo_height, 20:20+logo_width]

    background_showing_through = cv2.bitwise_and(frame_region_under_logo, frame_region_under_logo, mask=mask_where_logo_transparent)
    logo_pixels_only = cv2.bitwise_and(logo_color, logo_color, mask=mask_where_logo_visible)
    frame[20:20+logo_height, 20:20+logo_width] = cv2.add(background_showing_through, logo_pixels_only)

    cv2.imshow("Logo Overlay", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
