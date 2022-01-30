import numpy as np
import os
import cv2

def clamp(color):
    return max(0, min(255, color))

def floyd_steinberg_dither(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = (image.shape[0], image.shape[1])
    
    for y in range(0, height-1):
        for x in range(1, width-1):
            old_p = image[y, x]
            new_p = np.round(old_p/255.0) * 255
            image[y, x] = new_p
            
            quant_error_p = old_p - new_p

            image[y, x+1] = clamp(image[y, x+1] + quant_error_p * 7 / 16.0)
            image[y+1, x-1] = clamp(image[y+1, x-1] + quant_error_p * 3 / 16.0)
            image[y+1, x] = clamp(image[y+1, x] + quant_error_p * 5 / 16.0)
            image[y+1, x+1] = clamp(image[y+1, x+1] + quant_error_p * 1 / 16.0)

    return image

def video_from_frames(output_name):
    os.system(f'ffmpeg -framerate 25 -i ./output/frame%3d.png -r 25 -y ./output/{output_name}.mp4')
    os.system('rm ./output/*.png')

def main(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width, height = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    fps = cap.get(cv2.CAP_PROP_FPS)

    current_frame = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            dithered_frame = floyd_steinberg_dither(frame)
            cv2.imwrite(os.path.join(os.path.dirname(__file__), '../output/frame{:03d}.png'.format(current_frame)), dithered_frame)
            current_frame += 1
            print(f'{current_frame}/{total}')
        else:
            break

    video_from_frames(os.path.splitext(os.path.basename(video_path))[0])

    cap.release()
    cv2.destroyAllWindows()

video_path = os.path.join(os.path.dirname(__file__), '../input/vid.mp4')
main(video_path)