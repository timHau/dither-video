import numpy as np
import os
import sys
import getopt
import cv2
from numba import jit

@jit(nopython=True)
def clamp(color):
    return max(0, min(255, color))

@jit(nopython=True)
def floyd_steinberg_dither(image):
    height, width = (image.shape[0], image.shape[1])
    
    for y in range(0, height-1):
        for x in range(1, width-1):
            old_p = image[y, x]
            new_p = np.round(old_p/255.0) * 255
            image[y, x] = new_p
            
            quant_error_p = old_p - new_p

            image[y, x+1] = clamp(image[y, x+1] + quant_error_p * 0.4375) # 7 / 16.0
            image[y+1, x-1] = clamp(image[y+1, x-1] + quant_error_p * 0.1875) # 3 / 16.0
            image[y+1, x] = clamp(image[y+1, x] + quant_error_p * 0.3125) # 5 / 16.0
            image[y+1, x+1] = clamp(image[y+1, x+1] + quant_error_p * 0.0625) # 1 / 16.0

    return image

def video_from_frames(output_name):
    os.system(f'ffmpeg -i ./output/frame%3d.png -r 25 ./output/"{output_name}.mp4"')
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
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dithered_frame = floyd_steinberg_dither(frame)
            cv2.imwrite(os.path.join(os.path.dirname(__file__), '../output/frame{:03d}.png'.format(current_frame)), dithered_frame)
            current_frame += 1
            print(f'{current_frame}/{total}')
        else:
            break

    video_from_frames(os.path.splitext(os.path.basename(video_path))[0])

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:')
    except getopt.GetoptError:
        print('Usage: python main.py -i <video_name>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-i':
            if not arg.endswith('.mp4'):
                print('Only mp4 files are supported')
                sys.exit(2)
            video_path = os.path.join(os.path.dirname(__file__), f'../input/{arg}')
            main(video_path)
        else:
            print('Usage: python main.py -i <video_name>')
            sys.exit(2)