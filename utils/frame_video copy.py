import cv2
import imageio
import os

def images_to_gif():
    images = []
    numberlist = os.listdir('output_frames/mask-rcnn')
    for i in range(len(numberlist)):
        if i== 641:
            break
        if i<=423:
            continue
        images.append(imageio.imread('output_frames/mask-rcnn/' + numberlist[i]))
    imageio.mimsave('mask-rcnn.gif', images, 'GIF', duration=0.00001)


def video_to_frames():
    path = "datasets/AICity_data/train/S03/c010/vdo.avi"
    videoCapture = cv2.VideoCapture()
    videoCapture.open(path)
    # 帧率
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    # 总帧数
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", int(fps), "frames=", int(frames))

    for i in range(int(frames)):
        ret, frame = videoCapture.read()
        cv2.imwrite("datasets/frames/frame_{:04d}.jpg".format(i), frame)



def images_to_video():
    fps = 15  # 帧率
    # num_frames = 500
    img_array = []
    img_width = 480
    img_height = 270
    # video_name = 'video_test.avi'
    # video = cv2.VideoWriter(video_name,
    #                         cv2.VideoWriter_fourcc('M', 'P', '4', 'S'), 20,
    #                         # cv2.VideoWriter_fourcc('H', '2', '6', '4'), 10,
    #                         (576, 324))
    video = cv2.VideoWriter('ssd.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (img_width, img_height))
    # out = cv2.VideoWriter('demo_people.avi', cv2.VideoWriter_fourcc(*'mp4v'), fps, (img_width, img_height))

    path = "output_frames/ssd/"
    for file_name in os.listdir(path):
        img = cv2.imread(path + file_name)
        if img is None:
            print(file_name + " is non-existent!")
            continue
        # img_array.append(img)
        video.write(img)

    # print(img_array)


    # for i in range(len(img_array)):
    #     out.write(img_array[i])
    # out.release()


    # dir_path = "output_frames"
    # for i in range(1000):
    #     frame_path = dir_path.joinpath('GMG','frame_{:04d}.png'.format(i + 1))
    #     image = cv2.imread(str(frame_path))
    #     print(image)
    # video

def main():
    # images_to_video()
    # images_to_gif()
    video_to_frames()


if __name__ == "__main__":
    main()