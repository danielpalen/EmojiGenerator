
import glob
import imageio


def create_gif(glob_img_path, gif_path):
    """
    Create a gif images.
    :param glob_img_path: the glob path string to collect images
    :param gif_path: the name of the gif
    :return: true, if gif has been created
    """
    print(f'\nCREATE GIF FILE:')
    with imageio.get_writer(gif_path, mode='I') as writer:
        filenames = glob.glob(glob_img_path)
        filenames = sorted(filenames)
        last = -1

        # Only use approx. 20 images
        step_size = len(filenames) // 20

        for i in range(0, len(filenames), step_size):
            filename = filenames[i]
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    print(f'- Gif saved at {gif_path}')


if __name__ == '__main__':
    create_gif(f'output/images/image*.png', f'../output/emojigan.gif')
