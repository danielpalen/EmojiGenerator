import PIL
import imageio
from pandas import np

from EmojiReader import EmojiReader
from utilities import constants
from PIL import Image
import scipy.misc

def colorChanger(x):

    if(x[3]<20):
        return [255,255,255,0]
    if(np.dot(x,[ 0.2126,0.7152,0.0722,0])>250):
        return [255,255,255,0]
    else:
        if((np.dot(x,[ 0.2126,0.7152,0.0722,0]))>150):
            return [180,180,180,0]
        else:
           return [90,90,90,0]

def toJpg(x):
    if (x[3] < 20):
        return [255, 255, 255, 0]
    else:
        return x

def preProcessing(image,path,file,where):
    image=np.repeat(np.repeat(image,4,axis=0),4,axis=1)
    imageorg=np.copy(image)
    for row in image:
        for line in row:
            line[:] = colorChanger(line)
    for row in imageorg:
        for line in row:
            line[:] = toJpg(line)

    imageio.imwrite(path+where+'\\'+file.replace('png','jpg'), np.concatenate((image, imageorg), axis=1)[:,:,0:3])



def quartering(image, path, file,which):
    if(which=='h'):
        imageio.imwrite(path+which+'1'+'\\'+file,image[:33,:33])
        imageio.imwrite(path+which+'2'+'\\'+file,image[:33,33:])
        imageio.imwrite(path+which+'3'+'\\'+file,image[33:,:33])
        imageio.imwrite(path+which+'4'+'\\'+file,image[33:,33:])
    else:
        imageio.imwrite(path + which + '1' + '\\' + file, image[:37, :33])
        imageio.imwrite(path + which + '2' + '\\' + file, image[:37, 33:])
        imageio.imwrite(path + which + '3' + '\\' + file, image[37:, :33])
        imageio.imwrite(path + which + '4' + '\\' + file, image[37:, 33:])

#reader = EmojiReader(databases=[f'apple'], emoji_names=constants.FACE_SMILING_EMOJIS)

#reader.read_images_from_sheet(pixel=64, debugging=False, png_format='RGB')


#reader.apply_preprocessing()
#print(reader.images_as_np.shape)

#imagenp=reader.images_as_np[0]
#original=reader.images_as_np[0];
#new =original







