from PIL import Image
from numpy import asarray, copy

def get_gray(img):
    
    
    # sample.png is the name of the image
    # file and assuming that it is uploaded
    # in the current directory or we need
    # to give the path
    # image = Image.open(img_loc)
    
    # # summarize some details about the image
    # print(image.format)
    # print(image.size)
    # print(image.mode)

    # img = asarray(image)
  
    # # <class 'numpy.ndarray'>
    # print(type(img))
    
    # #  shape
    # print(img.shape)



    img = copy(img)

    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    image = Image.fromarray(gray)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    #print(image.tobytes())
    return image




    # print(str(img_loc))
    # loc = str(img_loc)
    # new_img_loc = loc[:loc.find('.jpg')] + '_gray.jpg'
    # image.save("media/" + new_img_loc)
    # return new_img_loc