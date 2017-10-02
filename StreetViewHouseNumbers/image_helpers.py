import cv2
import numpy as np

# resizes to image_size/image_size while keeping aspect ratio the same.  pads on right/bottom as appropriate 
def read_image(file_path, image_size):
    img = cv2.imread(file_path, cv2.IMREAD_COLOR) #cv2.IMREAD_GRAYSCALE
    if (img.shape[0] >= img.shape[1]): # height is greater than width
       resizeto = (image_size, int (round (image_size * (float (img.shape[1])  / img.shape[0]))));
    else:
       resizeto = (int (round (image_size * (float (img.shape[0])  / img.shape[1]))), image_size);
    
    img2 = cv2.resize(img, (resizeto[1], resizeto[0]), interpolation=cv2.INTER_CUBIC)
    img3 = cv2.copyMakeBorder(img2, 0, image_size - img2.shape[0], 0, image_size - img2.shape[1], cv2.BORDER_CONSTANT, 0)

    return img3[:,:,::-1]  # turn into rgb forma

def prep_data(image_paths, image_size, num_channels, pixel_depth):
    count = len(image_paths)
    data = np.ndarray((count, image_size, image_size, num_channels), dtype=np.float32)

    for i, image_file in enumerate(image_paths):
        image = read_image(image_file, image_size);
        image_data = np.array (image, dtype=np.float32);
        image_data[:,:,0] = (image_data[:,:,0].astype(float) - pixel_depth / 2) / pixel_depth
        image_data[:,:,1] = (image_data[:,:,1].astype(float) - pixel_depth / 2) / pixel_depth
        image_data[:,:,2] = (image_data[:,:,2].astype(float) - pixel_depth / 2) / pixel_depth
        
        data[i] = image_data; # image_data.T
        if i%250 == 0: print('Processed {} of {}'.format(i, count))    
    return data



