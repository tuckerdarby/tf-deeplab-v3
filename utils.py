from PIL import Image
import numpy as np
import tensorflow as tf

# colour map
label_colours = [(0,0,0)
                # 0=background
                ,(128,0,0),(0,128,0),(128,128,0),(0,0,128),(128,0,128)
                # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                ,(0,128,128),(128,128,128),(64,0,0),(192,0,0),(64,128,0)
                # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                ,(192,128,0),(64,0,128),(192,0,128),(64,128,128),(192,128,128)
                # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
                ,(0,64,0),(128,64,0),(0,192,0),(128,192,0),(0,64,128)]
                # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor


def decode_labels(mask, num_images=1, num_classes=21):
    # Decode batch of segmentation masks
    batch_size, height, width, channels = mask.shape
    assert(batch_size >= num_images),\
        'Batch size %d should be greater or equal than number of images to save %d' % (batch_size, num_images)
    outputs = np.zeros((num_images, height, width, 3), dtype=np.uint8)
    for i in range(num_images):
        image = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = image.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_,j_] = label_colours[k]
        outputs[i] = np.array(image)
    return outputs


def prepare_labels(input_batch, new_size, num_classes, one_hot=True):
    # Resize masks and perform one-hot encoding
    with tf.name_scope('label_encode'):
        input_batch = tf.image.resize_nearest_neighbor(input_batch, new_size)
        input_batch = tf.squeeze(input_batch, squeeze_dims=[3]) # reduce the channel dimension
        if one_hot:
            input_batch = tf.one_hot(input_batch, depth=num_classes)
    return input_batch


def inv_preprocess(images, num_images, image_mean):
    # Inverse preprocessing of the batch of images
    # Add the mean vector and convert from BGR to RGB
    batch_size, height, width, channels = images.shape
    assert(batch_size >= num_images),\
        'Batch size %d should be greater or equal than number of images to save %d' % (batch_size, num_images)
    outputs = np.zeros((num_images, height, width, channels), dtype=np.uint8)
    for i in range(num_images):
        outputs[i] = (images[i] + image_mean)[:, :, ::-1].astype(np.uint8)
    return outputs
