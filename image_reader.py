import tensorflow as tf


def image_scaling(image, label, min_val=0.5, max_val=1.5):
    # Randomly scales the image between 0.5 to 1.5 times the original size
    scale = tf.random_uniform([1], minval=min_val, maxval=max_val, dtype=tf.float32, seed=None)
    height_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(image)[0]), scale))
    width_new = tf.to_int32(tf.multiply(tf.to_float(tf.shape(image)[1]), scale))
    new_shape = tf.squeeze(tf.stack([height_new, width_new]), squeeze_dims=[1])
    image = tf.image.resize_images(image, new_shape)
    label = tf.image.resize_nearest_neighbor(tf.expand_dims(label, 0), new_shape)
    label = tf.squeeze(label, squeeze_dims=[0])
    return image, label


def image_mirroring(image, label):
    # Randomly mirrors the images
    distort_left_right_random = tf.random_uniform([1], 0, 1.0, dtype=tf.float32)[0]
    mirror = tf.less(tf.stack([1.0, distort_left_right_random, 1.0]), 0.5)
    mirror = tf.boolean_mask([0, 1, 2], mirror)
    image = tf.reverse(image, mirror)
    label = tf.reverse(label, mirror)
    return image, label


def random_crop_and_pad(image, label, crop_height, crop_width, ignore_label=255):
    # Randomly crop and pads the images
    label = tf.cast(label, dtype=tf.float32)
    label -= ignore_label
    combined = tf.concat(axis=2, values=[image, label])
    image_shape = tf.shape(image)
    max_height = tf.maximum(crop_height, image_shape[0])
    max_width = tf.maximum(crop_width, image_shape[1])
    combined_pad = tf.image.pad_to_bounding_box(combined, 0, 0, max_height, max_width)

    last_image_dim = tf.shape(image)[-1]
    last_label_dim = tf.shape(label)[-1]
    combined_crop = tf.random_crop(combined_pad, [crop_height, crop_width, 4])
    image_crop = combined_crop[:, :, :last_image_dim]
    label_crop = combined_crop[:, :, last_image_dim:]
    label_crop += ignore_label
    label_crop = tf.cast(label_crop, tf.int64)

    # Set static shape to know shape at compile time
    image_crop.set_shape((crop_height, crop_width, 3))
    label_crop.set_shape((crop_height, crop_width, 1))
    return image_crop, label_crop


def read_labeled_image_list(index_loc, data_dir, mask_dir):
    # Reads text file containing paths to images and ground truth masks
    f = open(index_loc, 'r')
    images = []
    masks = []
    i = 0
    for line in f:
        name = line.strip()
        images.append(data_dir + name + '.jpg')
        masks.append(mask_dir + name + '.png')
    return images, masks


def read_images_from_disk(input_queue, input_size, random_scale, random_mirror, ignore_label, image_mean):
    # Read one image and its corresponding mask with optional pre-processing
    img_contents = tf.read_file(input_queue[0])
    label_contents = tf.read_file(input_queue[1])

    image = tf.image.decode_jpeg(img_contents, channels=3)
    red, green, blue = tf.split(axis=2, num_or_size_splits=3, value=image)
    image = tf.cast(tf.concat(axis=2, values=[blue, green, red]), dtype=tf.float32)
    # Extract mean
    image -= image_mean

    mask = tf.image.decode_png(label_contents, channels=1)
    if input_size is not None:
        height, width = input_size
        if random_scale:
            image, mask = image_scaling(image, mask)
        if random_mirror:
            image, mask = image_mirroring(image, mask)

        image, mask = random_crop_and_pad(image, mask, height, width, ignore_label)

    return image, mask


class ImageReader(object):
    def __init__(self, index_loc, data_dir, mask_dir, input_size,
                 random_scale, random_mirror, ignore_label, img_mean, coord):
        self.index_loc = index_loc
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.input_size = input_size
        self.coord = coord

        self.image_list, self.label_list = read_labeled_image_list(self.index_loc, self.data_dir, self.mask_dir)
        self.images = tf.convert_to_tensor(self.image_list, dtype=tf.string)
        self.labels = tf.convert_to_tensor(self.label_list, dtype=tf.string)
        self.queue = tf.train.slice_input_producer([self.images, self.labels], shuffle=input_size is not None)
        self.image, self.label = read_images_from_disk(self.queue, self.input_size, random_scale, random_mirror,
                                                       ignore_label, img_mean)

    def dequeue(self, num_elements):
        # Pack Images and labels into a batch
        image_batch, label_batch = tf.train.batch([self.image, self.label], num_elements)
        return image_batch, label_batch
