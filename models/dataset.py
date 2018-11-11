import gzip, struct, numpy
import util

train_data_file = util.DATA_PATH+'train-images-idx3-ubyte.gz'
train_labels_file = util.DATA_PATH+'train-labels-idx1-ubyte.gz'
test_data_file = util.DATA_PATH+'t10k-images-idx3-ubyte.gz'
test_labels_file = util.DATA_PATH+'t10k-labels-idx1-ubyte.gz'

class Dataset(object):
    def __init__(self):
        self.train_data = self.extract_data(train_data_file, 60000)
        self.train_labels = self.extract_labels(train_labels_file, 60000)

        self.test_data = self.extract_data(test_data_file, 10000)
        self.test_labels = self.extract_labels(test_labels_file, 10000)

        self.validation_data = self.train_data[:util.VALIDATION_SIZE, :, :, :]
        self.validation_labels = self.train_labels[:util.VALIDATION_SIZE]
        self.train_data = self.train_data[util.VALIDATION_SIZE:, :, :, :]
        self.train_labels = self.train_labels[util.VALIDATION_SIZE:]
        self.train_size = self.train_labels.shape[0]

    def extract_data(self,filename, num_images):
        """Extract the images into a 4D tensor [image index, y, x, channels].

        For MNIST data, the number of channels is always 1.

        Values are rescaled from [0, 255] down to [-0.5, 0.5].
        """
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            # Skip the magic number and dimensions; we know these values.
            bytestream.read(16)

            buf = bytestream.read(util.IMAGE_SIZE * util.IMAGE_SIZE * num_images)
            data = numpy.frombuffer(buf, dtype=numpy.uint8).astype(numpy.float32)
            data = (data - (util.PIXEL_DEPTH / 2.0)) / util.PIXEL_DEPTH
            data = data.reshape(num_images, util.IMAGE_SIZE, util.IMAGE_SIZE, 1)
            return data

    # def extract_labels(filename, num_images):
    #     """Extract the labels into a 1-hot matrix [image index, label index]."""
    #     print('Extracting', filename)
    #     with gzip.open(filename) as bytestream:
    #         # Skip the magic number and count; we know these values.
    #         bytestream.read(8)
    #         buf = bytestream.read(1 * num_images)
    #         labels = numpy.frombuffer(buf, dtype=numpy.uint8)
    #     # Convert to dense 1-hot representation.
    #     return (numpy.arange(util.NUM_CLASSES) == labels[:, None]).astype(numpy.float32)

    def extract_labels(self,filename, num_images):
        """Extract the labels into a 1-hot matrix [image index, label index]."""
        print('Extracting', filename)
        with gzip.open(filename) as bytestream:
            # Skip the magic number and count; we know these values.
            bytestream.read(8)
            buf = bytestream.read(1 * num_images)
            labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        # Convert to dense 1-hot representation.
        return (numpy.arange(util.NUM_CLASSES) == labels[:, None]).astype(numpy.float32)





