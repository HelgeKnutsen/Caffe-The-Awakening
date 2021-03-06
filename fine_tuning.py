from PIL import Image
import sys
import caffe
import numpy as np
from pylab import *
import tempfile
import os
from caffe import layers as L
from caffe import params as P
from caffe.proto import caffe_pb2
import time


# Read before running script:
# --------------------------------------
# When running the script make sure to set the correct caffe_root. All the files used in the script should be located in the caffe_root-folder.
#
# Make sure to set the correct source_path where the list of traning and test data can be found. These files should be named train.txt and test.txt, respectively.
# If necessary, change the label_path to the correct path with the label file.
#
# In order to keep working on the weights from previous runs, set savedWeights to the given folder.
# Then set reset = False in loadWeights function call.
#
# If this is the first time running the script and there are no weights saved from previous runs, set reset = True and load the weights from reference weights.
# If necessary, change the reference weights path.
#
# Choose the number of iterations niter appropriately.
#
# In order to save the progress of the trained weights, set tmp_save = False in the run_solvers function call.
# --------------------------------------

weight_param = dict(lr_mult=1, decay_mult=1)
bias_param = dict(lr_mult=2, decay_mult=0)
learned_param = [weight_param, bias_param]

frozen_param = [dict(lr_mult=0)] * 2

NUM_STYLE_LABELS = 2

# Functions that initialize the caffenet:
# --------------------------------------

def conv_relu(bottom, ks, nout, stride=1, pad=0, group=1,
              param=learned_param,
              weight_filler=dict(type='gaussian', std=0.01),
              bias_filler=dict(type='constant', value=0.1)):
    conv = L.Convolution(bottom, kernel_size=ks, stride=stride,
                         num_output=nout, pad=pad, group=group,
                         param=param, weight_filler=weight_filler,
                         bias_filler=bias_filler)
    return conv, L.ReLU(conv, in_place=True)


def fc_relu(bottom, nout, param=learned_param,
            weight_filler=dict(type='gaussian', std=0.005),
            bias_filler=dict(type='constant', value=0.1)):
    fc = L.InnerProduct(bottom, num_output=nout, param=param,
                        weight_filler=weight_filler,
                        bias_filler=bias_filler)
    return fc, L.ReLU(fc, in_place=True)


def max_pool(bottom, ks, stride=1):
    return L.Pooling(bottom, pool=P.Pooling.MAX, kernel_size=ks, stride=stride)


def caffenet(data, label=None, train=True, num_classes=1000,
             classifier_name='fc8', learn_all=False):
    """Returns a NetSpec specifying CaffeNet, following the original proto text
       specification (./models/bvlc_reference_caffenet/train_val.prototxt)."""
    n = caffe.NetSpec()
    n.data = data
    param = learned_param if learn_all else frozen_param
    n.conv1, n.relu1 = conv_relu(n.data, 11, 96, stride=4, param=param)
    n.pool1 = max_pool(n.relu1, 3, stride=2)
    n.norm1 = L.LRN(n.pool1, local_size=5, alpha=1e-4, beta=0.75)
    n.conv2, n.relu2 = conv_relu(n.norm1, 5, 256, pad=2, group=2, param=param)
    n.pool2 = max_pool(n.relu2, 3, stride=2)
    n.norm2 = L.LRN(n.pool2, local_size=5, alpha=1e-4, beta=0.75)
    n.conv3, n.relu3 = conv_relu(n.norm2, 3, 384, pad=1, param=param)
    n.conv4, n.relu4 = conv_relu(n.relu3, 3, 384, pad=1, group=2, param=param)
    n.conv5, n.relu5 = conv_relu(n.relu4, 3, 256, pad=1, group=2, param=param)
    n.pool5 = max_pool(n.relu5, 3, stride=2)
    n.fc6, n.relu6 = fc_relu(n.pool5, 4096, param=param)
    if train:
        n.drop6 = fc7input = L.Dropout(n.relu6, in_place=True)
    else:
        fc7input = n.relu6
    n.fc7, n.relu7 = fc_relu(fc7input, 4096, param=param)
    if train:
        n.drop7 = fc8input = L.Dropout(n.relu7, in_place=True)
    else:
        fc8input = n.relu7
    # always learn fc8 (param=learned_param)
    fc8 = L.InnerProduct(fc8input, num_output=num_classes, param=learned_param)
    # give fc8 the name specified by argument `classifier_name`
    n.__setattr__(classifier_name, fc8)
    if not train:
        n.probs = L.Softmax(fc8)
    if label is not None:
        n.label = label
        n.loss = L.SoftmaxWithLoss(fc8, n.label)
        n.acc = L.Accuracy(fc8, n.label)
    # write the net to a temporary file and return its filename
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(n.to_proto()))
        return f.name
    # In order to save permanently the prototxt, uncomment the following four lines below and comment the corresponding three lines above.
    #subset = 'train' if train else 'test'
    #with open(caffe_root+'myData/net%s.prototxt'% (subset), 'w') as f:
        #f.write(str(n.to_proto()))
        #return f.name


def style_net(source_path, train=True, learn_all=False, subset=None): # Choose learn_all = True to train all layers, learn_all = False to only tune last layer
    if subset is None:
        subset = 'train' if train else 'test'
    source = caffe_root + source_path + '%s.txt' % subset # text-file with the image-paths (and label numbers)
    transform_param = dict(mirror=train, crop_size=227,
                           mean_file=caffe_root + 'data/ilsvrc12/imagenet_mean.binaryproto')
    style_data, style_label = L.ImageData(transform_param=transform_param, source=source,
                                          batch_size=50, new_height=256, new_width=256, ntop=2)
    return caffenet(data=style_data, label=style_label, train=train,
                    num_classes=NUM_STYLE_LABELS,
                    classifier_name='fc8_tx1',
                    learn_all=learn_all)

# Display functions:
# --------------------------------------

def disp_preds(net, image, labels, k=5, name='ImageNet'):
    input_blob = net.blobs['data']
    net.blobs['data'].data[0, ...] = image
    probs = net.forward(start='conv1')['probs'][0]
    top_k = (-probs).argsort()[:k]
    print 'top %d predicted %s labels =' % (k, name)
    print '\n'.join('\t(%d) %5.2f%% %s' % (i+1, 100*probs[p], labels[p])
                    for i, p in enumerate(top_k))


# Solver functions:
# --------------------------------------

def solver(train_net_path, test_net_path=None, base_lr=0.0001):
    s = caffe_pb2.SolverParameter()

    # Specify locations of the train and (maybe) test networks.
    s.train_net = train_net_path
    if test_net_path is not None:
        s.test_net.append(test_net_path)
        s.test_interval = 1000  # Test after every 1000 training iterations.
        s.test_iter.append(100)  # Test on 100 batches each time we test.

    # The number of iterations over which to average the gradient.
    # Effectively boosts the training batch size by the given factor, without
    # affecting memory utilization.
    s.iter_size = 1

    s.max_iter = 100000  # # of times to update the net (training iterations)

    # Solve using the stochastic gradient descent (SGD) algorithm.
    # Other choices include 'Adam' and 'RMSProp'.
    s.type = 'SGD'

    # Set the initial learning rate for SGD.
    s.base_lr = base_lr

    # Set `lr_policy` to define how the learning rate changes during training.
    # Here, we 'step' the learning rate by multiplying it by a factor `gamma`
    # every `stepsize` iterations.
    s.lr_policy = 'step'
    s.gamma = 0.1
    s.stepsize = 20000

    # Set other SGD hyperparameters. Setting a non-zero `momentum` takes a
    # weighted average of the current gradient and previous gradients to make
    # learning more stable. L2 weight decay regularizes learning, to help prevent
    # the model from overfitting.
    s.momentum = 0.9
    s.weight_decay = 5e-4

    # Display the current training loss and accuracy every 1000 iterations.
    s.display = 1000

    # Snapshots are files used to store networks we've trained.  Here, we'll
    # snapshot every 10K iterations -- ten times during training.
    s.snapshot = 10000
    s.snapshot_prefix = caffe_root + 'models/finetune_flickr_style/finetune_flickr_style'

    # Train on the GPU.  Using the CPU to train large networks is very slow.
    s.solver_mode = caffe_pb2.SolverParameter.GPU

    # Write the solver to a temporary file and return its filename.
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write(str(s))
        return f.name


def run_solvers(niter, solvers, disp_interval=10, tmp_save = True):
    """Run solvers for niter iterations,
       returning the loss and accuracy recorded each iteration.
       `solvers` is a list of (name, solver) tuples."""
    blobs = ('loss', 'acc')
    loss, acc = ({name: np.zeros(niter) for name, _ in solvers}
                 for _ in blobs)
    for it in range(niter):
        for name, s in solvers:
            s.step(1)  # run a single SGD step in Caffe
            loss[name][it], acc[name][it] = (s.net.blobs[b].data.copy()
                                             for b in blobs)
        if it % disp_interval == 0 or it + 1 == niter:
            loss_disp = '; '.join('%s: loss=%.3f, acc=%2.5f%%' %
                                  (n, loss[n][it], np.round(100*acc[n][it]))
                                  for n, _ in solvers)
            print '%3d) %s' % (it, loss_disp)
    # Save the learned weights from both nets.
    if tmp_save == True: # Temporary save weights
        weight_dir = tempfile.mkdtemp()
    else: # Permanently save weights
        weight_dir = caffe_root + savedWeights # Directory
    weights = {}
    for name, s in solvers:
        filename = 'weights.%s.caffemodel' % name
        weights[name] = os.path.join(weight_dir, filename)
        s.net.save(weights[name])
    return loss, acc, weights


def eval_style_net(source_path, weights, test_iters=10):
    test_net = caffe.Net(style_net(source_path, train=False), weights, caffe.TEST)
    accuracy = 0
    for it in xrange(test_iters):
        accuracy += test_net.forward()['acc']
    accuracy /= test_iters
    return test_net, accuracy


# --------------------------------------

# Load style labels to style_labels
def loadLables(path):
    style_label_file = caffe_root + path
    style_labels = list(np.loadtxt(style_label_file, str, delimiter='\n'))
    if NUM_STYLE_LABELS > 0:
        style_labels = style_labels[:NUM_STYLE_LABELS]
    return style_labels

# Load weights
def loadWeights(reset = True):
    if reset == True: # Reset weights to reference weights
        return caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
    else: # Load previously trained weights
        return caffe_root + savedWeights + 'weights.pretrained.caffemodel'


# Initialize:
# --------------------------------------

#caffe.set_device(0)
caffe.set_mode_cpu()

caffe_root = '/home/helge/caffe/'

source_path = 'myData/2016-06-29 20m/' # path to text-file with the image-paths (and label numbers)
#sys.path.insert(0, caffe_root + 'python')

label_path = 'myData/test_names.txt' # file with labels

savedWeights = 'savedWeights/2016-06-29 20m, weights/' # Make sure to create a folder in caffe_root where weights can be saved
weights = loadWeights(reset = False) # Set reset = True to import weights from pretrained net with different classes,
                                     # set reset = False to keep training the saved weights with current classes
assert os.path.exists(weights)
print weights

style_labels = loadLables(label_path)



print '\nLoaded style labels:\n', ', '.join(style_labels)

niter = 50 # number of iterations

style_solver_filename = solver(style_net(source_path, train=True, learn_all = False)) # create tmp-file, return filename,
# add argument under style_net learn_all = True to train all layers, e.g. style_net(train = True, learn_all = True)
style_solver = caffe.get_solver(style_solver_filename) # get solver
style_solver.net.copy_from(weights) # Import weights in solver

print 'Running solvers for %d iterations...' % niter
solvers = [('pretrained', style_solver)] # first variable yields name of file where trained weights are saved, e.g. 'pretrained' yields 'weights.pretrained.caffemodel'
loss, acc, weights = run_solvers(niter, solvers, tmp_save = False) # Set tmp_save = True to save the weights temporarily (only during runtime of the program),
                                                                   # set tmp_save = False to save weights permanently in given folder caffe_root + savedWeights
print 'Done.'

del style_solver, solvers # delete to save memory

style_weights = weights['pretrained']
test_net, accuracy = eval_style_net(source_path, style_weights, test_iters = 5)
print 'Accuracy, trained from ImageNet initialization: %3.1f%%' % (100*accuracy, )

#########################################################################
# Helper function for deprocessing preprocessed images, e.g., for display.
def deprocess_net_image(image):
    image = image.copy()              # don't modify destructively
    image = image[::-1]               # BGR -> RGB
    image = image.transpose(1, 2, 0)  # CHW -> HWC
    image += [123, 117, 104]          # (approximately) undo mean subtraction

    # clamp values in [0, 255]
    image[image < 0], image[image > 255] = 0, 255

    # round and cast from float32 to uint8
    image = np.round(image)
    image = np.require(image, dtype=np.uint8)

    return image

n = 5
avg_time = 0 # average searching time
for batch_index in xrange(n):
    tic = time.time()
    image = test_net.blobs['data'].data[batch_index]
    #Show image
    img = deprocess_net_image(image)
    img = Image.fromarray(np.asarray(img, dtype=np.uint8), 'RGB')
    #img.show()
    plt.imshow(deprocess_net_image(image))
    print 'actual label =', style_labels[int(test_net.blobs['label'].data[batch_index])]
    disp_preds(test_net, image, style_labels, k=2)
    toc = time.time()
    print 'Searching time = ', toc-tic ,'sec'
    print '\n'
    plt.pause(2)
    plt.close('all')
    avg_time += toc - tic

avg_time /= n

print 'The average searching time was ', avg_time, 'sec'