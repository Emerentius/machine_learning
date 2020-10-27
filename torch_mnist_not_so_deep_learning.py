import numpy as np
import gzip
import torch
from typing import List, Tuple, Any

LayerWeights = Any  # torch.Tensor
LayerBiases = Any  # torch.Tensor
Model = List[Tuple[LayerWeights, LayerBiases]]

# ==================================================
# Original MNIST reading code taken from https://stackoverflow.com/a/53570674.
# Then adapted with less hardcoded values.
# ==================================================
def read_mnist_image_file(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        header = f.read(16)
        # first 4 bytes are to check endianness of 32-bit
        # integers.
        # but it's irrelevant if we're constructing ints from
        # bytes. We know the endianness of the storage already.
        n_images = int.from_bytes(header[4:8], "big")
        n_rows = int.from_bytes(header[8:12], "big")
        n_columns = int.from_bytes(header[12:16], "big")

        # np.fromfile doesn't work with gzip as it doesn't return a real file object
        # (one with a file descriptor)
        # data = np.fromfile(f, dtype=np.uint8) / 256
        # TODO: how the heck can you just construct a torch tensor from bytes
        #       without a numpy array.
        #       And get rid of all this copying.
        buf = f.read()
        # array must be writeable for torch, so copy it.
        data = np.frombuffer(
            buf,
            dtype=np.uint8,
        ).copy()
        data: torch.Tensor = torch.from_numpy(data).float()

        return data.view(n_images, n_rows, n_columns)


def read_mnist_label_file(path: str) -> torch.Tensor:
    with gzip.open(path, "rb") as f:
        # 8-byte header contains magic number for endianness check
        # and number of labels.
        # neither are needed.
        _header = f.read(8)
        n_labels = int.from_bytes(_header[4:8], "big")

        # data: torch.Tensor = np.fromfile(f, dtype=np.uint8)
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)

        assert data.size == n_labels
        return torch.from_numpy(data)


training_data = read_mnist_image_file("MNIST/train-images-idx3-ubyte.gz")
n_images, n_rows, n_cols = training_data.shape
training_labels = read_mnist_label_file("MNIST/train-labels-idx1-ubyte.gz")


# import matplotlib.pyplot as plt

# for i in range(10):
#     print(training_labels[i])
#     image = np.asarray(training_data[i])
#     plt.imshow(image)
#     plt.show()
# ==================================================


def sigmoid(x):
    x = x.clamp(-500, 500)
    return 1 / (1 + torch.exp(-x))


def sigmoid_derivative(x):
    # correct, but less numerically stable
    # neg_exp = np.exp(-x)
    # return neg_exp / (1 + neg_exp) ** 2

    s = sigmoid(x)
    return (1 - s) * s


def apply_model(
    input_: torch.Tensor, model: List[Tuple[torch.Tensor, torch.Tensor]]
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Returns result of last layer as well as z for all neuron layers"""
    # input is 28x28 array
    last_layer = input_

    assert input_.dtype == torch.float

    neuron_outputs_and_z = [(input_, None)]
    for weights, bias in model:
        z = weights @ last_layer + bias
        last_layer = sigmoid(z)
        neuron_outputs_and_z.append((last_layer, z))

    return neuron_outputs_and_z


# def cost(output, expected_output):
#     return np.square(output - expected_output)


def cost_derivative(output, expected_output):
    # leaving out factor of 2 as it's irrelevant for this algorithm to work
    return output - expected_output


def backpropagate(
    model: Model,
    weights_transposes: List[torch.Tensor],
    input: torch.Tensor,
    # output: torch.Tensor,
    expected_output: torch.Tensor,
    neuron_outputs_and_z: List[Tuple[torch.Tensor, torch.Tensor]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Returns desired changes for given input"""

    # # cost_ = cost(output, expected_output)
    # delta = cost_derivative(output, expected_output) * sigmoid_derivative(zs[-1])
    model_changes = [(torch.zeros(w.shape), torch.zeros(b.shape)) for w, b in model]

    # # grad_b
    # model_changes[-1][1] = delta
    # layer_inputs, _ = neuron_outputs_and_z[-2]

    # for idx, _ in np.ndenumerate(model_changes[-1][0]):
    #     row, col = idx
    #     model_changes[-1][0][idx] = delta[row] * layer_inputs[col]

    delta = None
    for layer_idx in reversed(range(len(model))):
        _, layer_z = neuron_outputs_and_z[layer_idx + 1]
        previous_layer_output, _ = neuron_outputs_and_z[layer_idx]

        if delta is None:
            output, _ = neuron_outputs_and_z[layer_idx + 1]
            delta = cost_derivative(output, expected_output) * sigmoid_derivative(
                layer_z
            )
        else:
            w_t = weights_transposes[layer_idx + 1]
            delta = w_t @ delta * sigmoid_derivative(layer_z)

        # assert False

        # weights
        weights_changes = -delta.view(-1, 1) @ previous_layer_output.view(-1, 1).T

        # biases
        # have to overwrite tuple in list because tuples are "immutable"
        model_changes[layer_idx] = weights_changes, -delta

    return model_changes


def train(model: Model, input_batch: List[Tuple[torch.Tensor, torch.Tensor]]):

    n = len(input_batch)
    accumulated_model_changes = [
        (torch.zeros(w.shape), torch.zeros(b.shape)) for w, b in model
    ]

    for input_, expected_output in input_batch:
        neuron_outputs_and_z = apply_model(input_, model)

        weights_transposes = [w.T for w, _bias in model]
        model_changes = backpropagate(
            model, weights_transposes, input_, expected_output, neuron_outputs_and_z
        )

        add_model_changes(accumulated_model_changes, model_changes)

    # add average changes onto model
    learning_rate = 0.1
    div_model(accumulated_model_changes, n / learning_rate)
    # print("changes")
    # print(accumulated_model_changes)
    add_model_changes(model, accumulated_model_changes)


def add_model_changes(model: Model, diff: Model):
    for (weights, biases), (diff_weights, diff_biases) in zip(model, diff):
        weights += diff_weights
        biases += diff_biases


def div_model(model: Model, divisor: float):
    for weights, biases in model:
        weights /= divisor
        biases /= divisor


# ==================================================

linear_training_data = training_data.view(n_images, n_rows * n_cols)

from torch import randn

model = [
    (randn(16, n_rows * n_cols), randn(16)),
    (randn(16, 16), randn(16)),
    (randn(10, 16), randn(10)),
]

neuron_outputs_and_zs = apply_model(linear_training_data[2], model)
print(neuron_outputs_and_zs[-1])


def label_to_array(label: int) -> torch.Tensor:
    a = torch.zeros(10)
    a[label] = 1.0
    return a


def show_result(model, input_, expected_output):
    neuron_outputs_and_zs = apply_model(input_, model)
    output, _ = neuron_outputs_and_zs[-1]
    print(output)
    print(expected_output)


expected_outputs = [label_to_array(label) for label in training_labels]
# train(model, list(zip(linear_training_data[:20], expected_outputs[:20])))

batch_size = 10
N = len(linear_training_data)

import random

rng = random.Random(0)

training_data_ = list(
    zip(
        linear_training_data,
        expected_outputs,
    )
)

sample = training_data_[:100]


def nn_answer(model: Model, input_: torch.Tensor) -> int:
    activations_and_zs = apply_model(input_, model)

    return torch.argmax(activations_and_zs[-1][0])


for epoch in range(15):
    n_correct = sum(
        (nn_answer(model, input_) == torch.argmax(expected_output)).item()
        for input_, expected_output in sample
    )

    percentage_correct = round(n_correct / len(sample) * 100, 1)
    print(f"{n_correct}/{len(sample)} correct ({percentage_correct})")

    rng.shuffle(training_data_)

    print(f"n_images: {N}")
    for i in range(N // batch_size):
        if (i + 1) % 1000 == 0:
            print(f"{(i+1) * batch_size} / {N}")
        train(
            model,
            training_data_[i * batch_size : (i + 1) * batch_size],
        )

    print(f"\nepoch {epoch}:")
