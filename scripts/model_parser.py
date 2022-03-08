import torch
import copy
from collections import OrderedDict

modules = ["d1.0.0.weight", "d1.0.0.bias", "d1.1.0.weight", "d1.1.0.bias", "d2.0.0.weight", "d2.0.0.bias",
           "d2.1.0.weight", "d2.1.0.bias", "d3.0.0.weight", "d3.0.0.bias", "d3.1.0.weight", "d3.1.0.bias",
           "d3.2.0.weight", "d3.2.0.bias", "d4.0.0.weight", "d4.0.0.bias", "d4.1.0.weight", "d4.1.0.bias",
           "d4.2.0.weight", "d4.2.0.bias", "d5.0.0.weight", "d5.0.0.bias", "d5.1.0.weight", "d5.1.0.bias",
           "d5.2.0.weight", "d5.2.0.bias", "classifier.0.weight", "classifier.0.bias", "classifier.3.weight",
           "classifier.3.bias", "classifier.6.weight", "classifier.6.bias"]

modules_no_classifier = ["d1.0.0.weight", "d1.0.0.bias", "d1.1.0.weight", "d1.1.0.bias", "d2.0.0.weight",
                         "d2.0.0.bias", "d2.1.0.weight", "d2.1.0.bias", "d3.0.0.weight", "d3.0.0.bias",
                         "d3.1.0.weight", "d3.1.0.bias", "d3.2.0.weight", "d3.2.0.bias", "d4.0.0.weight",
                         "d4.0.0.bias", "d4.1.0.weight", "d4.1.0.bias", "d4.2.0.weight", "d4.2.0.bias",
                         "d5.0.0.weight", "d5.0.0.bias", "d5.1.0.weight", "d5.1.0.bias", "d5.2.0.weight", "d5.2.0.bias"]


def parse(no_classifier=True):
    state_dict = torch.load('../models/vgg16_classifier_torchvision.pth')
    new_dict = OrderedDict()
    counter = 0
    for (k, v) in state_dict.items():
        if no_classifier and counter >= 26:
            break
        new_dict[modules[counter]] = copy.deepcopy(v)
        counter += 1
    torch.save(new_dict, '../models/vunet_parsed.pth')


def main():
    parse()


if __name__ == "__main__":
    main()