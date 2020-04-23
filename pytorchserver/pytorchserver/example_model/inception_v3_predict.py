import torch
import torchvision.models as models


def predict():
    # https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth
    model_file = "/Users/jinxiang/Downloads/inception_v3_google-1a9a5a14.pth"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputs = torch.randn(1, 3, 299, 299).to(device)
    net = models.inception_v3().to(device)
    net.load_state_dict(torch.load(model_file, map_location=device))
    net.eval()
    output = net(inputs)
    print(output)

class Net(models.Inception3):
    def __init__(self):
        super(Net, self).__init__()


def predict_with_class():
    # https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth
    model_file = "/Users/jinxiang/Downloads/inception_v3_google-1a9a5a14.pth"
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    inputs = torch.randn(1, 3, 299, 299).to(device)
    net = Net().to(device)
    net.load_state_dict(torch.load(model_file, map_location=device))
    net.eval()
    output = net(inputs)
    print(output)


def predict_mlu():
    # https://download.pytorch.org/models/inception_v3_google-1a9a5a14.pth
    model_file = "/Users/jinxiang/Downloads/inception_v3_google-1a9a5a14.pth"
    inputs = torch.randn(1, 3, 299, 299)
    net = models.inception_v3()
    net.load_state_dict(torch.load(model_file))
    net.eval().float().mlu()
    output = net(inputs.mlu())
    output = output.cpu().type(torch.FloatTensor)
    print(output)


if __name__ == '__main__':
    # predict()
    predict_with_class()
