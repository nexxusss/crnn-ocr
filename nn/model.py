import torch.nn as nn
import torch

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size) -> None:
        super(BidirectionalLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True)
        # multiply by 2 for bidirectional
        self.fc = nn.Linear(hidden_size * 2, output_size)

        self.hidden_size = hidden_size

    def forward(self, input_data):
        # input_data = (sequence length, batch size, input_size)

        out, _ = self.lstm(input_data)

        # Concatenate the forward and backward hidden states
        concatenated_out = torch.cat([out[:, :, :self.hidden_size], out[:, :, self.hidden_size:]], dim=2)

        # feed to the fully connected

        final_output = self.fc(concatenated_out[-1, :, :]) 

        return final_output
    

class CRNN(nn.Module):

    def __init__(self, image_height, number_channels, n_classes, n_hidden, n_rnn=2, leakyRelu=False) -> None:
        # number of LSTM layers by default = 2 as per the paper
        super(CRNN, self).__init__()

        # check if image height is a multiple of 16
        if image_height % 16 != 0:
            raise ValueError(f'Image height should be modulo 16 but given {image_height}')
        
        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        # routine to add layers to the Sequential
        # smart way presented by the paper authors to create the conv layers with corresponding hyperparams
        def convRelu(i, batchNormalization=False):
            nIn = number_channels if i == 0 else nm[i - 1]
            nOut = nm[i]
            cnn.add_module(f'conv{i}', nn.Conv2d(nIn, nOut, ks[i], ss[i], ps[i]))
            if batchNormalization:
                cnn.add_module(f'batchnorm{i}', nn.BatchNorm2d(nOut))
            if leakyRelu:
                cnn.add_module(f'relu{i}', nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f'relu{i}', nn.ReLU(True))

        convRelu(0)
        cnn.add_module(f"pooling0", nn.MaxPool2d(2, 2))
        convRelu(1)
        cnn.add_module(f"pooling1", nn.MaxPool2d(2, 2))
        # in the paper we apply batch normalization here
        convRelu(2, batchNormalization=True)

        convRelu(3)
        cnn.add_module('pooling2', nn.MaxPool2d((2, 2), (2, 1), (0, 1))) 

        convRelu(4, batchNormalization=True)
        convRelu(5)
        cnn.add_module('pooling2', nn.MaxPool2d((2, 2), (2, 1), (0, 1))) 
        convRelu(6, True)

        self.cnn = cnn

        # LSTM Layers
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, n_hidden, n_hidden),
            BidirectionalLSTM(n_hidden, n_hidden, n_classes)
        )

    def forward(self, input):
        # conv features
        conv = self.cnn(input)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # rnn features
        output = self.rnn(conv)

        return output

