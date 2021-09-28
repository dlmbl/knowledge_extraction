import torch

class Vgg2D(torch.nn.Module):
    def __init__(
            self,
            input_size,
            input_channels,
            fmaps=12,
            downsample_factors=[(2, 2), (2, 2), (2, 2), (2, 2)],
            output_classes=6):

        self.input_size = input_size

        super(Vgg2D, self).__init__()

        current_fmaps = 1
        current_size = tuple(input_size)

        features = []
        for i in range(len(downsample_factors)):

            features += [
                torch.nn.Conv2d(
                    current_fmaps,
                    fmaps,
                    kernel_size=3,
                    padding=1),
                torch.nn.BatchNorm2d(fmaps),
                torch.nn.ReLU(inplace=True),
                torch.nn.Conv2d(
                    fmaps,
                    fmaps,
                    kernel_size=3,
                    padding=1),
                torch.nn.BatchNorm2d(fmaps),
                torch.nn.ReLU(inplace=True),
                torch.nn.MaxPool2d(downsample_factors[i])
            ]

            current_fmaps = fmaps
            fmaps *= 2

            size = tuple(
                int(c/d)
                for c, d in zip(current_size, downsample_factors[i]))
            check = (
                s*d == c
                for s, d, c in zip(size, downsample_factors[i], current_size))
            assert all(check), \
                "Can not downsample %s by chosen downsample factor" % \
                (current_size,)
            current_size = size

        self.features = torch.nn.Sequential(*features)

        classifier = [
            torch.nn.Linear(
                current_size[0] *
                current_size[1] *
                current_fmaps,
                4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(
                4096,
                4096),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(),
            torch.nn.Linear(
                4096,
                output_classes)
        ]

        self.classifier = torch.nn.Sequential(*classifier)

        #print(self)
    
    def forward(self, raw):
        shape = tuple(raw.shape)
        if shape[1] != 1: #== (1, 3, 128, 128) - rgb conversion needed for captum.
            raw = raw[:,0,:,:].reshape(shape[0],1,shape[2], shape[3])
            
        raw_with_channels = raw.reshape(
            shape[0],
            1,
            shape[2],
            shape[3])
    
        raw_with_channels = raw
        f = self.features(raw_with_channels)
        f = f.view(f.size(0), -1)
        y = self.classifier(f)
        return y
