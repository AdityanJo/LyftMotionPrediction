import torch
import torch.nn as nn
import torch.nn.functional as F

class SusNetv2(nn.Module):
    # v1 is buggy with tensorboard :(
    def __init__(self, num_history_channels, future_num_frames):
        super(SusNetv2, self).__init__()
        self.num_history_channels = (num_history_channels+1) * 2
        self.num_in_channels = 3 + self.num_history_channels
        self.future_num_frames = future_num_frames
        self.num_targets = 2 * self.future_num_frames

        self.preprocess = nn.Sequential(
            nn.Conv2d(self.num_in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.stem_branch1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.stem_branch2=nn.MaxPool2d(2)

        self.abstractor_lvl_1_1_left = self.build_left_block(64, 64)
        self.abstractor_lvl_1_1_right = self.build_right_block(64, 64)
        self.squeeze_lvl_1_1 = nn.Conv2d(192, 128, kernel_size=1, stride=1)

        self.abstractor_lvl_1_2_left = self.build_left_block(128, 128)
        self.abstractor_lvl_1_2_right = self.build_right_block(128, 128)
        self.squeeze_lvl_1_2 = nn.Conv2d(384, 128, kernel_size=1, stride=1)

        self.abstractor_lvl_1_3_left = self.build_left_block(128, 128)
        self.abstractor_lvl_1_3_right = self.build_right_block(128, 256)
        self.squeeze_lvl_1_3 = nn.Conv2d(512, 128, kernel_size=1, stride=1)

        self.shrink_lvl_1 = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)

        self.abstractor_lvl_2_1_left = self.build_left_block(128, 128)
        self.abstractor_lvl_2_1_right = self.build_right_block(128, 256)
        self.squeeze_lvl_2_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.abstractor_lvl_2_2_left = self.build_left_block(256, 128)
        self.abstractor_lvl_2_2_right = self.build_right_block(256, 128)
        self.squeeze_lvl_2_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.abstractor_lvl_2_3_left = self.build_left_block(256, 128)
        self.abstractor_lvl_2_3_right = self.build_right_block(256, 128)
        self.squeeze_lvl_2_3 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.abstractor_lvl_2_4_left = self.build_left_block(256, 128)
        self.abstractor_lvl_2_4_right = self.build_right_block(256, 128)
        self.squeeze_lvl_2_4 = nn.Conv2d(512, 256, kernel_size=1, stride=1)

        self.shrink_lvl_2 = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

        self.abstractor_lvl_3_1_left = self.build_left_block(256, 256)
        self.abstractor_lvl_3_1_right = self.build_right_block(256, 128)
        self.squeeze_lvl_3_1 = nn.Conv2d(640, 256, kernel_size=1, stride=1)

        self.abstractor_lvl_3_2_left = self.build_left_block(256, 256)
        self.abstractor_lvl_3_2_right = self.build_right_block(256, 128)
        self.squeeze_lvl_3_2 = nn.Conv2d(640, 128, kernel_size=1, stride=1)

        self.abstractor_lvl_3_3_left = self.build_left_block(128, 128)
        self.abstractor_lvl_3_3_right = self.build_right_block(128, 128)
        self.squeeze_lvl_3_3 = nn.Conv2d(384, 256, kernel_size=1, stride=1)

        self.abstractor_lvl_3_4_left = self.build_left_block(256, 256)
        self.abstractor_lvl_3_4_right = self.build_right_block(256, 128)
        self.squeeze_lvl_3_4 = nn.Conv2d(640, 256, kernel_size=1, stride=1)

        self.abstractor_lvl_3_5_left = self.build_left_block(256, 256)
        self.abstractor_lvl_3_5_right = self.build_right_block(256, 256)
        self.squeeze_lvl_3_5 = nn.Conv2d(768, 512, kernel_size=1, stride=1)

        self.abstractor_lvl_3_6_left = self.build_left_block(512, 256)
        self.abstractor_lvl_3_6_right = self.build_right_block(512, 256)
        self.squeeze_lvl_3_6 = nn.Conv2d(1024, 512, kernel_size=1, stride=1)

        self.shrink_lvl_3 = nn.Conv2d(512, 2048, kernel_size=3, stride=2, padding=1)


        self.predictor=nn.Linear(2048,self.num_targets)

    def build_left_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features//2, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_features//2),
            nn.ReLU(),
            nn.Conv2d(out_features//2, out_features, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU(),
            nn.Conv2d(out_features, out_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )
    def build_right_block(self, in_features, out_features):
        return nn.Sequential(
            nn.Conv2d(in_features, out_features*2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_features*2),
            nn.ReLU(),
            nn.Conv2d(out_features*2, out_features, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_features),
            nn.ReLU()
        )
    def forward(self, x):

        x = self.preprocess(x)
        x1 = self.stem_branch1(x)
        x2 = self.stem_branch2(x)
        x = torch.cat([x1, x2], axis=1)

        x1 = self.abstractor_lvl_1_1_left(x)
        x2 = self.abstractor_lvl_1_1_right(x)
        x = self.squeeze_lvl_1_1(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_1_2_left(x)
        x2 = self.abstractor_lvl_1_2_right(x)
        x = self.squeeze_lvl_1_2(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_1_3_left(x)
        x2 = self.abstractor_lvl_1_3_right(x)
        x = self.squeeze_lvl_1_3(torch.cat([x,x1,x2],axis=1))

        x = self.shrink_lvl_1(x)

        x1 = self.abstractor_lvl_2_1_left(x)
        x2 = self.abstractor_lvl_2_1_right(x)
        x = self.squeeze_lvl_2_1(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_2_2_left(x)
        x2 = self.abstractor_lvl_2_2_right(x)
        x = self.squeeze_lvl_2_2(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_2_3_left(x)
        x2 = self.abstractor_lvl_2_3_right(x)
        x = self.squeeze_lvl_2_3(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_2_4_left(x)
        x2 = self.abstractor_lvl_2_4_right(x)
        x = self.squeeze_lvl_2_4(torch.cat([x,x1,x2],axis=1))

        x = self.shrink_lvl_2(x)

        x1 = self.abstractor_lvl_3_1_left(x)
        x2 = self.abstractor_lvl_3_1_right(x)
        x = self.squeeze_lvl_3_1(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_3_2_left(x)
        x2 = self.abstractor_lvl_3_2_right(x)
        x = self.squeeze_lvl_3_2(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_3_3_left(x)
        x2 = self.abstractor_lvl_3_3_right(x)
        x = self.squeeze_lvl_3_3(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_3_4_left(x)
        x2 = self.abstractor_lvl_3_4_right(x)
        x = self.squeeze_lvl_3_4(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_3_5_left(x)
        x2 = self.abstractor_lvl_3_5_right(x)
        x = self.squeeze_lvl_3_5(torch.cat([x,x1,x2],axis=1))

        x1 = self.abstractor_lvl_3_6_left(x)
        x2 = self.abstractor_lvl_3_6_right(x)
        x = self.squeeze_lvl_3_6(torch.cat([x,x1,x2],axis=1))

        x = self.shrink_lvl_3(x)
        x=F.adaptive_avg_pool2d(x, (1,1))
        x=nn.Flatten()(x)
        x=self.predictor(x)
        return x
