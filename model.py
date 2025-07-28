import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger

from dataloader import PETDataModule


class AttentionBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.depthwise = nn.Conv2d(
            filters, filters, kernel_size=3, padding=1, groups=filters
        )
        self.pointwise = nn.Conv2d(filters, filters, kernel_size=1)
        self.conv1x1 = nn.Conv2d(filters, filters, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(filters)
        self.bn2 = nn.BatchNorm2d(filters)
        self.att_conv = nn.Conv2d(filters, 1, kernel_size=1)
        self.bn_att = nn.BatchNorm2d(1)

    def forward(self, x):
        g = self.depthwise(x)
        g = self.pointwise(g)
        x = self.conv1x1(x)
        g = self.bn1(g)
        x = self.bn2(x)
        g = F.relu(g)
        x = F.relu(x)

        att = g + x
        att = self.att_conv(att)
        att = self.bn_att(att)
        att = torch.sigmoid(att)

        return x * att


class MultiScaleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1x1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding="same"
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.depthwise3x3 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding="same", groups=in_channels
        )
        self.pointwise3x3 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding="same"
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.depthwise5x5 = nn.Conv2d(
            in_channels, in_channels, kernel_size=5, padding="same", groups=in_channels
        )
        self.pointwise5x5 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding="same"
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.output_conv = nn.Conv2d(
            out_channels * 3, out_channels, kernel_size=1, padding="same"
        )
        self.bn_out = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        conv1x1 = F.hardswish(self.bn1(self.conv1x1(x)))

        conv3x3 = self.depthwise3x3(x)
        conv3x3 = F.hardswish(self.bn2(self.pointwise3x3(conv3x3)))

        conv5x5 = self.depthwise5x5(x)
        conv5x5 = F.hardswish(self.bn3(self.pointwise5x5(conv5x5)))

        concat = torch.cat([conv1x1, conv3x3, conv5x5], dim=1)

        output = F.hardswish(self.bn_out(self.output_conv(concat)))
        return output


class SEBlock(nn.Module):
    def __init__(self, in_channels, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(in_channels, in_channels // ratio)
        self.fc2 = nn.Linear(in_channels // ratio, in_channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class GatedResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size, strides):
        super().__init__()
        # Calculate padding
        padding = (kernel_size - 1) // 2

        self.depthwise_shortcut = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=strides,
            padding=0,  # No padding for 1x1 conv
            groups=in_channels,
        )
        self.pointwise_shortcut = nn.Conv2d(
            in_channels, filters, kernel_size=1, padding=0
        )
        self.bn_shortcut = nn.BatchNorm2d(filters)

        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=strides,
            padding=padding,
            groups=in_channels,
        )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, filters, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(filters)

        self.gate = nn.Conv2d(filters, filters, kernel_size=1)
        self.bn_gate = nn.BatchNorm2d(filters)

    def forward(self, x):
        shortcut = self.bn_shortcut(self.pointwise_shortcut(self.depthwise_shortcut(x)))

        x = F.hardswish(self.bn1(self.depthwise(x)))
        x = self.bn2(self.pointwise(x))

        gate = torch.sigmoid(self.bn_gate(self.gate(x)))

        x = x * gate
        x = x + shortcut
        return F.hardswish(x)


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, f1, f2, f3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, f1, kernel_size=1, padding="same")
        self.bn1 = nn.BatchNorm2d(f1)

        self.conv3_reduce = nn.Conv2d(in_channels, f2[0], kernel_size=1, padding="same")
        self.bn3_reduce = nn.BatchNorm2d(f2[0])
        self.conv3_depthwise = nn.Conv2d(
            f2[0], f2[0], kernel_size=3, padding="same", groups=f2[0]
        )
        self.conv3_pointwise = nn.Conv2d(f2[0], f2[1], kernel_size=1, padding="same")
        self.bn3 = nn.BatchNorm2d(f2[1])

        self.conv5_reduce = nn.Conv2d(in_channels, f3[0], kernel_size=1, padding="same")
        self.bn5_reduce = nn.BatchNorm2d(f3[0])
        self.conv5_depthwise = nn.Conv2d(
            f3[0], f3[0], kernel_size=5, padding="same", groups=f3[0]
        )
        self.conv5_pointwise = nn.Conv2d(f3[0], f3[1], kernel_size=1, padding="same")
        self.bn5 = nn.BatchNorm2d(f3[1])

        self.pool_proj = nn.Conv2d(in_channels, f1, kernel_size=1, padding="same")
        self.bn_pool = nn.BatchNorm2d(f1)

        self.output_conv = nn.Conv2d(
            f1 + f2[1] + f3[1] + f1,
            sum([f1, f2[1], f3[1], f1]),
            kernel_size=1,
            padding="same",
        )
        self.bn_out = nn.BatchNorm2d(sum([f1, f2[1], f3[1], f1]))

    def forward(self, x):
        conv1 = F.hardswish(self.bn1(self.conv1(x)))

        conv3 = F.hardswish(self.bn3_reduce(self.conv3_reduce(x)))
        conv3 = self.conv3_depthwise(conv3)
        conv3 = F.hardswish(self.bn3(self.conv3_pointwise(conv3)))

        conv5 = F.hardswish(self.bn5_reduce(self.conv5_reduce(x)))
        conv5 = self.conv5_depthwise(conv5)
        conv5 = F.hardswish(self.bn5(self.conv5_pointwise(conv5)))

        pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        pool = F.hardswish(self.bn_pool(self.pool_proj(pool)))

        output = torch.cat([conv1, conv3, conv5, pool], dim=1)
        output = F.hardswish(self.bn_out(self.output_conv(output)))

        return output


class DenseBlock(nn.Module):
    def __init__(self, in_channels, num_layers, growth_rate, dropout_rate=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                self._make_dense_layer(
                    in_channels + i * growth_rate, growth_rate, dropout_rate
                )
            )

    def _make_dense_layer(self, in_channels, growth_rate, dropout_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.Dropout(dropout_rate),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
            ),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.bn(x)
        x = F.hardswish(x)
        x = self.conv(x)
        x = self.pool(x)
        return x


class TopLayer(nn.Module):
    def __init__(self, in_features, num_classes, dropout_rate=0.5):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride):
        super().__init__()
        hidden_dim = in_channels * expand_ratio
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(nn.Conv2d(in_channels, hidden_dim, 1, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU(inplace=True))

        layers.extend(
            [
                nn.Conv2d(
                    hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
                nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
            ]
        )

        self.conv = nn.Sequential(*layers)
        self.se = SEBlock(out_channels)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.se(self.conv(x))
        else:
            return self.se(self.conv(x))


class GuruNet(pl.LightningModule):
    def __init__(self, num_classes=5):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)

        self.inverted_residual_blocks = self._make_inverted_residual_blocks()

        self.dense_block1 = DenseBlock(320, num_layers=6, growth_rate=12)
        self.transition1 = TransitionLayer(320 + 6 * 12, out_channels=256)

        self.attention1 = AttentionBlock(256)

        self.dense_block2 = DenseBlock(256, num_layers=6, growth_rate=8)
        self.transition2 = TransitionLayer(256 + 6 * 8, 256)

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.multi_scale_block = MultiScaleBlock(256, 128)

        self.inception_block = InceptionBlock(128, 64, (128, 192), (64, 192))
        self.attention2 = AttentionBlock(512)

        self.gated_residual_block = GatedResidualBlock(
            512, 512, kernel_size=3, strides=2
        )

        self.attention3 = AttentionBlock(512)

        self.final_conv = nn.Conv2d(512, 128, kernel_size=1, padding="same")
        self.final_bn = nn.BatchNorm2d(128)
        self.attention4 = AttentionBlock(128)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.top_layer = TopLayer(128, num_classes)

    def _make_inverted_residual_blocks(self):
        blocks = nn.ModuleList()
        block_params = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 40, 2, 2),
            (6, 80, 2, 3),
            (6, 112, 1, 3),
            (6, 192, 2, 4),
            (6, 320, 1, 1),
        ]

        in_channels = 128
        for expand_ratio, filters, stride, repeats in block_params:
            for _ in range(repeats):
                blocks.append(
                    InvertedResidualBlock(in_channels, filters, expand_ratio, stride)
                )
                in_channels = filters
                stride = 1

        return blocks

    def forward(self, x):
        x = F.hardswish(self.bn1(self.conv1(x)))

        for block in self.inverted_residual_blocks:
            x = block(x)

        x = self.dense_block1(x)
        x = self.transition1(x)
        x = self.attention1(x)

        x = self.dense_block2(x)
        x = self.transition2(x)

        x = self.avg_pool(x)

        x = self.multi_scale_block(x)

        x = self.inception_block(x)
        x = self.attention2(x)

        x = self.gated_residual_block(x)
        x = self.attention3(x)

        x = F.hardswish(self.final_bn(self.final_conv(x)))
        x = self.attention4(x)

        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.top_layer(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss, on_epoch=True)
        self.log("test_acc", acc, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
            optimizer, mode="min", factor=0.1, patience=10, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss",
        }


def train_gurunet(num_classes=5, batch_size=32, max_epochs=100):
    data_module = PETDataModule("./data", batch_size=batch_size)

    model = GuruNet(num_classes)
    logger = TensorBoardLogger("./lightning_logs/GuruNet/")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/GuruNet/version_{logger.version}",
        filename="gurunet-{epoch:02d}-{val_loss:.5f}-{val_acc:5f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        verbose=True,
    )

    early_stop_callback = EarlyStopping(monitor="val_loss", patience=15, mode="min")

    lr_monitor = LearningRateMonitor(logging_interval="epoch", log_momentum=True)
    trainer = pl.Trainer(
        accelerator="gpu",
        max_epochs=max_epochs,
        callbacks=[
            checkpoint_callback,
            early_stop_callback,
            lr_monitor,
        ],
        logger=logger,
        precision="16-mixed",
        enable_progress_bar=True,
        enable_checkpointing=True,
        accumulate_grad_batches=4,
        profiler="simple",
        min_epochs=50,
    )

    trainer.fit(model=model, datamodule=data_module)


if __name__ == "__main__":
    train_gurunet()
