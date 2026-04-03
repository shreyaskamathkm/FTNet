"""Tests for dataloader data attribute dataclasses."""

from ftnet.core.dataloaders.data_attributes import ImageSize, ImageSizes


class TestImageSize:
    def test_stores_dimensions(self):
        size = ImageSize(width=640, height=480)
        assert size.width == 640
        assert size.height == 480

    def test_equality(self):
        assert ImageSize(100, 200) == ImageSize(100, 200)
        assert ImageSize(100, 200) != ImageSize(200, 100)


class TestImageSizes:
    def test_stores_base_and_crop(self):
        base = ImageSize(width=640, height=480)
        crop = ImageSize(width=512, height=512)
        sizes = ImageSizes(base_size=base, crop_size=crop)
        assert sizes.base_size == base
        assert sizes.crop_size == crop

    def test_fields_are_accessible(self):
        sizes = ImageSizes(
            base_size=ImageSize(320, 240),
            crop_size=ImageSize(256, 256),
        )
        assert sizes.base_size.width == 320
        assert sizes.crop_size.height == 256
