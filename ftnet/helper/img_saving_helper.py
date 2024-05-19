# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# from utils.visualize import get_color_palette


# def plot_confusion_matrix(
#     save_dir: Path, confusion_matrix: np.ndarray, class_names: list[str]
# ) -> None:
#     cmn = confusion_matrix / confusion_matrix.sum(1)[:, None]
#     cmn[np.isnan(cmn)] = 0

#     # name = getattr(self.trainer, "test_name", "Final")

#     fig, _ = plt.subplots(figsize=(15, 15))
#     sns.heatmap(
#         cmn,
#         annot=True,
#         fmt=".1f",
#         xticklabels=class_names,
#         yticklabels=class_names,
#     )
#     plt.ylabel("Actual")
#     plt.xlabel("Predicted")
#     fig.savefig(
#         save_dir / f"{name}_confusion_matrix.png",
#         bbox_inches="tight",
#     )
#     plt.close()


# def save_images(
#     original: np.ndarray,
#     groundtruth: np.ndarray,
#     prediction: np.ndarray,
#     filename: List[str],
# ) -> None:
#     calframe = inspect.getouterframes(inspect.currentframe(), 2)[1][3]
#     if calframe == "validation_step":
#         base_path = Path(self.seg_dir) / str(self.current_epoch)
#         std = self.val_dataset.std
#         mean = self.val_dataset.mean
#     elif calframe == "test_step":
#         base_path = self.seg_dir
#         std = self.test_dataset.std
#         mean = self.test_dataset.mean
#     else:
#         raise ValueError("Standard Deviation and Mean not found")

#     original_img = np.clip(np.moveaxis(original, 1, 3) * std + mean, a_min=0, a_max=1)

#     if not base_path.exists():
#         base_path.mkdir(parents=True, mode=0o770, exist_ok=True)

#     if self.args.save_images_as_subplots:
#         for i in range(original_img.shape[0]):
#             fig = plt.figure(figsize=(8.5, 11))
#             plt.subplot(1, 3, 1)
#             plt.imshow(original_img[i])
#             plt.subplot(1, 3, 2)
#             plt.imshow(np.array(get_color_palette(groundtruth[i], self.args.dataset)))
#             plt.subplot(1, 3, 3)
#             plt.imshow(np.array(get_color_palette(prediction[i], self.args.dataset)))
#             plt.axis("off")
#             fig.savefig(
#                 base_path / f"{Path(filename[i]).stem}.png",
#                 bbox_inches="tight",
#             )
#             plt.close()
#     else:
#         for i in range(original_img.shape[0]):
#             plt.imsave(
#                 base_path / f"Pred_{Path(filename[i]).stem}.png",
#                 np.array(get_color_palette(prediction[i], self.args.dataset)),
#             )


# def save_edge_images(
#     self,
#     original: np.ndarray,
#     groundtruth: np.ndarray,
#     prediction: np.ndarray,
#     edge_map: np.ndarray,
#     filename: List[str],
# ) -> None:
#     calframe = inspect.getouterframes(inspect.currentframe(), 2)[1][3]
#     if calframe == "validation_step":
#         base_path = Path(self.seg_dir) / str(self.current_epoch)
#         std = self.val_dataset.std
#         mean = self.val_dataset.mean
#     elif calframe == "test_step":
#         base_path = self.seg_dir
#         std = self.test_dataset.std
#         mean = self.test_dataset.mean
#     else:
#         raise ValueError("Standard Deviation and Mean not found")

#     original_img = np.clip(np.moveaxis(original, 1, 3) * std + mean, a_min=0, a_max=1)

#     if not base_path.exists():
#         base_path.mkdir(parents=True, mode=0o770, exist_ok=True)

#     if self.args.save_images_as_subplots:
#         for i in range(original_img.shape[0]):
#             fig = plt.figure(figsize=(8.5, 11))
#             plt.subplot(1, 4, 1)
#             plt.imshow(original_img[i])
#             plt.subplot(1, 4, 2)
#             plt.imshow(np.array(get_color_palette(groundtruth[i], self.args.dataset)))
#             plt.subplot(1, 4, 3)
#             plt.imshow(np.array(get_color_palette(prediction[i], self.args.dataset)))
#             plt.subplot(1, 4, 4)
#             plt.imshow(np.array(edge_map[i][0]))
#             plt.axis("off")
#             fig.savefig(
#                 base_path / f"{Path(filename[i]).stem}.png",
#                 bbox_inches="tight",
#             )
#             plt.close()
#     else:
#         for i in range(original_img.shape[0]):
#             plt.imsave(
#                 base_path / f"Pred_{Path(filename[i]).stem}.png",
#                 np.array(get_color_palette(prediction[i], self.args.dataset)),
#             )
#             plt.imsave(
#                 base_path / f"Edges_{Path(filename[i]).stem}.png",
#                 np.array(edge_map[i][0]),
#             )
