import tensorflow_datasets as tfds
import tensorflow as tf
import os

LANDMARK_HEADINGS = ("lefteye_x lefteye_y righteye_x righteye_y "
                     "nose_x nose_y leftmouth_x leftmouth_y rightmouth_x "
                     "rightmouth_y").split()
ATTR_HEADINGS = (
    "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs "
    "Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair "
    "Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair "
    "Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache "
    "Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline "
    "Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings "
    "Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
).split()

_CITATION = """\
@inproceedings{conf/iccv/LiuLWT15,
  added-at = {2018-10-09T00:00:00.000+0200},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  biburl = {https://www.bibsonomy.org/bibtex/250e4959be61db325d2f02c1d8cd7bfbb/dblp},
  booktitle = {ICCV},
  crossref = {conf/iccv/2015},
  ee = {http://doi.ieeecomputersociety.org/10.1109/ICCV.2015.425},
  interhash = {3f735aaa11957e73914bbe2ca9d5e702},
  intrahash = {50e4959be61db325d2f02c1d8cd7bfbb},
  isbn = {978-1-4673-8391-2},
  keywords = {dblp},
  pages = {3730-3738},
  publisher = {IEEE Computer Society},
  timestamp = {2018-10-11T11:43:28.000+0200},
  title = {Deep Learning Face Attributes in the Wild.},
  url = {http://dblp.uni-trier.de/db/conf/iccv/iccv2015.html#LiuLWT15},
  year = 2015
}
"""

_DESCRIPTION = """\
CelebFaces Attributes Dataset (CelebA) is a large-scale face attributes dataset\
 with more than 200K celebrity images, each with 40 attribute annotations. The \
images in this dataset cover large pose variations and background clutter. \
CelebA has large diversities, large quantities, and rich annotations, including\
 - 10,177 number of identities,
 - 202,599 number of face images, and
 - 5 landmark locations, 40 binary attributes annotations per image.
The dataset can be employed as the training and test sets for the following \
computer vision tasks: face attribute recognition, face detection, and landmark\
 (or facial part) localization.
"""




class CelebAFixed(tfds.image.celeba.CelebA):
  """CelebA dataset. Aligned and cropped. With metadata."""

  VERSION = tfds.core.Version(
      "2.0.1", "New split API (https://tensorflow.org/datasets/splits)")
  SUPPORTED_VERSIONS = [
      tfds.core.Version("2.0.0"),
  ]

  def _info(self):
    features = {
            "image":
                tfds.features.Image(
                    shape=(218, 178, 3), encoding_format="jpeg")}
    features.update({name: tfds.features.ClassLabel(num_classes=2) for name in ATTR_HEADINGS})
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(features),
        homepage="http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html",
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    downloaded_dirs = dl_manager.download({
        "img_align_celeba": 'https://drive.google.com/u/0/uc?export=download&confirm=P5cN&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM',
        "list_eval_partition": "https://drive.google.com/uc?export=download&id=1SB57NXZWgi75EH0e_KCTEQXzEP5QUz69",
        "list_attr_celeba": "https://drive.google.com/uc?export=download&id=1u-7-cQAgL38lAIZb0WqPZZlhtnAF99jO",
        "landmarks_celeba": "https://drive.google.com/uc?export=download&id=1h1qDs6I_5r-5eBA69w370ulFyC7zgHkB",
    })

    # Load all images in memory (~1 GiB)
    # Use split to convert: `img_align_celeba/000005.jpg` -> `000005.jpg`
    all_images = {
        k.split("/")[-1]: img for k, img in
        dl_manager.iter_archive(downloaded_dirs["img_align_celeba"])
    }

    return [
        tfds.core.SplitGenerator(
            name=tfds.Split.TRAIN,
            gen_kwargs={
                "file_id": 0,
                "downloaded_dirs": downloaded_dirs,
                "downloaded_images": all_images,
            }),
        tfds.core.SplitGenerator(
            name='valid',
            gen_kwargs={
                "file_id": 1,
                "downloaded_dirs": downloaded_dirs,
                "downloaded_images": all_images,
            }),
        tfds.core.SplitGenerator(
            name=tfds.Split.TEST,
            gen_kwargs={
                "file_id": 2,
                "downloaded_dirs": downloaded_dirs,
                "downloaded_images": all_images,
            })
        ,
        tfds.core.SplitGenerator(
            name='extra',
            gen_kwargs={
                "file_id": 3,
                "downloaded_dirs": downloaded_dirs,
                "downloaded_images": all_images,
            })
    ]

  def _process_celeba_config_file(self, file_path):
    """Unpack the celeba config file.
    The file starts with the number of lines, and a header.
    Afterwards, there is a configuration for each file: one per line.
    Args:
      file_path: Path to the file with the configuration.
    Returns:
      keys: names of the attributes
      values: map from the file name to the list of attribute values for
              this file.
    """
    with tf.io.gfile.GFile(file_path) as f:
      data_raw = f.read()
    lines = data_raw.split("\n")

    keys = lines[1].strip().split()
    values = {}
    # Go over each line (skip the last one, as it is empty).
    for line in lines[2:-1]:
      row_values = line.strip().split()
      # Each row start with the 'file_name' and then space-separated values.
      values[row_values[0]] = [int(v) for v in row_values[1:]]
    return keys, values

  def _generate_examples(self, file_id, downloaded_dirs, downloaded_images):
    """Yields examples."""

    img_list_path = downloaded_dirs["list_eval_partition"]
    landmarks_path = downloaded_dirs["landmarks_celeba"]
    attr_path = downloaded_dirs["list_attr_celeba"]


    first_10000 = file_id == 0
    if file_id == 3:
        file_id = 0

    with tf.io.gfile.GFile(img_list_path) as f:
      files = [
          line.split()[0]
          for (lid, line) in enumerate(f.readlines())
          if int(line.split()[1]) == file_id and
             (file_id != 0 or
              ((first_10000 and lid < 10000) or ((not first_10000) and lid >= 10000))
             )
      ]

    attributes = self._process_celeba_config_file(attr_path)
    landmarks = self._process_celeba_config_file(landmarks_path)

    for file_name in sorted(files):
      record = {
          "image": downloaded_images[file_name]
               }
      record.update({
              k: int(v > 0) for k, v in zip(attributes[0], attributes[1][file_name])
      })
      yield file_name, record
