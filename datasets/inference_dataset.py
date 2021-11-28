from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils
import numpy as np
import cv2


class InferenceDataset(Dataset):

	def __init__(self, root, opts, transform=None, preprocess=None, parse_net=None):
		self.paths = sorted(data_utils.make_dataset(root))
		self.transform = transform
		self.preprocess = preprocess
		self.opts = opts
		self.parse_net = parse_net

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		from_im = cv2.imread(from_path)
		input_im = self.preprocess(from_path, self.parse_net)
		input_im = self.transform(input_im)
		return np.asarray(from_im), np.asarray(input_im)
