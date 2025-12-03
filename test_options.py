from argparse import ArgumentParser


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		
		self.parser.add_argument('--exp_dir', type=str, default='/data/home/zjw/show/1_output/infer_style_transformer', help='Path to experiment output directory')
		self.parser.add_argument('--checkpoint_path', default='/data/zjw/results/55_results_newse/train_style_transformer/checkpoints/best_model.pt', type=str, help='Path to pSp model checkpoint')
		# self.parser.add_argument('--data_path', type=str, default='/data/zjw/30000', help='Path to directory of images to evaluate')
		self.parser.add_argument('--data_path', type=str, default='/data/zjw/celeba-1024', help='Path to directory of images to evaluate')
		self.parser.add_argument('--couple_outputs', action='store_true', help='Whether to also save inputs + outputs side-by-side')
		self.parser.add_argument('--resize_outputs', action='store_true', help='Whether to resize outputs to 256x256 or keep at 1024x1024')

		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--test_workers', default=1, type=int, help='Number of test/inference dataloader workers')
		
		# 由于少参数暂时补充的
		self.parser.add_argument('--n_images', default=None, type=int)
		self.parser.add_argument('--n_styles', default=1, type=int)
		self.parser.add_argument('--resize_factors', type=str, default=None, help='Downsampling factor for super-res (should be a single value for inference).')

		self.parser.add_argument('-i', '--input_latent_codes_path1', type=str, default='/data/zjw/output/55_output_e/infer_style_transformer/latent_code/000016.npy',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (optional)')
		self.parser.add_argument('-j', '--input_latent_codes_path2', type=str, default='/data/zjw/output/55_output_e/infer_style_transformer/latent_code/000016.npy',
                      help='If specified, will load latent codes from given '
                           'path instead of randomly sampling. (optional)')
		#n_outputs_to_generate不需要，除非用random去混合图像
		self.parser.add_argument('--n_outputs_to_generate', type=int, default=5, help='Number of outputs to generate per input image.')
		self.parser.add_argument('--mix_alpha', type=float, default=1, help='Alpha value for style-mixing')
		self.parser.add_argument('--latent_mask', type=str, default='8,9,10,11,12,13,14,15,16,17', help='Comma-separated list of latents to perform style-mixing with')


	def parse(self):
		opts = self.parser.parse_args()
		return opts
