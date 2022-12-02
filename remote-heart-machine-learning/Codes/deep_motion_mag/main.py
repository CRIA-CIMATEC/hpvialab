import argparse
import json
import os
import tensorflow as tf
import setproctitle
from configobj import ConfigObj
from validate import Validator
from magnet import MagNet3Frames

import sys
sys.path.append('../artificial_dataset_creator')

from utils import Dataset, make_dir, check_video_pattern, print_log, MRNirp
from datetime import datetime
import logging
import shutil

parser = argparse.ArgumentParser(description='')
parser.add_argument('--phase', dest='phase', default='train',
					help='train, test, run, interactive')
parser.add_argument('--config_file', dest='config_file',
					help='path to config file')
parser.add_argument('--config_spec', dest='config_spec',
					default='configs/configspec.conf',
					help='path to config spec file')
# for inference
parser.add_argument('--vid_dir', dest='vid_dir', default=None,
					help='Video folder to run the network on.')
parser.add_argument('--frame_ext', dest='frame_ext', default='png',
					help='Video frame file extension.')
parser.add_argument('--out_dir', dest='out_dir', default=None,
					help='Output folder of the video run.')
parser.add_argument('--amplification_factor', dest='amplification_factor',
					type=float, default=5,
					help='Magnification factor for inference.')
parser.add_argument('--velocity_mag', dest='velocity_mag', action='store_true',
					help='Whether to do velocity magnification.')
# For temporal operation.
parser.add_argument('--fl', dest='fl', type=float,
					help='Low cutoff Frequency.')
parser.add_argument('--fh', dest='fh', type=float,
					help='High cutoff Frequency.')
parser.add_argument('--fs', dest='fs', type=float,
					help='Sampling rate.')
parser.add_argument('--n_filter_tap', dest='n_filter_tap', type=int,
					help='Number of filter tap required.')
parser.add_argument('--filter_type', dest='filter_type', type=str,
					help='Type of filter to use, must be Butter or FIR.')

arguments = parser.parse_args()

def main(args):
	configspec = ConfigObj(args.config_spec, raise_errors=True)
	config = ConfigObj(args.config_file,
					   configspec=configspec,
					   raise_errors=True,
					   file_error=True)
	# Validate to get all the default values.
	config.validate(Validator())
	if not os.path.exists(config['exp_dir']):
		# checkpoint directory.
		os.makedirs(os.path.join(config['exp_dir'], 'checkpoint'))
		# Tensorboard logs directory.
		os.makedirs(os.path.join(config['exp_dir'], 'logs'))
		# default output directory for this experiment.
		os.makedirs(os.path.join(config['exp_dir'], 'sample'))
	network_type = config['architecture']['network_arch']
	exp_name = config['exp_name']
	setproctitle.setproctitle('{}_{}_{}' \
							  .format(args.phase, network_type, exp_name))
	tfconfig = tf.ConfigProto(allow_soft_placement=True,
							  log_device_placement=False)
	tfconfig.gpu_options.allow_growth = True

	with tf.Session(config=tfconfig) as sess:
		model = MagNet3Frames(sess, exp_name, config['architecture'])
		checkpoint = config['training']['checkpoint_dir']
		if args.phase == 'train':
			train_config = config['training']
			if not os.path.exists(train_config['checkpoint_dir']):
				os.makedirs(train_config['checkpoint_dir'])
			model.train(train_config)
		elif args.phase == 'run':
			model.run(checkpoint,
					  args.vid_dir,
					  args.frame_ext,
					  args.out_dir,
					  args.amplification_factor,
					  args.velocity_mag)
		elif args.phase == 'run_temporal':
			model.run_temporal(checkpoint,
							   args.vid_dir,
							   args.frame_ext,
							   args.out_dir,
							   args.amplification_factor,
							   args.fl,
							   args.fh,
							   args.fs,
							   args.n_filter_tap,
							   args.filter_type)
		else:
			raise ValueError('Invalid phase argument. '
							 'Expected ["train", "run", "run_temporal"], '
							 'got ' + args.phase)

def magnify_dataset(dataset: Dataset, out_dataset_path: str, base_dataset_path: str, video_name_pattern='Frame%05d.pgm'):
	"""Funcao que recebe a estrutura de pasta para processar cada video com a funcao process_video, \
		produzindo dessa forma o dataset artificial (cada video com as ROIs cortadas, processadas com estabilizadores, etc).

	Um JSON estruturado por sujeito e pasta interna vai salvar o caminho de cada video, assim como o caminho para seu PPG.
	Um log com todos os prints e ocorrencias de exceptions vai ser salvo na pasta final do dataset.
	Cada video vai receber os mesmos parametros de processamento:
	- FaceMesh como detector facial;
	- Threshold como estabilizador:
		- Regiao da boca: replace method
		- Olho direito: mean method
		- Olho esquerdo: mean method

	- O padrao (video_name_pattern) so pode ter uma string com o seguinte formatador (%d ou nenhum):
	- (1) 'Frame%05d.pgm' no qual sera feito: pattern % 0

	- O objeto dataset deve pertencer a uma classe que herda da classe Dataset
	- A classe do objeto dataset deve ter implementado os metodos save_video e ppg_flatten

	- Todos os caminhos vao ser verificados durante o processo, um erro vai ser emitido sem prejudicar \
	os outros processos caso eles nao existam.
	- O diretorio out_dataset_path nao deve existir (para nao sobrescrever a pasta anterior).

	Keyword arguments:
	- dataset: Dataset -- Objeto que pertence a uma classe que herda da classe Dataset e implementa todos os seus metodos.\
	dataset.paths: Iterable -- Lista de dicionarios que contem a seguinte estrutura:\n
	[
		{
		'subject_folder_name': 'Nome do sujeito ou da pasta do sujeito', 
		'subdir_folder_name': 'Nome da pasta interna que contem o video do sujeito', 
		'video_path': 'Caminho relativo ou absoluto do video com o nome do arquivo',
		'ppg_path': 'Caminho relativo ou absoluto do PPG em formato .mat com o nome do arquivo'
		},
		...
	]
	- out_dataset_path: str -- Caminho da pasta na qual deseja a estrutura final do dataset output (artificial).
	- video_name_pattern: str -- Padrao do nome da sequencia de imagens; ex.: 'Frame%05d.pgm' (%05d diz \
	que a sequencia vai ter cinco digitos, sendo que os primeiros vao ser preenchidos com zeros: Frame00000.pgm, \
		Frame00001.pgm, Frame00002.pgm, ...)

	Return: 
	None
	"""

	# pasta do dataset artificial nao pode existir
	assert os.path.isdir(out_dataset_path) == False, f"out_dataset_path ja existe: {out_dataset_path}"
	assert issubclass(dataset.__class__, Dataset), "O objeto dataset deve pertencer a uma classe que herda da classe Dataset"

	os.mkdir(out_dataset_path)

	log_file = open(os.path.join(out_dataset_path, "dataset_log.txt"), "w")
	print_log(f"""\rCaminho do dataset artificial: {out_dataset_path}
			\rPadrao do nome dos videos: {video_name_pattern}\n""", log_file)

	print_log("Videos que vao ser processados:", log_file)
	[print_log(path['video_path'], log_file) for path in dataset.paths]

	exceptions_count = 0
	arguments.phase = 'run'
	arguments.config_file = './configs/o3f_hmhm2_bg_qnoise_mix4_nl_n_t_ds3.conf'
	arguments.fl = 0.75
	arguments.fh = 4.0
	arguments.fs = 30
	arguments.velocity_mag = False
	arguments.amplification_factor = 5

	# para cada video que precisa ser processado
	for path in dataset.paths:
		subject_folder_name = path['subject_folder_name']
		subdir_folder_name = path['subdir_folder_name']
		video_path = path['video_path']

		print_log(f'\n{"=" * 10} Video {subject_folder_name}_{subdir_folder_name} {"=" * 10}', log_file)
		out_path = os.path.join(out_dataset_path, subject_folder_name, subdir_folder_name)
		make_dir(out_path, log_file)

		arguments.vid_dir = os.path.dirname(video_path)
		arguments.frame_ext = video_path.split('.')[-1]
		arguments.out_dir = out_path
		
		print_log(f"\nProcessando video {subject_folder_name}_{subdir_folder_name} com os argumentos:", log_file)
		# [print_log(f'{key}: {value}', log_file) for key, value in kwargs.items()]
		[print_log(json.dumps(arguments.__dict__, indent=4), log_file)]

		shutil.copyfile(
			src=os.path.join(os.path.dirname(arguments.vid_dir), "pulseOx_bpm.mat"),
			dst=os.path.join(os.path.dirname(arguments.out_dir), "pulseOx_bpm.mat")
		)

		try:
			assert check_video_pattern(video_path, video_name_pattern, log_file), f"O seguinte video_path nao foi encontrado: {video_path}"
			main(arguments)
		except Exception as e:
			print_log(f"\nException nao mapeada: \n{logging.traceback.format_exc()}", log_file)
			exceptions_count += 1

	with open(os.path.join(base_dataset_path, "dataset_info.json"), 'r') as fp:
		dataset_json = json.load(fp)

	for subject in dataset_json.keys():
		for key, value in dataset_json[subject].items():
			if isinstance(value, list):
				for i, path in enumerate(value):
					dataset_json[subject][key][i] = path.replace(
						os.path.basename(os.path.realpath(base_dataset_path)), 
						os.path.basename(os.path.realpath(out_dataset_path)),
					)
			elif isinstance(value, str):
				dataset_json[subject][key] = value.replace(
					os.path.basename(os.path.realpath(base_dataset_path)), 
					os.path.basename(os.path.realpath(out_dataset_path)),
				)
	
	with open(os.path.join(out_dataset_path, "dataset_info.json"), 'w') as fp:
		json.dump(dataset_json, fp, indent=1)

	print_log(f"\nQuantidade de Exceptions: {exceptions_count}", log_file)
	print_log(f"\nFim da execucao: {datetime.now()}", log_file)
	log_file.close()

if __name__ == '__main__':
	# main(arguments)

	dataset = MRNirp(image_extension='png')

	rois = ['bottom_face', 'right_eye', 'left_eye', 'middle_face']

	# para cada sujeito
	for i in range(1, 9):
		for roi in rois:
			dataset.paths.append({
				'subject_folder_name': f'Subject{i}_still_940', 
				'subdir_folder_name': f'RGB_demosaiced/{roi}', 
				'video_path': f'/home/victorrocha/scratch/desafio2_2021/Datasets/artificial_v2/Subject{i}_still_940/RGB_demosaiced/{roi}/Frame%05d.{dataset.image_extension}',
				'ppg_path': ''
			})

	magnify_dataset(
		dataset=dataset,
		out_dataset_path='/home/victorrocha/scratch/desafio2_2021/Datasets/artificial_v3',
		base_dataset_path='/home/victorrocha/scratch/desafio2_2021/Datasets/artificial_v2',
		video_name_pattern='Frame%05d.' + dataset.image_extension
	)