import os.path as osp
import math
import time
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torchmetrics import F1, Precision, Recall
from BlinkFormer import BlinkFormer, BlinkFormer_with_BSE_head
from optimizer import build_optimizer

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, base_lr, objective, min_lr=5e-5, last_epoch=-1):
	""" Create a schedule with a learning rate that decreases following the
	values of the cosine function between 0 and `pi * cycles` after a warmup
	period during which it increases linearly between 0 and base_lr.
	"""
	# step means epochs here
	def lr_lambda(current_step):
		current_step += 1
		if current_step <= num_warmup_steps:
			return float(current_step) / float(max(1, num_warmup_steps)) # * base_lr 
		progress = min(float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps)), 1)
		if objective == 'mim':
			return 0.5 * (1. + math.cos(math.pi * progress))
		else:
			factor = 0.5 * (1. + math.cos(math.pi * progress))
			return factor*(1 - min_lr/base_lr) + min_lr/base_lr

	return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

class VideoClassificaiton(pl.LightningModule):

	def __init__(self,
				 configs,
				 trainer,
				 ckpt_dir,
				 do_eval,
				 do_test):
		super().__init__()
		self.configs = configs
		self.trainer = trainer

		if self.configs.arch == 'BlinkFormer':
			self.model = BlinkFormer(num_classes=2, dim = self.configs.dim, depth=self.configs.depth, heads=self.configs.heads, mlp_dim=self.configs.mlp_dim)
		elif self.configs.arch == 'BlinkFormer_with_BSE_head':
			self.model = BlinkFormer_with_BSE_head(num_classes=2, dim = self.configs.dim, depth=self.configs.depth, heads=self.configs.heads, mlp_dim=self.configs.mlp_dim)
		else:
			raise NotImplementedError
		
		self.max_F1 = 0
		self.train_F1 = F1(multiclass=False)
		self.loss_fn = nn.CrossEntropyLoss()
		self.regression_loss = nn.MSELoss()

		# common
		self.iteration = 0
		self.data_start = 0
		self.ckpt_dir = ckpt_dir
		self.do_eval = do_eval
		self.do_test = do_test
		if self.do_eval:
			self.val_F1 = F1(multiclass=False)
			self.val_Precision = Precision(multiclass=False)
			self.val_Recall = Recall(multiclass=False)
		if self.do_test:
			self.test_F1 = F1(multiclass=False)
			self.test_Precision = Precision(multiclass=False)
			self.test_Recall = Recall(multiclass=False)

	def configure_optimizers(self):
		optimizer = build_optimizer(self.configs, self)
		
		lr_scheduler = None
		lr_schedule = self.configs.lr_schedule
		if lr_schedule == 'multistep':
			lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, 
														  milestones=[5, 11],
														  gamma=0.1)
		elif lr_schedule == 'cosine':
			lr_scheduler = get_cosine_schedule_with_warmup(optimizer, 
														  num_warmup_steps=self.configs.warmup_epochs, 
														  num_training_steps=self.trainer.max_epochs,
														  base_lr=self.configs.lr,
														  min_lr=self.configs.min_lr,
														  objective=self.configs.objective)
		return [optimizer], [lr_scheduler]

	def parse_batch(self, batch, train):
		inputs, labels, frames_orig, img_folder, anno = *batch,
		return inputs, labels, frames_orig, img_folder, anno

	# epoch schedule
	def _get_momentum(self, base_value, final_value):
		return final_value - (final_value - base_value) * (math.cos(math.pi * self.trainer.current_epoch / self.trainer.max_epochs) + 1) / 2

	def _weight_decay_update(self):
		for i, param_group in enumerate(self.optimizers().optimizer.param_groups):
			if i == 1:  # only the first group is regularized
				param_group["weight_decay"] = self._get_momentum(base_value=self.configs.weight_decay, final_value=self.configs.weight_decay_end)

	def clip_gradients(self, clip_grad, norm_type=2):
		layer_norm = []
		model_wo_ddp = self.module if hasattr(self, 'module') else self
		for name, p in model_wo_ddp.named_parameters():
			if p.grad is not None:
				param_norm = torch.norm(p.grad.detach(), norm_type)
				layer_norm.append(param_norm)
				if clip_grad:
					clip_coef = clip_grad / (param_norm + 1e-6)
					if clip_coef < 1:
						p.grad.data.mul_(clip_coef)
		total_grad_norm = torch.norm(torch.stack(layer_norm), norm_type)
		return total_grad_norm

	def get_progress_bar_dict(self):
		# don't show the version number
		items = super().get_progress_bar_dict()
		items.pop("v_num", None)
		return items

	# Trainer Pipeline
	def training_step(self, batch, batch_idx):
		inputs, labels, _, _, eye_annos = self.parse_batch(batch, train=True)
		if self.configs.arch == "BlinkFormer_with_BSE_head":
			preds, reg_preds = self.model(inputs)
			softmax_preds = torch.softmax(preds, dim=1)
			eye_annos = eye_annos.float()
			alpha = 0.1
			loss = alpha * self.loss_fn(softmax_preds, labels) +(1-alpha)*self.regression_loss(reg_preds, eye_annos)
		else:
			preds = self.model(inputs)
			softmax_preds = torch.softmax(preds, dim=1)	
			loss = self.loss_fn(softmax_preds, labels)

		self.log("train/loss", loss)
		self.train_F1(preds.softmax(dim=-1), labels)
		return {'loss': loss}
	
	def on_after_backward(self):
		param_norms = self.clip_gradients(self.configs.clip_grad)
		self._weight_decay_update()
		lr = self.optimizers().optimizer.param_groups[0]['lr']
		self.log("train/lr", lr)

	def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
		optimizer_closure, on_tpu, using_native_amp, using_lbfgs):
		optimizer.step(closure=optimizer_closure)
		self.iteration += 1

	def training_epoch_end(self, outputs):
		timestamp = time.strftime('%Y-%m-%d---%H-%M-%S', time.localtime())

		mean_F1 = self.train_F1.compute()
		self.print(f'{timestamp} - Train',
					"[Epoch {}]:".format(self.trainer.current_epoch), 
					f'F1:{mean_F1:.3f}')
		self.log("train/F1", mean_F1)
		self.train_F1.reset()

	def validation_step(self, batch, batch_indx):
		if self.do_eval:
			inputs, labels, frames_orig, img_folder, anno = self.parse_batch(batch, train=False)
			preds = self.model(inputs)
			if self.configs.arch == "BlinkFormer_with_BSE_head":
				video_cls, _ = preds
			else:
				video_cls = preds
			self.val_F1(video_cls.softmax(dim=-1), labels)
			self.val_Precision(video_cls.softmax(dim=-1), labels)
			self.val_Recall(video_cls.softmax(dim=-1), labels)
	
	def validation_epoch_end(self, outputs):
		if self.do_eval:
			mean_F1 = self.val_F1.compute()
			mean_Precision = self.val_Precision.compute()
			mean_Recall = self.val_Recall.compute()

			timestamp = time.strftime('%Y-%m-%d---%H-%M-%S', time.localtime())
			self.print(f'{timestamp} -                         Val',
					   f'F1:{mean_F1:.3f} ', 
					   f'Precision:{mean_Precision:.3f} ',
					   f'Recall:{mean_Recall:.3f} ')
			
			self.log("val/F1", mean_F1)
			self.log("val/Precision", mean_Precision)
			self.log("val/Recall", mean_Recall)

			self.val_F1.reset()
			self.val_Precision.reset()
			self.val_Recall.reset()

			# save best checkpoint
			if self.trainer.current_epoch > 2 and mean_F1 > self.max_F1:
				save_path = osp.join(self.ckpt_dir,
									 f'{timestamp}_'+
									 f'ep_{self.trainer.current_epoch}_'+
									 f'F1_{mean_F1:.3f}.pth')
				self.trainer.save_checkpoint(save_path)
				self.print(f"save best checkpoint to {save_path}")
				self.max_F1 = mean_F1

			
	def test_step(self, batch, batch_idx):
		if self.do_test:
			inputs, labels, frames_orig, img_folder, anno  = self.parse_batch(batch, train=False)
			preds = self.model(inputs)
			if self.configs.arch == "BlinkFormer_with_BSE_head":
				video_cls, _ = preds
			else:
				video_cls = preds
			
			self.test_F1(video_cls.softmax(dim=-1), labels)
			self.test_Precision(video_cls.softmax(dim=-1), labels)
			self.test_Recall(video_cls.softmax(dim=-1), labels)
	
	def test_epoch_end(self, outputs):
		if self.do_test:
			mean_F1 = self.test_F1.compute()
			mean_Precision = self.test_Precision.compute()
			mean_Recall = self.test_Recall.compute()

			timestamp = time.strftime('%Y-%m-%d---%H-%M-%S', time.localtime())
			self.print(f'{timestamp} - Test',
					   f'mean_F1:{mean_F1:.6f}',
					   f'mean_Precision:{mean_Precision:.6f}',
					   f'mean_Recall:{mean_Recall:.6f}')

			self.log("test/F1", mean_F1)
			self.log("test/Precision", mean_Precision)
			self.log("test/Recall", mean_Recall)

			self.test_F1.reset()
			self.test_Precision.reset()
			self.test_Recall.reset()
