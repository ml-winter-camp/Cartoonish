import argparse
import PIL
import os
import numpy as np
import torch
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import torch.nn as nn

from unet import UNet
from discriminator import SimpleNet

parser = argparse.ArgumentParser(description='PyTorch cycle_face')
parser.add_argument('--real_path', default = '/home/yylovehjr/winter-camp-pek/gan/GANface/test_pic')
parser.add_argument('--cart_path', default = '/home/yylovehjr/winter-camp-pek/gan/data/cartoon/cartoonset100k')
parser.add_argument('--lr', default = 0.001)
parser.add_argument('--momentum', default = 0.9)
parser.add_argument('--max_iter', default = 10000)
parser.add_argument('--num_batch', default = 1)
parser.add_argument('--batch_size', default = 32)
parser.add_argument('--save_folder', default = 'ckpt')
parser.add_argument('--img_size', default = 500)
parser.add_argument('--save_freq', default = 1)
parser.add_argument('--Gpretrained', default = 'ckpt/5/G.weight')
parser.add_argument('--Dpretrained', default = None)

def main():

	args = parser.parse_args()

	transform = transforms.Compose([
			transforms.Resize(args.img_size),
			transforms.CenterCrop(args.img_size),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
		])

	transform_test = transforms.Compose([
			transforms.Resize(args.img_size),
			transforms.ToTensor(),
			transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
		])

	# Prepare Dataset
	real_dataset = datasets.ImageFolder(args.real_path, transform = transform)
	real_loader = DataLoader(
        real_dataset, batch_size=args.batch_size, shuffle=True)

	cart_dataset = datasets.ImageFolder(args.cart_path, transform = transform)
	cart_loader = DataLoader(
		cart_dataset, batch_size=args.batch_size, shuffle=True)

	print('Read Dataset Done!')

	G=UNet(3,3)
	if args.Gpretrained is not None:
		G = torch.load(args.Gpretrained)
	D=SimpleNet()
	if args.Dpretrained is not None:
		D = torch.load(args.Dpretrained)

	G.cuda()
	D.cuda()

	criterion=nn.MSELoss()

	optimizer_gen = torch.optim.SGD(G.parameters(), args.lr,
			                    momentum=args.momentum)

	optimizer_dis = torch.optim.SGD(D.parameters(), args.lr,
                                momentum=args.momentum)

	tb_logger = None

	# train(G,D,real_loader,cart_loader,optimizer_gen,optimizer_dis,args,criterion, tb_logger)
	test(G,D,real_loader,cart_loader,optimizer_gen,optimizer_dis,args,criterion, tb_logger)


def train(G,D,real_loader,cart_loader,optimizer_gen,optimizer_dis,args,
        criterion, tb_logger):
	G.train()
	D.train()

	# one = torch.FloatTensor([1]).cuda()
	# mone = one * -1
	true_labels = torch.ones(args.batch_size).cuda()
	fake_labels = torch.zeros(args.batch_size).cuda()

	for i in np.arange(args.max_iter):

		for p in D.parameters():
			p.requires_grad=True

		lossD_gen = []
		lossD_real = []
		lossG_gen = []
		lossG_real = []

		iter_cart = iter(cart_loader)
		# print(real_imgs.shape)

		for j, (real_imgs, _) in enumerate(real_loader):

			real_imgs = torch.autograd.Variable(real_imgs.cuda())

			gen_imgs = G(real_imgs)

			optimizer_dis.zero_grad()

			scoreD_gen=D(torch.autograd.Variable(gen_imgs.data))
			errorD_gen = criterion(scoreD_gen, fake_labels)
			# scoreD_gen.backward(torch.FloatTensor([1]).cuda())
			errorD_gen.backward()

			lossD_gen.append(scoreD_gen.data.clone())

			cart_imgs, _ = next(iter_cart)
			cart_imgs = torch.autograd.Variable(cart_imgs.cuda())
			
			scoreD_real=D(cart_imgs)
			errorD_real = criterion(scoreD_real, true_labels)
			
			# scoreD_real.backward(torch.FloatTensor([1]).cuda()*-1)
			errorD_real.backward()

			lossD_real.append(scoreD_real.data.clone())

			optimizer_dis.step()

			print(j)

			if j >= args.num_batch:
				break

		for p in D.parameters():
			p.requires_grad=False

		for j, (real_imgs, _) in enumerate(real_loader):

			real_imgs = torch.autograd.Variable(real_imgs.cuda())

			optimizer_gen.zero_grad()

			gen_imgs = G(real_imgs)

			scoreG_real = criterion(real_imgs, gen_imgs)
			scoreG_real.backward()

			gen_imgs = G(real_imgs)
			scoreG_gen = D(gen_imgs)
			errorG_gen = criterion(scoreG_gen, fake_labels)
			errorG_gen.backward()

			lossG_gen.append(scoreG_gen.data.clone())
			lossG_real.append(scoreG_real.data.clone())

			optimizer_gen.step()

			print(j)

			if j >= args.num_batch:
				break

		def compute_loss(x):
			x = sum(x) / len(x)
			try:
				x = sum(x) / len(x)
			except:
				x = x
			return float(x.data)
			
		lossD_gen_avg = compute_loss(lossD_gen)
		lossD_real_avg = compute_loss(lossD_real)
		lossG_gen_avg = compute_loss(lossG_gen)
		lossG_real_avg = compute_loss(lossG_real)

		print('iter: %d lossD: %.2f %.2f lossG: %.2f MSE: %.2f'%(
				i, lossD_gen_avg, lossD_real_avg, lossG_gen_avg, lossG_real_avg))

		if i%args.save_freq==0:

			save_path=args.save_folder+'/'+str(i)
			os.system('mkdir -p ' + save_path)

			torch.save(G, save_path+'/G.weight')
			torch.save(D, save_path+'/D.weight')


def transImg(img):
        img = img.squeeze(0)
        img = transforms.ToPILImage()(img)
        return img

def test(G,D,real_loader,cart_loader,optimizer_gen,optimizer_dis,args,criterion, tb_logger):

	G.eval()
	
	for i, (real_imgs, _) in enumerate(real_loader):
		if i >= 10:
			break
		save_path=args.save_folder+'_test/'
		real_imgs = torch.autograd.Variable(real_imgs.cuda())
		gen_imgs = G(real_imgs)
		for j in range(gen_imgs.size()[0]):
			filename=os.path.join(save_path,str(j))
			#cv2.imwrite(filename+'_gen.jpg',gen_imgs[j].data.cpu().numpy().transpose(1,2,0))
			gen_img = transImg(gen_imgs[j].cpu())
			gen_img.save(filename + '_gen.jpg')
			real_img = transImg(real_imgs[j].cpu())
			real_img.save(filename + '_real.jpg')
			# cv2.imwrite(filename+'_real.jpg',real_imgs[j].data.cpu().numpy().transpose(1,2,0))


if __name__ == '__main__':
	main()
