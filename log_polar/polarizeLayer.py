import torch
image = torch.randn(4,1,28,28)
polar = torch.randn(4,3,28,28)
center = torch.FloatTensor([image.shape[3]/2,image.shape[2]/2])

for batch in range(image.shape[0]):
	for y in range(image.shape[2]):
		for x in range(image.shape[3]):
			Cart_coord = torch.FloatTensor([x - center[0], -(y - center[1])])
			rho = torch.mul(Cart_coord,Cart_coord).sum(-1).sqrt().log()
			theta = torch.atan2(Cart_coord[1],Cart_coord[0])
			polar[batch,0,y,x] = image[batch,0,y,x]
			polar[batch,1,y,x] = rho
			polar[batch,2,y,x] = theta