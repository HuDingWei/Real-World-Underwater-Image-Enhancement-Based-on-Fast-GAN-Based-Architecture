class Generator_DCE_compensation_2(nn.Module):
	"""
	MSG-DCE-GAN Generator
	"""
	def __init__(self,
		img_dim=256,
		patch_dim=16,
		embedding_dim=512,
		num_channels=3,
		num_heads=8,
		num_layers=4,
		hidden_dim=256,
		dropout_rate=0.0,
		attn_dropout_rate=0.0,
		in_ch=3, 
		out_ch=3,
		conv_patch_representation=True,
		positional_encoding_type="learned",
		use_eql=True):
		super(Generator_DCE_compensation_2, self).__init__()
		assert embedding_dim % num_heads == 0
		assert img_dim % patch_dim == 0

		self.out_ch=out_ch 					# Number of output channels
		self.in_ch=in_ch                         		# Number of input channels
		self.img_dim = img_dim   				# Input image dimension
		self.embedding_dim = embedding_dim  			# Dimension of patch embeddings (default 512)
		self.num_heads = num_heads  				# Number of attention heads in multi-head attention
		self.patch_dim = patch_dim  				# Size of each image patch
		self.num_channels = num_channels  			# Number of channels in the input image (3 for RGB)  
		self.dropout_rate = dropout_rate  			# Dropout rate applied to MLP and embedding layers 
		self.attn_dropout_rate = attn_dropout_rate  		# Dropout rate applied inside the attention module 
		self.conv_patch_representation = conv_patch_representation  #True

		self.num_patches = int((img_dim // patch_dim) ** 2)  	# Total number of patches in the image
		self.seq_length = self.num_patches  			# Sequence length for transformer = number of patches
		self.flatten_dim = 128 * num_channels  			# 128*3=384


		#layer Norm
		self.pre_head_ln = nn.LayerNorm(embedding_dim)

		if self.conv_patch_representation:

			self.Conv_x = nn.Conv2d(
				256,
				self.embedding_dim,  #512
				kernel_size=3,
				stride=1,
				padding=1
		    )

		self.bn = nn.BatchNorm2d(256)
		self.relu = nn.ReLU(inplace=True)



		#modulelist
		self.rgb_to_feature=ModuleList([from_rgb(32),from_rgb(64),from_rgb(128)])
		self.feature_to_rgb=ModuleList([to_rgb(32),to_rgb(64),to_rgb(128),to_rgb(256)])

		self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.Con1 = conv_block(self.in_ch, 32)
		self.Con2 = conv_block(32, 32)
		self.Con3 = conv_block(32, 32)


		self.Conv5 = conv_block(512,256)

		#self.Conv_x = conv_block(256,512)
		self.mtc = ChannelTransformer(channel_num=[32,64,128,256],patchSize=[32, 16, 8, 4])
								

		self.Conv = nn.Conv2d(32, self.out_ch, kernel_size=1, stride=1, padding=0)

        
		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 

		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        
		self.Con = nn.Conv2d(number_f,number_f,3,1,1,bias=True)
		self.Up = up_conv(32, 32)
        

     
	def reshape_output(self,x): # Resize the Transformer output to the original feature map size
		x = x.view(
			x.size(0),
			int(self.img_dim / self.patch_dim),
			int(self.img_dim / self.patch_dim),
			self.embedding_dim,
			)#B,16,16,512
		x = x.permute(0, 3, 1, 2).contiguous()

		return x

	def forward(self, x):
		#print(x.shape)

		enhance_img_1 = []
		enhance_img = []
		r_img = []

		e1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
		e2 = self.relu(self.e_conv2(e1))
		# p2 = self.maxpool(x2)
		e3 = self.relu(self.e_conv3(e2))
		# p3 = self.maxpool(x3)
        
		e2 = self.Maxpool(e2)
		e2 = self.relu(self.Con(e2))
		e2 = self.Up(e2)
        
		d3 = self.relu(self.e_conv4(e3))

		d2 = self.relu(self.e_conv5(torch.cat([e3,d3],1)))
		# x5 = self.upsample(x5)
		d1 = self.relu(self.e_conv6(torch.cat([e2,d2],1)))

		x_r = F.tanh(self.e_conv7(torch.cat([e1,d1],1)))
		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)
        
        
		x = x + r1*(torch.pow(x,2)-x)
		x = x + r2*(torch.pow(x,2)-x)
		x = x + r3*(torch.pow(x,2)-x)
		enhance_image_1 = x + r4*(torch.pow(x,2)-x)
		enhance_img_1.append(enhance_image_1)
		
		x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + r6*(torch.pow(x,2)-x)
		enhance_img.append(x)
        
		x = x + r7*(torch.pow(x,2)-x)
		enhance_img.append(x)
        
		enhance_image = x + r8*(torch.pow(x,2)-x)
		enhance_img.append(enhance_image)        
        
		r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
		r_img.append(r) 
        

		return enhance_img