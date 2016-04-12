require 'nn'
require 'torch'
require 'optim'

noutputs = 30

nfeats = 3
width = 32 
height = 32
ninputs = nfeats*width*height

nstates = {64, 64, 128}
filtsize = 5
poolsize = 2


function create_model()
	print('==> creating model')
	local model = nn.Sequential()

	local convnet = nn.Sequential()
	convnet:add(nn.SpatialFullConvolution(nfeats, nstates[1], filtsize, filtsize))
	convnet:add(nn.ReLU())
	convnet:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

	convnet:add(nn.SpatialFullConvolution(nstates[1], nstates[2], filtsize, filtsize))
	convnet:add(nn.ReLU())
	convnet:add(nn.SpatialMaxPooling(poolsize, poolsize, poolsize, poolsize))

	convnet:add(nn.View(nstates[2]*filtsize*filtsize))
	convnet:add(nn.Dropout(0.5))
	convnet:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
	convnet:add(nn.Linear(nstates[3], noutputs)) 

	local onehot = nn.Sequential()
	onehot:add(nn.Linear(9, 9)) 

	model:add(nn.ParallelTable())
	model:add(onehot)
	model:add(convnet)
	model:add(nn.JoinTable(1))

	-- Add more layers here for deconvolution

	print(model)
	return model

end

function define_criterion()
	print('==> define loss')
end

function get_action_data(filename)
	
end

function get_frame_data(filename)

end

function make_training_data()

end

function train()

end

function test()

end

model = create_model()

--[[
if opt.visualize then
   if opt.model == 'convnet' then
      if itorch then
	 print '==> visualizing ConvNet filters'
	 print('Layer 1 filters:')
	 itorch.image(model:get(1).weight)
	 print('Layer 2 filters:')
	 itorch.image(model:get(5).weight)
      else
	 print '==> To visualize filters, start the script in itorch notebook'
      end
   end
end
--]]

